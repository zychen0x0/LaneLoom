"""Export the eight template/refinement geometry panels from an existing trace.

Recover data coordinates from SVG tick calibration, preserving every source
curve and visibility gap. No inference, fitting, or GT-based reselection occurs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.lines import Line2D

from export_query_stage_panels import NS, QUERY_COLOR, plt, save_panel

GT_COLOR = '#ff3b30'


def inverse_axis_transform(axis, coordinate):
    """Calibrate a linear source axis against its explicit tick marks."""
    positions, values = [], []
    for group in axis.findall('./s:g', NS):
        if not group.get('id', '').startswith('matplotlib.axis_'):
            continue
        for tick in group.findall('./s:g', NS):
            if not tick.get('id', '').startswith(f'{coordinate}tick_'):
                continue
            text = tick.find('.//s:text', NS)
            marker = tick.find('.//s:use', NS)
            if text is None or marker is None:
                raise ValueError('Expected editable tick text and a tick marker')
            values.append(float(''.join(text.itertext()).replace('−', '-')))
            positions.append(float(marker.get(coordinate)))
    if len(values) < 2:
        raise ValueError(f'Need at least two ticks to calibrate {coordinate}')
    scale = (values[-1] - values[0]) / (positions[-1] - positions[0])
    offset = values[0] - positions[0] * scale
    np.testing.assert_allclose(np.asarray(positions) * scale + offset, values, atol=2e-6)
    return scale, offset


def path_points(path):
    tokens = (path.get('d') or '').split()
    if len(tokens) % 3:
        raise ValueError('Expected source polylines with explicit M/L commands')
    points = []
    for index in range(0, len(tokens), 3):
        command, x, y = tokens[index:index + 3]
        if command not in ('M', 'L'):
            raise ValueError(f'Unsupported command {command}')
        if command == 'M' and points:
            points.append((np.nan, np.nan))
        points.append((float(x), float(y)))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def source_panel(root, index):
    axis = root.find(f'.//s:g[@id="axes_{index}"]', NS)
    if axis is None:
        raise ValueError(f'Missing geometry panel {index}')
    transforms = [inverse_axis_transform(axis, c) for c in ('x', 'y')]
    curves = []
    limits = None
    for group in axis.findall('./s:g', NS):
        if not group.get('id', '').startswith('line2d_'):
            continue
        for path in group.findall('./s:path', NS):
            style = dict(item.strip().split(':', 1) for item in path.get('style', '').split(';') if ':' in item)
            style = {k.strip(): v.strip() for k, v in style.items()}
            if 'stroke' not in style:
                continue
            points = path_points(path)
            for dimension, (scale, offset) in enumerate(transforms):
                points[:, dimension] = points[:, dimension] * scale + offset
            role = ('gt' if 'stroke-dasharray' in style else
                    'background' if float(style.get('stroke-opacity', 1)) < 1 else 'query')
            curves.append(dict(points=points, role=role, color=style['stroke'],
                               alpha=float(style.get('stroke-opacity', 1))))
            if limits is None:
                clip_id = path.get('clip-path').removeprefix('url(#').removesuffix(')')
                rect = root.find(f'.//s:clipPath[@id="{clip_id}"]/s:rect', NS)
                x, y, w, h = (float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
                sx, ox = transforms[0]
                sy, oy = transforms[1]
                limits = ((x * sx + ox, (x + w) * sx + ox),
                          ((y + h) * sy + oy, y * sy + oy))
    return curves, limits


def reference_layout(path):
    root = ET.parse(path).getroot()
    _, _, width, height = map(float, root.get('viewBox').split())
    rect = root.find('.//s:clipPath/s:rect', NS)
    x, y, w, h = (float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
    return width, height, x, y, w, h


def template_palette(root):
    """Use Pred green for the actual source families of the tracked queries."""
    templates, _ = source_panel(root, 5)
    q0, _ = source_panel(root, 6)
    background = [curve for curve in q0 if curve['role'] == 'background']
    query_ids = []
    for legend in root.findall('.//s:g', NS):
        if not legend.get('id', '').startswith('legend_'):
            continue
        for text in legend.findall('.//s:text', NS):
            label = ''.join(text.itertext())
            if label.startswith('q') and label[1:].isdigit():
                query_ids.append(int(label[1:]))
    if not query_ids or len(background) % len(templates):
        raise ValueError('Cannot establish tracked query template provenance')
    seed_count = len(background) // len(templates)
    used = {query // seed_count for query in query_ids}
    for query in query_ids:
        if background[query]['color'] != templates[query // seed_count]['color']:
            raise ValueError('Query ID and source template family disagree')
    unused_colors = iter(('#0072b2', '#9467bd', '#7f7f7f', '#8c6d31', '#17becf'))
    colors = {curve['color']: QUERY_COLOR if i in used else next(unused_colors)
              for i, curve in enumerate(templates)}
    return colors, used


def clip_height_curve(points, z_limits):
    """Clip visible line segments to the shared z viewport, retaining gaps."""
    clipped = []
    for start, end in zip(points[:-1], points[1:]):
        if not np.isfinite([start, end]).all():
            continue
        delta = end[1] - start[1]
        if delta == 0:
            if not z_limits[0] <= start[1] <= z_limits[1]:
                continue
            first, last = 0., 1.
        else:
            enter, leave = sorted((np.asarray(z_limits) - start[1]) / delta)
            first, last = max(0., enter), min(1., leave)
            if first > last:
                continue
        a, b = start + first * (end - start), start + last * (end - start)
        if clipped and not np.array_equal(clipped[-1], a):
            clipped.append(np.array([np.nan, np.nan]))
        clipped.extend((a, b))
    return np.asarray(clipped).reshape(-1, 2)


def geometry_panel(curves, limits, reference, template=False, height=False):
    width, page_height, left, top, plot_width, plot_height = reference
    bottom = page_height - top - plot_height
    if height:
        # Match the original height panels' 2:1 plot ratio, and the BEV width.
        plot_height = plot_width / 2
        page_height = top + plot_height + bottom
    fig = plt.figure(figsize=(width / 72, page_height / 72))
    axis = fig.add_axes([left / width, bottom / page_height,
                         plot_width / width, plot_height / page_height])
    for curve in curves:
        points, role = curve['points'], curve['role']
        if role == 'background':
            continue  # Show only the tracked final-NMS survivors at Q0.
        if height:
            points = clip_height_curve(points, limits[1])
        color = (curve['color'] if template else
                 GT_COLOR if role == 'gt' else QUERY_COLOR)
        axis.plot(points[:, 0], points[:, 1], color=color,
                  lw=2.5,
                  alpha=curve['alpha'], linestyle='-',
                  zorder=2 if role == 'gt' else 3)
    if height:
        axis.set(xlim=(0, 100), ylim=limits[1], xlabel='y / m', ylabel='z / m')
        axis.set_yticks([-.6, -.4, -.2, 0])
    else:
        axis.set(xlim=(-25, 25), ylim=(0, 100), xlabel='x / m', ylabel='y / m')
        axis.set_aspect('equal', adjustable='box')
    axis.set_facecolor('#f5f5f5')
    axis.set_axisbelow(True)
    axis.grid(True, alpha=.35)
    axis.tick_params(labelsize=9)
    if not template:
        axis.legend(handles=[Line2D([0], [0], color=GT_COLOR, lw=2.5, label='GT'),
                             Line2D([0], [0], color=QUERY_COLOR, lw=2.5, label='Pred')],
                    loc='upper right', fontsize=10, framealpha=1,
                    facecolor='white', edgecolor='#cccccc')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-svg', type=Path, required=True)
    parser.add_argument('--bev-reference', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path)
    args = parser.parse_args()
    root = ET.parse(args.source_svg).getroot()
    reference = reference_layout(args.bev_reference)
    height_limits = source_panel(root, 10)[1][1]
    template_colors, used_templates = template_palette(root)
    print('Pred source template numbers:', sorted(i + 1 for i in used_templates))
    names = ['05_relative_templates_bev', '06_q0_bev', '07_iteration_1_bev',
             '08_iteration_2_bev', '09_template_height', '10_q0_height',
             '11_iteration_1_height', '12_iteration_2_height']
    with tempfile.TemporaryDirectory(prefix='query-geometry-qa-') as temporary:
        qa_dir = args.qa_dir or Path(temporary)
        qa_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names):
            curves, limits = source_panel(root, index + 5)
            if index >= 4:
                limits = (limits[0], height_limits)
            template = index in (0, 4)
            if template:
                for curve in curves:
                    curve['color'] = template_colors[curve['color']]
            expected = 5 if template else 513 if index == 1 else 8
            if len(curves) != expected:
                raise ValueError(f'{name}: expected {expected} source curves, got {len(curves)}')
            fig = geometry_panel(curves, limits, reference, template, height=index >= 4)
            output = args.source_svg.parent / f'{name}.svg'
            save_panel(fig, output, qa_dir)
            print(f'Saved: {output}; source paths={len(curves)}, '
                  f'empty={sum(not len(c["points"]) for c in curves)}; source limits={limits}', flush=True)


if __name__ == '__main__':
    main()
