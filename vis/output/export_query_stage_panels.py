"""Export three camera stages and a point-cloud BEV from an existing trace SVG.

Reuse vector projection coordinates, rather than fitting curves or rerunning the
model. The source trace and all unrelated panels remain untouched.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
NS = {'s': 'http://www.w3.org/2000/svg'}
QUERY_COLOR = '#00ff66'
EXPORT_DPI = 600
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10})


def trace_camera_curves(source, image_size):
    """Invert Matplotlib's equal-aspect camera transform without resampling.

    The trace uses adjustable='datalim': its camera viewport is wider than the
    image. Recover the true image origin from the centered viewport and its
    vertical scale, rather than stretching curves to the whole viewport.
    """
    root = ET.parse(source).getroot()
    width, height = image_size
    stages = []
    for axis in root.findall('.//s:g', NS):
        if not axis.get('id', '').startswith('axes_'):
            continue
        if axis.find('.//s:image', NS) is None:
            continue
        titles = [''.join(t.itertext()) for t in axis.findall('./s:g/s:text', NS)]
        title = next((t for t in titles if t.startswith(('Q0:', 'Iteration '))), None)
        if title is None:
            continue
        paths = [p for g in axis.findall('./s:g', NS)
                 if g.get('id', '').startswith('line2d_')
                 for p in g.findall('./s:path', NS)
                 if 'stroke:' in p.get('style', '')
                 and 'stroke-dasharray' not in p.get('style', '')]
        if not paths:
            raise ValueError(f'No tracked curves in {title}')
        curves = []
        for path in paths:
            clip = re.fullmatch(r'url\(#([^)]*)\)', path.get('clip-path', ''))
            if clip is None:
                raise ValueError('Expected clipped Matplotlib camera path')
            rect = root.find(f'.//s:clipPath[@id="{clip[1]}"]/s:rect', NS)
            x, y, w, h = (float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
            scale = h / height
            left = x + w / 2 - width * scale / 2
            tokens = path.get('d').split()
            points = []
            for index in range(0, len(tokens), 3):
                command, px, py = tokens[index:index + 3]
                if command not in ('M', 'L'):
                    raise ValueError(f'Unsupported path command: {command}')
                if command == 'M' and points:
                    points.append((np.nan, np.nan))
                points.append(((float(px) - left) / scale,
                               (float(py) - y) / scale))
            curves.append(np.asarray(points))
        stages.append((title, curves))
    if len(stages) != 3:
        raise ValueError(f'Expected Q0 and two iterations; found {len(stages)}')
    if len({len(curves) for _, curves in stages}) != 1:
        raise ValueError('Tracked curve counts differ between stages')
    return stages


def save_panel(fig, output, qa_dir):
    """Audit the final render, keeping only the requested SVG in delivery."""
    fig.canvas.draw()
    audit = Path('/root/.codex/skills/nature-figure/scripts')
    if audit.is_dir():
        sys.path.insert(0, str(audit))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=qa_dir / f'{output.stem}.alignment.json', strict=True)
    pdf = qa_dir / f'{output.stem}.pdf'
    fig.savefig(pdf, dpi=EXPORT_DPI)
    if audit.is_dir():
        for name in ('audit_pdf_text.py', 'audit_figure_collisions.py'):
            command = [sys.executable, str(audit / name), str(pdf)]
            if name == 'audit_figure_collisions.py':
                command += ['--json-out', str(qa_dir / f'{output.stem}.collision-audit.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(result.stdout.strip())
    fig.savefig(output, dpi=EXPORT_DPI)
    # Temporary preview uses the identical figure and selected Python backend.
    fig.savefig(qa_dir / f'{output.stem}.png', dpi=300)
    plt.close(fig)


def camera_panel(image, curves):
    height, width = image.shape[:2]
    # One SVG unit per source image pixel; preserve the entire raw photograph.
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=300)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.imshow(image, extent=(0, width, height, 0), interpolation='none')
    for curve in curves:
        axis.plot(curve[:, 0], curve[:, 1], color=QUERY_COLOR,
                  linewidth=6, solid_capstyle='round', solid_joinstyle='round')
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.set_axis_off()
    axis.legend(handles=[Line2D([0], [0], color=QUERY_COLOR, lw=6, label='Pred')],
                loc='upper right', fontsize=32, borderaxespad=.6,
                framealpha=.9, facecolor='white', edgecolor='#cccccc')
    return fig


def point_cloud_panel(ply_path, reference):
    # Match both the reference page and its physical plot-area rectangle.
    root = ET.parse(reference).getroot()
    _, _, width, height = map(float, root.get('viewBox').split())
    rect = root.find('.//s:clipPath/s:rect', NS)
    x, y, w, h = (float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
    fig = plt.figure(figsize=(width / 72, height / 72))
    axis = fig.add_axes([x / width, (height - y - h) / height, w / width, h / height])
    vertices = PlyData.read(ply_path)['vertex'].data
    valid = np.isfinite(vertices['x']) & np.isfinite(vertices['y'])
    rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]) / 255.
    axis.scatter(vertices['x'][valid], vertices['y'][valid], c=rgb[valid],
                 s=.8, linewidths=0, rasterized=True, zorder=2)
    axis.set(xlim=(-25, 25), ylim=(0, 100), xlabel='x / m', ylabel='y / m', aspect='equal')
    axis.set_facecolor('#f5f5f5')
    axis.set_axisbelow(True)
    axis.grid(True, alpha=.35)
    axis.tick_params(labelsize=9)
    print(f'PLY: {len(vertices)} vertices, {valid.sum()} finite xy; '
          f'{np.count_nonzero(valid & (vertices["x"] >= -25) & (vertices["x"] <= 25) & (vertices["y"] >= 0) & (vertices["y"] <= 100))} inside viewport; no downsampling')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-svg', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--reference-ply', type=Path, required=True)
    parser.add_argument('--bev-reference', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path,
                        help='Optional QA folder outside the sample output directory.')
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert('RGB'))
    stages = trace_camera_curves(args.source_svg, (image.shape[1], image.shape[0]))
    output = args.source_svg.parent
    with tempfile.TemporaryDirectory(prefix='query-panels-qa-') as temporary:
        qa_dir = args.qa_dir or Path(temporary)
        qa_dir.mkdir(parents=True, exist_ok=True)
        for index, (title, curves) in enumerate(stages):
            path = output / f'{index + 1:02d}_{"q0" if index == 0 else f"iteration_{index}"}_camera.svg'
            save_panel(camera_panel(image, curves), path, qa_dir)
            print(f'Saved: {path} ({len(curves)} tracked curves)')
        path = output / '04_pointcloud_bev.svg'
        save_panel(point_cloud_panel(args.reference_ply, args.bev_reference), path, qa_dir)
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
