"""Read the user's official-validation scene list without mixing training rows."""
import re
from pathlib import Path


GROUPS = {'正常': 'normal', '遮挡': 'occlusion', '上下坡': 'slope',
          '交叉路口': 'intersection', '弯道': 'curve', '汇聚分叉': 'merge_split',
          '夜间': 'night', '逆光': 'backlight', '极端': 'extreme', '复杂': 'complex'}
HARD_GROUPS = ('slope', 'intersection', 'curve', 'merge_split')


def read_validation_scenarios(path):
    text = Path(path).read_text()
    marker = re.search(r'^##\s+验证\b.*$', text, flags=re.MULTILINE)
    if marker is None:
        raise ValueError('Scene list must contain a ## 验证 section')
    selected, seen, group = [], set(), None
    for line in text[marker.end():].splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r'-{3,}', line) or line.startswith('## '):
            break
        if line.startswith('#'):
            continue
        if line in GROUPS:
            group = GROUPS[line]
            continue
        if re.fullmatch(r'data/openlane/images/validation/segment-[^/]+/?', line):
            # Directory references identify a scene, not an additional frame.
            continue
        match = re.fullmatch(r'data/openlane/images/validation/(segment-[^/]+)/(\d+)\.jpg', line)
        if match is None or group is None:
            raise ValueError(f'Unexpected validation scene-list entry: {line}')
        segment, frame = match.groups()
        if (segment, frame) in seen:
            raise ValueError(f'Duplicate validation sample: {line}')
        seen.add((segment, frame))
        selected.append(dict(group=group, segment=segment, frame=frame, image=line))
    if not selected:
        raise ValueError('Validation scene list is empty')
    return selected
