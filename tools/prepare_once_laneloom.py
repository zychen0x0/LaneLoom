"""Prepare the p0+p1 ONCE-3DLanes image/LiDAR subset for LaneLoom."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


class _ConcatenatedReader:
    """Read split archive parts as one forward-only byte stream."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def _next_file(self):
        if self.current is not None:
            self.current.close()
        try:
            self.current = open(next(self.paths), 'rb')
        except StopIteration:
            self.current = None

    def read(self, size=-1):
        chunks = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.current is None:
                self._next_file()
                if self.current is None:
                    break
            chunk = self.current.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
            else:
                self._next_file()
                if self.current is None:
                    break
        return b''.join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


def _safe_relative(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe archive member: {name}')
    return Path(*relative.parts)


def _extract_zip(zip_path, destination, overwrite=False):
    print(f'Extracting labels: {zip_path}')
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or '/.nfs' in f'/{info.filename}':
                continue
            relative = _safe_relative(info.filename)
            target = destination / relative
            if target.is_file() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, 'wb') as output:
                shutil.copyfileobj(source, output)


def _extract_tar_members(paths, destination, wanted, marker,
                         overwrite=False):
    paths = [Path(path) for path in paths]
    if marker.is_file() and not overwrite:
        print(f'Skipping completed archive: {marker.name}')
        return
    if not paths:
        raise FileNotFoundError('No archive parts matched the requested input')
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing archive parts: '
            + ', '.join(str(path) for path in missing))

    print('Scanning archive: ' + ', '.join(path.name for path in paths))
    extracted = 0
    reader = _ConcatenatedReader(paths)
    try:
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in wanted:
                    continue
                relative = _safe_relative(member.name)
                target = destination / relative
                if target.is_file() and not overwrite:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, open(target, 'wb') as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f'  extracted {extracted} files')
    finally:
        reader.close()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f'extracted={extracted}\n', encoding='utf-8')


def _read_split(path):
    return [
        line.strip().lstrip('/')
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _image_members(entries):
    return {f'data/{entry}' for entry in entries}


def _lidar_members(entries):
    return {
        'data/' + entry.replace('/cam01/', '/lidar_roof/').replace(
            '.jpg', '.bin')
        for entry in entries
    }


def _info_members(entries):
    sequences = {entry.split('/', 1)[0] for entry in entries}
    return {f'data/{sequence}/{sequence}.json' for sequence in sequences}


def _write_split(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(f'/{entry}\n' for entry in entries), encoding='utf-8')


def _sample_paths(root, entry, annotation_split):
    sequence, _, filename = entry.split('/')
    stem = Path(filename).stem
    return (
        root / '3D_images/data' / entry,
        root / '3D_lidars/data' / sequence / 'lidar_roof' / f'{stem}.bin',
        root / '3D_infos/data' / sequence / f'{sequence}.json',
        None if annotation_split is None else
        root / 'ONCE_3DLanes' / annotation_split / sequence / 'cam01'
        / f'{stem}.json',
    )


def _verify(root):
    split_specs = (
        ('train_p0p1', 'train'),
        ('val', 'val'),
        ('test', None),
    )
    failures = []
    for split_name, annotation_split in split_specs:
        split_path = root / 'LaneLoom_splits' / f'{split_name}.txt'
        if not split_path.is_file():
            failures.append(str(split_path))
            continue
        entries = _read_split(split_path)
        for entry in entries:
            image, lidar, info, label = _sample_paths(
                root, entry, annotation_split)
            for path in (image, lidar, info, label):
                if path is not None and not path.is_file():
                    failures.append(str(path))
                    if len(failures) >= 20:
                        break
            if lidar.is_file() and lidar.stat().st_size % 16:
                failures.append(f'{lidar} (not N x 4 float32)')
            if len(failures) >= 20:
                break
        print(f'{split_name}: {len(entries)} samples')
    if failures:
        raise RuntimeError(
            'ONCE preparation verification failed (first paths):\n'
            + '\n'.join(failures))
    print('ONCE preparation verification passed.')


def prepare(root, overwrite=False):
    root = root.expanduser().resolve()
    labels_root = root / 'ONCE_3DLanes'
    _extract_zip(root / 'ONCE_3DLanes.zip', labels_root, overwrite=overwrite)
    official_splits = {
        name: _read_split(labels_root / 'list' / f'{name}.txt')
        for name in ('train', 'val', 'test')
    }
    marker_root = root / 'LaneLoom_splits/.prepare_markers'

    image_root = root / '3D_images'
    train_image_members = _image_members(official_splits['train'])
    for partition in ('p0', 'p1'):
        archive = image_root / f'raw_cam01_{partition}.tar'
        _extract_tar_members(
            [archive], image_root, train_image_members,
            marker_root / f'raw_cam01_{partition}.complete', overwrite)
    for split_name in ('val', 'test'):
        _extract_tar_members(
            [image_root / f'{split_name}_cam01.tar'], image_root,
            _image_members(official_splits[split_name]),
            marker_root / f'{split_name}_cam01.complete', overwrite)

    lidar_root = root / '3D_lidars'
    train_lidar_members = _lidar_members(official_splits['train'])
    for partition in ('p0', 'p1'):
        parts = sorted(lidar_root.glob(
            f'raw_lidar_{partition}.tar.part*'))
        _extract_tar_members(
            parts, lidar_root, train_lidar_members,
            marker_root / f'raw_lidar_{partition}.complete', overwrite)
    for split_name in ('val', 'test'):
        _extract_tar_members(
            [lidar_root / f'{split_name}_lidar.tar'], lidar_root,
            _lidar_members(official_splits[split_name]),
            marker_root / f'{split_name}_lidar.complete', overwrite)

    train_p0p1 = []
    for entry in official_splits['train']:
        image, lidar, _, _ = _sample_paths(root, entry, 'train')
        if image.is_file() and lidar.is_file():
            train_p0p1.append(entry)
    _write_split(root / 'LaneLoom_splits/train_p0p1.txt', train_p0p1)
    _write_split(root / 'LaneLoom_splits/val.txt', official_splits['val'])
    _write_split(root / 'LaneLoom_splits/test.txt', official_splits['test'])

    info_root = root / '3D_infos'
    wanted_info = _info_members(
        train_p0p1 + official_splits['val'] + official_splits['test'])
    for archive_name in (
            'raw_small_infos.tar', 'raw_medium_infos.tar',
            'val_infos.tar', 'test_infos.tar'):
        _extract_tar_members(
            [info_root / archive_name], info_root, wanted_info,
            marker_root / f'{archive_name}.complete', overwrite)

    (root / 'pred_dir').mkdir(parents=True, exist_ok=True)
    _verify(root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ONCE-3DLanes p0+p1 for LaneLoom.')
    parser.add_argument('--once-root', type=Path, required=True)
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only validate an already prepared directory.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Replace extracted files and rescan completed archives.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        _verify(args.once_root.expanduser().resolve())
    else:
        prepare(args.once_root, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
