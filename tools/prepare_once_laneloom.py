"""Prepare ONCE-3DLanes images, LiDAR, and annotations for LaneLoom."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _extract_zip(zip_path, destination, overwrite=False, min_free_bytes=0):
    print(f'Extracting labels: {zip_path}')
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or '/.nfs' in f'/{info.filename}':
                continue
            relative = _safe_relative(info.filename)
            target = destination / relative
            if target.is_file() and not overwrite \
                    and target.stat().st_size == info.file_size:
                continue
            partial = target.with_name(f'.{target.name}.laneloom-part')
            if partial.is_file():
                partial.unlink()
            free_bytes = shutil.disk_usage(destination).free
            required_bytes = int(info.file_size) + int(min_free_bytes)
            if free_bytes < required_bytes:
                raise RuntimeError(
                    f'Insufficient free space under {destination}: '
                    f'{free_bytes / 1024**3:.1f} GiB available, '
                    f'{min_free_bytes / 1024**3:.1f} GiB reserved.')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(partial, 'wb') as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)


def _wanted_digest(wanted):
    digest = hashlib.sha256()
    for name in sorted(wanted):
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def _source_signatures(paths):
    return [
        dict(name=path.name, size=path.stat().st_size,
             mtime_ns=path.stat().st_mtime_ns)
        for path in paths
    ]


def _marker_is_complete(marker, paths, wanted, destination):
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding='utf-8'))
        if payload.get('version') != 1:
            return False
        if payload.get('wanted_sha256') != _wanted_digest(wanted):
            return False
        if payload.get('sources') != _source_signatures(paths):
            return False
        for relative, size in payload.get('outputs', []):
            target = destination / _safe_relative(relative)
            if not target.is_file() or target.stat().st_size != int(size):
                return False
        return True
    except (KeyError, OSError, TypeError, UnicodeError, ValueError,
            json.JSONDecodeError):
        return False


def _write_marker(marker, paths, wanted, outputs):
    payload = dict(
        version=1,
        sources=_source_signatures(paths),
        wanted_sha256=_wanted_digest(wanted),
        outputs=sorted(outputs.items()),
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    partial = marker.with_name(f'.{marker.name}.laneloom-part')
    if partial.is_file():
        partial.unlink()
    partial.write_text(
        json.dumps(payload, separators=(',', ':')) + '\n', encoding='utf-8')
    partial.replace(marker)


def _validate_tar_terminator(paths):
    required = 1024
    chunks = []
    for path in reversed(paths):
        with open(path, 'rb') as handle:
            size = path.stat().st_size
            take = min(size, required)
            handle.seek(size - take)
            chunks.append(handle.read(take))
        required -= take
        if required == 0:
            break
    tail = b''.join(reversed(chunks))
    if len(tail) < 1024 or not tail.endswith(b'\0' * 1024):
        raise RuntimeError(
            'ONCE tar archive is incomplete or missing its final split part: '
            + ', '.join(path.name for path in paths))


def _extract_tar_members(paths, destination, wanted, marker,
                         overwrite=False, min_free_bytes=0):
    paths = [Path(path) for path in paths]
    if not paths:
        raise FileNotFoundError('No archive parts matched the requested input')
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing archive parts: '
            + ', '.join(str(path) for path in missing))
    if _marker_is_complete(
            marker, paths, wanted, destination) and not overwrite:
        print(f'Skipping completed archive: {marker.name}')
        return
    _validate_tar_terminator(paths)

    print('Scanning archive: ' + ', '.join(path.name for path in paths))
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    outputs = {}
    reader = _ConcatenatedReader(paths)
    try:
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in wanted:
                    continue
                relative = _safe_relative(member.name)
                # ONCE archives wrap every extracted path in ``data/``.  The
                # dataset roots already describe the contents, so omit that
                # redundant archive-level directory on disk.
                target = destination / relative.relative_to('data')
                target_relative = target.relative_to(destination).as_posix()
                if target.is_file() and not overwrite \
                        and target.stat().st_size == member.size:
                    outputs[target_relative] = int(member.size)
                    continue
                partial = target.with_name(f'.{target.name}.laneloom-part')
                if partial.is_file():
                    partial.unlink()
                free_bytes = shutil.disk_usage(destination).free
                required_bytes = int(member.size) + int(min_free_bytes)
                if free_bytes < required_bytes:
                    raise RuntimeError(
                        f'Insufficient free space under {destination}: '
                        f'{free_bytes / 1024**3:.1f} GiB available, '
                        f'{min_free_bytes / 1024**3:.1f} GiB reserved.')
                source = archive.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, open(partial, 'wb') as output:
                    shutil.copyfileobj(source, output)
                partial.replace(target)
                outputs[target_relative] = int(member.size)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f'  extracted {extracted} files')
    finally:
        reader.close()
    _write_marker(marker, paths, wanted, outputs)


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


def _find_archive(roots, name):
    matches = [root / name for root in roots if (root / name).is_file()]
    if len(matches) > 1:
        raise RuntimeError(
            f'Duplicate ONCE archive {name}: '
            + ', '.join(str(path) for path in matches))
    return matches[0] if matches else roots[0] / name


def _find_archive_parts(roots, pattern):
    matches = {}
    for root in roots:
        for path in root.glob(pattern):
            if path.name in matches:
                raise RuntimeError(
                    f'Duplicate ONCE archive part {path.name}: '
                    f'{matches[path.name]}, {path}')
            matches[path.name] = path
    paths = [matches[name] for name in sorted(matches)]
    suffixes = [path.name.rsplit('.part', 1)[-1] for path in paths]
    if suffixes:
        width = len(suffixes[0])
        indices = []
        for suffix in suffixes:
            if len(suffix) != width or not suffix.isalpha() \
                    or not suffix.islower():
                raise RuntimeError(
                    f'Invalid ONCE archive part suffix: {suffix}')
            index = 0
            for character in suffix:
                index = index * 26 + ord(character) - ord('a')
            indices.append(index)
        if indices != list(range(len(indices))):
            raise RuntimeError(
                f'Incomplete ONCE archive part sequence for {pattern}: '
                + ', '.join(path.name for path in paths))
    return paths


def _flatten_data_directory(root):
    """Move an older ``root/data/*`` extraction one level upward."""
    nested_root = root / 'data'
    if not nested_root.is_dir():
        return

    children = list(nested_root.iterdir())
    conflicts = [root / child.name for child in children
                 if (root / child.name).exists()]
    if conflicts:
        raise FileExistsError(
            'Cannot flatten legacy ONCE data because targets already exist: '
            + ', '.join(str(path) for path in conflicts[:20]))

    print(f'Flattening legacy extraction: {nested_root}')
    for child in children:
        child.rename(root / child.name)
    nested_root.rmdir()


def _sample_paths(root, entry, annotation_split, labels_root):
    sequence, _, filename = entry.split('/')
    stem = Path(filename).stem
    return (
        root / '3D_images' / entry,
        root / '3D_lidars' / sequence / 'lidar_roof' / f'{stem}.bin',
        root / '3D_infos' / sequence / f'{sequence}.json',
        None if annotation_split is None else
        labels_root / annotation_split / sequence / 'cam01' / f'{stem}.json',
    )


def _verify(root, full_train=False):
    labels_root = root if full_train else root / 'ONCE_3DLanes'
    split_root = root / ('list' if full_train else 'LaneLoom_splits')
    train_split = 'train' if full_train else 'train_p0p1'
    split_specs = (
        (train_split, 'train'),
        ('val', 'val'),
        ('test', None),
    )
    failures = []
    for split_name, annotation_split in split_specs:
        split_path = split_root / f'{split_name}.txt'
        if not split_path.is_file():
            failures.append(str(split_path))
            continue
        entries = _read_split(split_path)
        for entry in entries:
            image, lidar, info, label = _sample_paths(
                root, entry, annotation_split, labels_root)
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


def prepare(root, overwrite=False, archive_root=None,
            lidar_archive_dirs=None, min_free_bytes=0, full_train=False):
    root = root.expanduser().resolve()
    archive_root = Path(archive_root or root).expanduser().resolve()
    lidar_archive_roots = [
        Path(path).expanduser().resolve()
        for path in (lidar_archive_dirs or [archive_root / '3D_lidars'])
    ]

    print(f'ONCE archive root: {archive_root}')
    print(f'ONCE output root:  {root}')
    print('LiDAR archives:   '
          + ', '.join(str(path) for path in lidar_archive_roots))

    labels_root = root if full_train else root / 'ONCE_3DLanes'
    _extract_zip(
        archive_root / 'ONCE_3DLanes.zip', labels_root, overwrite=overwrite,
        min_free_bytes=min_free_bytes)
    official_splits = {
        name: _read_split(labels_root / 'list' / f'{name}.txt')
        for name in ('train', 'val', 'test')
    }
    marker_root = root / 'LaneLoom_splits/.prepare_markers'
    partitions = [f'p{index}' for index in range(10 if full_train else 2)]

    image_root = root / '3D_images'
    image_archive_root = archive_root / '3D_images'
    _flatten_data_directory(image_root)
    train_image_members = _image_members(official_splits['train'])
    for partition in partitions:
        archive = image_archive_root / f'raw_cam01_{partition}.tar'
        _extract_tar_members(
            [archive], image_root, train_image_members,
            marker_root / f'raw_cam01_{partition}.complete', overwrite,
            min_free_bytes)
    for split_name in ('val', 'test'):
        _extract_tar_members(
            [image_archive_root / f'{split_name}_cam01.tar'], image_root,
            _image_members(official_splits[split_name]),
            marker_root / f'{split_name}_cam01.complete', overwrite,
            min_free_bytes)

    lidar_root = root / '3D_lidars'
    _flatten_data_directory(lidar_root)
    train_lidar_members = _lidar_members(official_splits['train'])
    for partition in partitions:
        parts = _find_archive_parts(
            lidar_archive_roots, f'raw_lidar_{partition}.tar.part*')
        _extract_tar_members(
            parts, lidar_root, train_lidar_members,
            marker_root / f'raw_lidar_{partition}.complete', overwrite,
            min_free_bytes)
    for split_name in ('val', 'test'):
        _extract_tar_members(
            [_find_archive(
                lidar_archive_roots, f'{split_name}_lidar.tar')], lidar_root,
            _lidar_members(official_splits[split_name]),
            marker_root / f'{split_name}_lidar.complete', overwrite,
            min_free_bytes)

    train_entries = official_splits['train']
    if not full_train:
        train_entries = []
        for entry in official_splits['train']:
            image, lidar, _, _ = _sample_paths(
                root, entry, 'train', labels_root)
            if image.is_file() and lidar.is_file():
                train_entries.append(entry)
        _write_split(root / 'LaneLoom_splits/train_p0p1.txt', train_entries)
        _write_split(root / 'LaneLoom_splits/val.txt', official_splits['val'])
        _write_split(root / 'LaneLoom_splits/test.txt', official_splits['test'])

    info_root = root / '3D_infos'
    info_archive_root = archive_root / '3D_infos'
    _flatten_data_directory(info_root)
    wanted_info = _info_members(
        train_entries + official_splits['val'] + official_splits['test'])
    info_archives = [
        'raw_small_infos.tar', 'raw_medium_infos.tar',
        'val_infos.tar', 'test_infos.tar',
    ]
    if full_train:
        info_archives.append('raw_large_infos.tar')
    for archive_name in info_archives:
        _extract_tar_members(
            [info_archive_root / archive_name], info_root, wanted_info,
            marker_root / f'{archive_name}.complete', overwrite,
            min_free_bytes)

    _verify(root, full_train=full_train)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ONCE-3DLanes for LaneLoom.')
    parser.add_argument('--once-root', type=Path, required=True)
    parser.add_argument(
        '--full-train', action='store_true',
        help='Use p0-p9 and the main config layout. The default prepares '
             'the legacy p0+p1 subset.')
    parser.add_argument(
        '--archive-root', type=Path,
        help='Root containing the original ONCE archives; defaults to '
             '--once-root.')
    parser.add_argument(
        '--lidar-archive-dir', type=Path, action='append', default=None,
        help='Directory containing LiDAR tar files or split parts. Repeat '
             'when parts live on multiple disks; defaults to '
             '<archive-root>/3D_lidars.')
    parser.add_argument(
        '--min-free-gb', type=float, default=20.0,
        help='Stop before target-disk free space drops below this reserve '
             '(default: 20 GiB).')
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only validate an already prepared directory.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Replace extracted files and rescan completed archives.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_free_gb < 0:
        raise ValueError('--min-free-gb must be non-negative.')
    if args.verify_only:
        _verify(args.once_root.expanduser().resolve(), full_train=args.full_train)
    else:
        prepare(
            args.once_root,
            overwrite=args.overwrite,
            archive_root=args.archive_root,
            lidar_archive_dirs=args.lidar_archive_dir,
            min_free_bytes=int(args.min_free_gb * 1024**3),
            full_train=args.full_train)


if __name__ == '__main__':
    main()
