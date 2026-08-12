from importlib.util import module_from_spec, spec_from_file_location
import io
from pathlib import Path
import tarfile

import pytest


TOOL_PATH = (
    Path(__file__).resolve().parents[2] / 'tools/prepare_once_laneloom.py')
SPEC = spec_from_file_location('prepare_once_laneloom', TOOL_PATH)
TOOL = module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_extract_tar_members_streams_across_split_parts(tmp_path):
    payload = _tar_bytes({
        'data/000001/lidar_roof/100.bin': b'wanted',
        'data/000002/lidar_roof/200.bin': b'ignored',
    })
    split = len(payload) // 3
    parts = []
    for index, content in enumerate(
            (payload[:split], payload[split:2 * split], payload[2 * split:])):
        path = tmp_path / f'raw.tar.part{index}'
        path.write_bytes(content)
        parts.append(path)

    output = tmp_path / 'output'
    marker = tmp_path / 'complete'
    TOOL._extract_tar_members(
        parts,
        output,
        {'data/000001/lidar_roof/100.bin'},
        marker)

    assert (
        output / 'data/000001/lidar_roof/100.bin').read_bytes() == b'wanted'
    assert not (output / 'data/000002/lidar_roof/200.bin').exists()
    assert marker.is_file()


def test_archive_member_rejects_parent_traversal():
    with pytest.raises(ValueError, match='Unsafe archive member'):
        TOOL._safe_relative('../../outside')


def test_extract_tar_members_reports_missing_split_parts(tmp_path):
    with pytest.raises(FileNotFoundError, match='No archive parts'):
        TOOL._extract_tar_members(
            [], tmp_path / 'output', set(), tmp_path / 'complete')
