"""Tests for the narrow model artifact metadata helpers."""

import base64

import pytest

from library.utils.model_metadata import file_to_data_url


def test_file_to_data_url(tmp_path):
    test_png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xff\xff\xff\x00"
        b"\x00\x00\x04\x00\x01\x9d\xb3\xa7c\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    thumbnail = tmp_path / "thumbnail.png"
    thumbnail.write_bytes(test_png_data)

    data_url = file_to_data_url(str(thumbnail))

    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == test_png_data


def test_file_to_data_url_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        file_to_data_url(str(tmp_path / "missing.png"))
