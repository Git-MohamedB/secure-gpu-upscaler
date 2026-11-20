"""Tests unitaires pour la verification d'integrite SHA-256 des poids."""
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_weights import WEIGHTS_NAME, download_weights


def test_existing_valid_weights_passes(tmp_path: Path):
    payload = b"valid-model-weights-payload"
    expected = hashlib.sha256(payload).hexdigest()
    target = tmp_path / WEIGHTS_NAME
    target.write_bytes(payload)

    with patch("scripts.download_weights.urllib.request.urlretrieve") as mock_fetch:
        out = download_weights(dest_dir=tmp_path, expected_sha256=expected)

    assert out == target
    mock_fetch.assert_not_called()


def test_existing_tampered_weights_raises(tmp_path: Path):
    target = tmp_path / WEIGHTS_NAME
    target.write_bytes(b"corrupted-payload")

    with pytest.raises(ValueError, match="SHA-256"):
        download_weights(dest_dir=tmp_path, expected_sha256="0" * 64)


def test_mocked_download_valid_checksum_atomic(tmp_path: Path):
    payload = b"downloaded-weights-ok"
    expected = hashlib.sha256(payload).hexdigest()

    def _fake_retrieve(_url, filename, _reporthook=None):
        Path(filename).write_bytes(payload)

    with patch("scripts.download_weights.urllib.request.urlretrieve", side_effect=_fake_retrieve):
        out = download_weights(dest_dir=tmp_path, expected_sha256=expected)

    assert out.exists()
    assert out.read_bytes() == payload
    assert not (tmp_path / f"{Path(WEIGHTS_NAME).stem}.pth.part").exists()


def test_mocked_download_wrong_checksum_cleans_part_file(tmp_path: Path):
    def _fake_retrieve(_url, filename, _reporthook=None):
        Path(filename).write_bytes(b"malicious-payload")

    with patch("scripts.download_weights.urllib.request.urlretrieve", side_effect=_fake_retrieve):
        with pytest.raises(ValueError, match="SHA-256"):
            download_weights(dest_dir=tmp_path, expected_sha256="f" * 64)

    assert not (tmp_path / WEIGHTS_NAME).exists()
    assert not (tmp_path / f"{Path(WEIGHTS_NAME).stem}.pth.part").exists()
