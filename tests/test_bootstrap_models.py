from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from agents.ml.face_blur.model_assets import YUNET_SHA256, bundled_yunet_path, verify_bundled_yunet
from agents.ml.face_blur.runpod.bootstrap_models import (
    BootstrapModelError,
    configure_face_blur_backend,
    verify_bundled_plate_model,
)


@pytest.fixture(autouse=True)
def _clean_face_env():
    keys = (
        "FACE_BLUR_DETECTOR",
        "FACE_BLUR_BACKEND",
        "FACE_BLUR_ONNX_MODEL_PATH",
        "FACE_BLUR_ONNX_SHA256",
        "FACE_BLUR_SCRFD_SHA256",
        "FACE_BLUR_ALLOW_NETWORK_MODELS",
        "PRIVACY_PLATE_ONNX_MODEL_PATH",
        "PRIVACY_PLATE_ONNX_SHA256",
    )
    original = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    yield
    for key in keys:
        os.environ.pop(key, None)
        if original[key] is not None:
            os.environ[key] = original[key]


def test_default_worker_selects_bundled_yunet_without_network():
    with mock.patch("urllib.request.urlretrieve") as fetch:
        resolved = configure_face_blur_backend("ztiles")
    assert resolved == "yunet"
    assert os.environ["FACE_BLUR_DETECTOR"] == "yunet"
    assert os.environ["FACE_BLUR_BACKEND"] == "opencv"
    fetch.assert_not_called()
    verify_bundled_yunet()
    verify_bundled_plate_model()


def test_missing_bundled_yunet_fails_closed(tmp_path, monkeypatch):
    path = bundled_yunet_path()
    if not path.is_file():
        pytest.skip("bundled yunet not present in checkout")
    backup = tmp_path / "yunet.onnx"
    backup.write_bytes(path.read_bytes())
    monkeypatch.setattr(
        "agents.ml.face_blur.model_assets.bundled_yunet_path",
        lambda: tmp_path / "missing.onnx",
    )
    with pytest.raises(BootstrapModelError):
        configure_face_blur_backend()


def test_corrupt_bundled_yunet_fails_closed(tmp_path, monkeypatch):
    path = bundled_yunet_path()
    if not path.is_file():
        pytest.skip("bundled yunet not present in checkout")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"x" * 64)
    monkeypatch.setattr("agents.ml.face_blur.model_assets.bundled_yunet_path", lambda: bad)
    with pytest.raises(BootstrapModelError):
        configure_face_blur_backend()


def test_scrfd_not_auto_selected_without_pinned_local_file():
    with mock.patch("agents.ml.face_blur.model_assets.ensure_scrfd_model", return_value=True) as scrfd:
        with mock.patch("urllib.request.urlretrieve") as fetch:
            resolved = configure_face_blur_backend()
    assert resolved == "yunet"
    scrfd.assert_not_called()
    fetch.assert_not_called()


def test_custom_onnx_requires_pinned_digest(tmp_path):
    model = tmp_path / "face.onnx"
    model.write_bytes(b"not-a-real-onnx")
    os.environ["FACE_BLUR_DETECTOR"] = "onnx"
    os.environ["FACE_BLUR_ONNX_MODEL_PATH"] = str(model)
    with pytest.raises(BootstrapModelError, match="sha256"):
        configure_face_blur_backend()


def test_custom_onnx_accepts_matching_digest(tmp_path):
    from agents.ml.face_blur.manifest import sha256_file

    model = tmp_path / "face.onnx"
    model.write_bytes(b"fake-onnx-bytes-for-test")
    os.environ["FACE_BLUR_DETECTOR"] = "onnx"
    os.environ["FACE_BLUR_ONNX_MODEL_PATH"] = str(model)
    os.environ["FACE_BLUR_ONNX_SHA256"] = sha256_file(model)
    resolved = configure_face_blur_backend()
    assert resolved == "onnx"
    assert os.environ["FACE_BLUR_BACKEND"] == "onnx"


def test_custom_plate_requires_digest(tmp_path):
    plate = tmp_path / "plate.onnx"
    plate.write_bytes(b"plate")
    os.environ["PRIVACY_PLATE_ONNX_MODEL_PATH"] = str(plate)
    os.environ["PRIVACY_PLATE_ONNX_SHA256"] = "deadbeef"
    with pytest.raises(BootstrapModelError, match="mismatch"):
        configure_face_blur_backend()


def test_custom_plate_without_digest_fails_closed(tmp_path):
    plate = tmp_path / "plate.onnx"
    plate.write_bytes(b"plate")
    os.environ["PRIVACY_PLATE_ONNX_MODEL_PATH"] = str(plate)
    with pytest.raises(BootstrapModelError, match="sha256_required"):
        configure_face_blur_backend()


def test_verify_bundled_yunet_matches_provenance():
    path = bundled_yunet_path()
    if not path.is_file():
        pytest.skip("bundled yunet not present")
    from agents.ml.face_blur.manifest import sha256_file

    assert sha256_file(path) == YUNET_SHA256
    verify_bundled_yunet()
