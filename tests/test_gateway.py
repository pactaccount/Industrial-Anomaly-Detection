"""The demo gateway must not actuate a reject during a dry run."""

from scripts.gateway import Gateway


def test_dry_run_does_not_create_reject_marker(tmp_path):
    image = tmp_path / "part.png"
    image.write_bytes(b"image")
    gateway = Gateway(
        api_url="http://localhost:8000",
        api_key="test",
        model_id="model",
        out_dir=tmp_path / "verdicts",
        on_reject="marker",
        dry_run=True,
    )
    gateway.signal_reject(image, {"image_score": 9.0, "defects": []})
    assert not image.with_suffix(".REJECT").exists()
