import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "e156-submission" / "config.json"
ENGINE_PATH = REPO_ROOT / "alburhan" / "engines" / "metafrontier.py"


def test_submission_config_uses_repo_relative_root():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert payload["path"] == ".."
    assert (CONFIG_PATH.parent / payload["path"]).resolve() == REPO_ROOT.resolve()


def test_metafrontier_fallback_uses_repo_sibling_default():
    text = ENGINE_PATH.read_text(encoding="utf-8")

    assert r"C:\MetaFrontierLab" not in text
    assert 'DEFAULT_METAFRONTIER_PATH = REPO_ROOT.parent / "MetaFrontierLab"' in text

