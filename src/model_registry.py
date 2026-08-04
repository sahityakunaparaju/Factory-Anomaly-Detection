"""Shared model-registry helpers.

``src/train.py`` writes ``models/current_model.json`` naming whichever model
artifact won (``model_file``) plus its native default threshold (``-offset_``).
This module is the single source of truth for resolving that file, so serving
(``src/consumer.py``) and the evaluation tooling can never pair a stale model
file with a new scaler bundle after a retrain picks a different model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST_PATH = Path("models/current_model.json")
DEFAULT_MODEL_FILE = "lof.joblib"

def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> dict[str, object] | None:
    """Load the current-model manifest.

    Returns None when the file is missing (callers then use the default model
    file). A PRESENT but unparseable manifest is a real problem - silently
    falling back to the default would recreate the exact stale-model/scaler
    mismatch this registry exists to prevent - so it is reported loudly on
    stderr before returning None.
    """
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[model_registry] WARNING: {path} exists but could not be parsed "
            f"({exc}); falling back to the default model file",
            file=sys.stderr,
        )
        return None

def resolve_model_path(
    manifest_path: str | Path = MANIFEST_PATH,
    default_model_file: str = DEFAULT_MODEL_FILE,
) -> Path:
    """Resolve the deployed model file from the manifest.

    The model file is resolved relative to the manifest's directory. Falls
    back to ``models/{default_model_file}`` when the manifest is missing or
    has no ``model_file``.
    """
    manifest = load_manifest(manifest_path)
    if manifest:
        model_file = manifest.get("model_file")
        if model_file:
            return Path(manifest_path).parent / str(model_file)
    return Path("models") / default_model_file
