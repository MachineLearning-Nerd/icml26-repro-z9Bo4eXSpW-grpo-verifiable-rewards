"""Register the portable local evidence archive in Trackio."""

from __future__ import annotations

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[2]
trackio.init(project="repro-grpo-verifiable-rewards", name="full-publication-gate", resume="allow")
trackio.log_artifact(ROOT / "outputs" / "reproduction_bundle.zip", name="repro-bundle", type="dataset")
trackio.finish()
