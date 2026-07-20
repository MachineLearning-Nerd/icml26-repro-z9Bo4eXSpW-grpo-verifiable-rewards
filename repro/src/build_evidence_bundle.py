"""Hash-address the complete local evidence set after the full gate passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
REQUIRED = (
    ROOT / "README.md",
    ROOT / "STATUS.md",
    ROOT / "docs" / "source_pins.json",
    ROOT / "docs" / "live_claims_2026-07-20.json",
    ROOT / "docs" / "arxiv_source.tar",
    ROOT / "docs" / "primary.pdf",
    ROOT / "docs" / "source" / "main.tex",
    ROOT / "repro" / "src" / "full_audit.py",
    ROOT / "repro" / "src" / "verify_claims.py",
    ROOT / "repro" / "src" / "run_tests.py",
    ROOT / "repro" / "tests" / "test_audit.py",
    OUT / "full_audit.json",
    OUT / "test_results.json",
    OUT / "claim_verification.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    claims = json.loads((OUT / "claim_verification.json").read_text(encoding="utf-8"))
    if not (
        claims["all_claims_complete"]
        and claims["tests_passed"]
        and claims["earned_points"] == 6
        and claims["publication_gate_passed"]
    ):
        raise SystemExit("cannot bundle an incomplete z9Bo4eXSpW reproduction")
    artifacts = {str(path.relative_to(ROOT)): sha256(path) for path in REQUIRED}
    bundle = {
        "paper": "z9Bo4eXSpW",
        "gate": "FULL_GATE_READY",
        "live_claim_count": 3,
        "earned_points": 6,
        "claim_outcomes": {
            "claim_1": "verified_with_source_notation_corrections_disclosed",
            "claim_2": "verified_for_exact_population_no_clip_objective",
            "claim_3": "verified_with_support_and_convergence_scope_corrections",
        },
        "supplemental_source_check": "Theorem 5 cumulative-TV bound verified in its stated conditional scope; not a current jury claim.",
        "artifacts": artifacts,
    }
    encoded = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    (OUT / "evidence_bundle.json").write_text(encoded, encoding="utf-8")
    marker = {
        "gate": "FULL_GATE_READY",
        "queue_marker": "FULL_GATE_READY: z9Bo4eXSpW",
        "paper": "z9Bo4eXSpW",
        "evidence_bundle_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "claim_verification_sha256": artifacts["outputs/claim_verification.json"],
        "claims_complete": True,
        "earned_points": 6,
        "tests_passed": True,
        "publication_gate_passed": True,
    }
    (OUT / "PUBLICATION_GATE_PASSED.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
