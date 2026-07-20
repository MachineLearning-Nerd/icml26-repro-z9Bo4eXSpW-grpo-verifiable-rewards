# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6c76651f4c67", "created_at": "2026-07-20T09:59:52+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T09:59:53+00:00"}
-->
All three current live claims are complete locally: 6/6 points, source pins verified, and four regression tests pass. C1 is an exact finite-space contrastive identity with source notation corrections disclosed; C2 is independently optimizer-certified for the exact population/no-clip objective; C3 is verified for interior-reference fixed-point amplification while support and non-convergence limits are explicitly retained.

## Scope & cost

| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | All three current website claims; clean-room finite policy and optimizer evidence | End-to-end parametric GRPO training, which the current claims do not require |
| Hardware | Local CPU; no GPU required | GPU would be required for LLM-scale training |
| Time | About 7 seconds per full audit | Not applicable to the source-theorem claims |
| Cost | $0 local compute | Not incurred |
| Outcome | Full local publication gate passed: 6/6 | Not represented as completed |


---
<!-- trackio-cell
{"type": "code", "id": "cell_8d69b9bb12c4", "created_at": "2026-07-20T10:00:21+00:00", "title": "Fail-closed claim verification", "command": ["python", "repro/src/verify_claims.py"], "exit_code": 0, "duration_s": 0.042}
-->
````bash
$ python repro/src/verify_claims.py
````

exit 0 · 0.0s


````python title=verify_claims.py
"""Fail-closed verifier for the current three-claim z9Bo4eXSpW contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pins = load(ROOT / "docs" / "source_pins.json")
    live = load(ROOT / "docs" / "live_claims_2026-07-20.json")
    audit = load(OUT / "full_audit.json")
    source_pinned = (
        sha256(ROOT / "docs" / "arxiv_source.tar") == pins["arxiv_source_sha256"]
        and sha256(ROOT / "docs" / "primary.pdf") == pins["primary_pdf_sha256"]
        and sha256(ROOT / "docs" / "source" / "main.tex") == pins["primary_tex_sha256"]
    )
    live_contract_pinned = (
        live["openreview_id"] == "z9Bo4eXSpW"
        and live["claim_count"] == 3
        and live["points_possible"] == 6
        and len(live["claims"]) == 3
    )
    c1 = audit["claims"]["claim_1_contrastive_loss"]
    c2 = audit["claims"]["claim_2_closed_form_policy"]
    c3 = audit["claims"]["claim_3_success_amplification"]
    supplemental = audit["supplemental_source_checks"]["theorem_5_parametric_approximation_bound"]
    test_path = OUT / "test_results.json"
    tests_passed = test_path.is_file() and load(test_path).get("tests_passed") is True

    claim_1 = bool(
        source_pinned
        and live_contract_pinned
        and c1["pass"]
        and c1["finite_distribution_cells"] == 1200
        and c1["maximum_clipped_contrastive_identity_error"] <= 5e-10
        and c1["literal_conditional_notation_rejections"] == 1200
        and c1["swapped_clipping_identity_rejections"] == 1200
        and c1["omitted_smoothing_rejections"] == 1200
    )
    claim_2 = bool(
        source_pinned
        and live_contract_pinned
        and c2["pass"]
        and c2["independent_slsqp_cells"] == 108
        and c2["optimizer_failures"] == 0
        and c2["maximum_optimizer_l1_error"] <= 3e-5
        and c2["maximum_kl_optimality_certificate_error"] <= 2e-9
        and c2["wrong_anchor_rejections"] >= 98
        and c2["wrong_old_policy_statistic_rejections"] >= 98
    )
    claim_3 = bool(
        source_pinned
        and live_contract_pinned
        and c3["pass"]
        and c3["finite_policy_recurrence_cells"] == 720
        and c3["dense_parameter_cells"] == 455
        and c3["maximum_policy_to_scalar_recurrence_error"] <= 2e-10
        and c3["amplification_failures"] == 0
        and c3["minimum_resolved_fixed_root_margin_over_reference"] > 0.0
        and c3["period_two_cells"] > 0
        and c3["support_boundary_controls"] == {"zero_reference": 0.0, "unit_reference": 1.0}
    )
    supplemental_theorem_5 = bool(
        supplemental["pass"]
        and supplemental["finite_policy_cumulative_tv_cells"] == 1600
        and supplemental["assumption_failures"] == 0
        and supplemental["theorem_bound_failures"] == 0
        and supplemental["omitted_cumulative_increment_controls_rejected"] == 1600
    )
    claims = (claim_1, claim_2, claim_3)
    payload = {
        "paper": "z9Bo4eXSpW",
        "source_pinned": source_pinned,
        "live_contract_pinned": live_contract_pinned,
        "live_claim_count": live["claim_count"],
        "claim_1_weighted_contrastive_loss": claim_1,
        "claim_2_closed_form_policy": claim_2,
        "claim_3_success_amplification": claim_3,
        "supplemental_theorem_5_parametric_bound": supplemental_theorem_5,
        "decisive_claims": sum(claims),
        "earned_points": 2 * sum(claims),
        "all_claims_complete": all(claims),
        "tests_passed": tests_passed,
        "publication_gate_passed": all(claims) and tests_passed,
        "scope": (
            "Population, no-clipping, exact policy optimization. Claim 3 verifies "
            "amplification/fixed points for interior reference support, while retaining "
            "zero-support and global-convergence counter-controls."
        ),
    }
    (OUT / "claim_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["all_claims_complete"]:
        raise SystemExit("incomplete z9Bo4eXSpW claim verification")


if __name__ == "__main__":
    main()

````


````output
{
  "paper": "z9Bo4eXSpW",
  "source_pinned": true,
  "live_contract_pinned": true,
  "live_claim_count": 3,
  "claim_1_weighted_contrastive_loss": true,
  "claim_2_closed_form_policy": true,
  "claim_3_success_amplification": true,
  "supplemental_theorem_5_parametric_bound": true,
  "decisive_claims": 3,
  "earned_points": 6,
  "all_claims_complete": true,
  "tests_passed": true,
  "publication_gate_passed": true,
  "scope": "Population, no-clipping, exact policy optimization. Claim 3 verifies amplification/fixed points for interior reference support, while retaining zero-support and global-convergence counter-controls."
}

````


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_dd0aaadf86c5", "created_at": "2026-07-20T10:00:23+00:00", "title": "Dashboard: repro-grpo-verifiable-rewards", "dashboard_project": "repro-grpo-verifiable-rewards"}
-->
**🎯 Trackio dashboard** `repro-grpo-verifiable-rewards`

trackio-local-dashboard://repro-grpo-verifiable-rewards
