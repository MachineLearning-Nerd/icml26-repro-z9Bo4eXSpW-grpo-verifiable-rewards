"""Independent regression tests for the current three-claim z9Bo4eXSpW audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "repro" / "src" / "full_audit.py"
SPEC = importlib.util.spec_from_file_location("full_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def test_claim_1_weighted_contrastive_identity(self) -> None:
        source = AUDIT.source_audit(ROOT / "docs" / "source")
        claim = AUDIT.audit_contrastive_identity()
        self.assertTrue(source["pass"])
        self.assertTrue(claim["pass"])
        self.assertEqual(claim["finite_distribution_cells"], 1200)
        self.assertLessEqual(claim["maximum_clipped_contrastive_identity_error"], AUDIT.TOL)
        self.assertGreaterEqual(claim["literal_conditional_notation_rejections"], 1140)

    def test_claim_2_closed_form_policy(self) -> None:
        claim = AUDIT.audit_closed_form_policy()
        self.assertTrue(claim["pass"])
        self.assertEqual(claim["independent_slsqp_cells"], 108)
        self.assertEqual(claim["optimizer_failures"], 0)
        self.assertGreaterEqual(claim["wrong_anchor_rejections"], 98)
        self.assertGreaterEqual(claim["wrong_old_policy_statistic_rejections"], 98)

    def test_claim_3_recurrence_and_amplification_scope(self) -> None:
        claim = AUDIT.audit_success_amplification()
        self.assertTrue(claim["pass"])
        self.assertEqual(claim["finite_policy_recurrence_cells"], 720)
        self.assertLessEqual(claim["maximum_policy_to_scalar_recurrence_error"], 2e-10)
        self.assertGreater(claim["minimum_resolved_fixed_root_margin_over_reference"], 0.0)
        self.assertGreater(claim["period_two_cells"], 0)
        self.assertEqual(claim["support_boundary_controls"], {"zero_reference": 0.0, "unit_reference": 1.0})

    def test_supplemental_theorem_5_parametric_cumulative_tv_bound(self) -> None:
        claim = AUDIT.audit_parametric_approximation_bound()
        self.assertTrue(claim["pass"])
        self.assertEqual(claim["finite_policy_cumulative_tv_cells"], 1600)
        self.assertEqual(claim["assumption_failures"], 0)
        self.assertEqual(claim["theorem_bound_failures"], 0)
        self.assertEqual(claim["omitted_cumulative_increment_controls_rejected"], 1600)


if __name__ == "__main__":
    unittest.main()
