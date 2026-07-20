# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_54fa384d4d20", "created_at": "2026-07-20T09:59:49+00:00", "title": "Source and method"}
-->
Clean-room finite-probability and independent numerical-optimization audit. Pinned accepted arXiv 2503.06639v4 source archive and primary TeX; no executable author code exists in the accepted source. The current website contract was fetched on 2026-07-20 and contains exactly three scoreable claims.


---
<!-- trackio-cell
{"type": "code", "id": "cell_7d09fd289f05", "created_at": "2026-07-20T10:00:13+00:00", "title": "Regression suite", "command": ["python", "repro/src/run_tests.py"], "exit_code": 0, "duration_s": 6.579}
-->
````bash
$ python repro/src/run_tests.py
````

exit 0 · 6.6s


````python title=run_tests.py
"""Run the full standard-library test suite and retain a machine-readable result."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "repro/tests", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "tests_passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (ROOT / "outputs" / "test_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(completed.stdout, end="")
    print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

````


````output
test_claim_1_weighted_contrastive_identity (test_audit.AuditTests.test_claim_1_weighted_contrastive_identity) ... ok
test_claim_2_closed_form_policy (test_audit.AuditTests.test_claim_2_closed_form_policy) ... ok
test_claim_3_recurrence_and_amplification_scope (test_audit.AuditTests.test_claim_3_recurrence_and_amplification_scope) ... ok
test_supplemental_theorem_5_parametric_cumulative_tv_bound (test_audit.AuditTests.test_supplemental_theorem_5_parametric_cumulative_tv_bound) ... ok

----------------------------------------------------------------------
Ran 4 tests in 5.974s

OK

````
