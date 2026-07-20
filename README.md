# GRPO with verifiable rewards reproduction

Clean-room, source-pinned reproduction of *Reinforcement Learning with
Verifiable Rewards: GRPO's Loss, Dynamics, and Success Amplification* (ICML
2026; OpenReview `z9Bo4eXSpW`; arXiv `2503.06639v4`).

The current challenge contract (refreshed on 2026-07-20) contains three claims
(six possible points):

1. Adaptive weighted-contrastive form for binary-reward GRPO.
2. Closed-form optimal policy recursion.
3. Success-probability amplification beyond reference performance.

The accepted source's Theorem 5 parametric approximation bound is included as
supplemental source evidence, but it is not counted as a separate current jury
claim.

The accepted source promises supplementary code but contains no executable
artifact. This package therefore uses only independent finite-probability and
optimization oracles, with all population/no-clipping/exact-optimizer limits
and source notation defects made explicit.
