# Status

- Paper: `z9Bo4eXSpW` — *Reinforcement Learning with Verifiable Rewards: GRPO's Loss, Dynamics, and Success Amplification*
- Owner: `codex-grpo-five-claims`
- State: `in_progress`
- Effective contract: 3 live claims / 6 possible points
- Primary source: arXiv `2503.06639v4`, source SHA-256 `6d2325bc504b43c41b3ab27fe9515554beb8ed598eb5848185b42cf10ad0f94a`
- Author code: none in the accepted arXiv source; clean-room implementation only

## Current step

The official claims endpoint was refreshed on 2026-07-20: it currently lists
three, not five, scoreable claims. The source-pinned audit covers all three:
1,200 contrastive cells, 108 independent policy-optimizer cells, and 720
finite recurrence cells plus 455 dense fixed-point cells. It retains clipping,
support, positive-smoothing, and two-cycle controls. A 1,600-cell cumulative-TV
calculation for source Theorem 5 is retained as supplemental, non-scoring
evidence.

## Next action

Promote the three-claim audit into a standalone fail-closed verifier and tests,
then create the evidence bundle and logbook.
