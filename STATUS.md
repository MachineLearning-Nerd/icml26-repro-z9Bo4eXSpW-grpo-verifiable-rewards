# Status

- Paper: `z9Bo4eXSpW` — *Reinforcement Learning with Verifiable Rewards: GRPO's Loss, Dynamics, and Success Amplification*
- Owner: `codex-grpo-five-claims`
- State: `publication_queued`
- Effective contract: 3 live claims / 6 possible points
- Primary source: arXiv `2503.06639v4`, source SHA-256 `6d2325bc504b43c41b3ab27fe9515554beb8ed598eb5848185b42cf10ad0f94a`
- Author code: none in the accepted arXiv source; clean-room implementation only

## Completed local gate

The official claims endpoint was refreshed on 2026-07-20: it currently lists
three, not five, scoreable claims. The source-pinned audit covers all three:
1,200 contrastive cells, 108 independent policy-optimizer cells, and 720
finite recurrence cells plus 455 dense fixed-point cells. It retains clipping,
support, positive-smoothing, and two-cycle controls. A 1,600-cell cumulative-TV
calculation for source Theorem 5 is retained as supplemental, non-scoring
evidence. All four regression tests pass, and the fail-closed verifier reports
all three live claims complete for 6/6 local points.

The public evidence repository is
`MachineLearning-Nerd/icml26-repro-z9Bo4eXSpW-grpo-verifiable-rewards` at
commit `78c1e4c552ba53454c2a060edf9d59ec61381af5`. After that push, the
gate-complete paper was atomically added as canonical backlog entry 70.

## Next action

Await the single shared Hugging Face backlog drain. After it creates the Space,
verify the public tags, commit SHA, and artifact bucket, then record the
readback here and in the shared coordination row.
