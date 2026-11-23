# Historical source reuse matrix

Date: 2026-11-23
Status: `SOURCE_ONLY / NOT_RUN`

| Component | Historical source | SHA256 | Decision | Incompatibility | Current call |
|---|---|---|---|---|---|
| B1 sampler | `stage_a_sampler_v2.py` | `3dcdebeb24fca37a6b90cba32a5c0f7cf5a619e78534d668abb598cb508ed9a0` | REUSE unchanged | returns ordinals, not labels | `plan.next_batch` |
| B1 schedule | `trajectory_data.py` `build_transition_descriptor` | `397f14bfc00b50b6dc4495da9b90b240a2733605c23a43dcfc31ad665a4c58b1` | ADAPT formula only | `make_pair_sample` builds dense history from anchor to current; frozen sampler uses a sparse prefix | `samples.schedule` |
| B1 loader | `stage_a_runner.py` lines 45–60 | `aea3fed12ab3ad8d58bd38f97983380431422cc2c543a7dbe5c7f26776bfbaed` | REFERENCE_ONLY | opens historical tensors; one-update cap | not imported |
| B2 DEV metadata | horizons manifest | `b7ad9bb1979285043c62c0e174b7b6515daa843ad97131444928b9a02e1de412` | REUSE | tensors are not in the manifest | `dev.load_rows` |
| B3 BEST/LAST | current `checkpoints.py` | worktree file | ADAPT | body hashes exist; trusted header admission is incomplete | `spine` publish/resume |
| B5 causal split | runner collate lines 64–88 | same runner SHA | ADAPT | historical tensors are not available in this source-only gate | `samples.synthetic_item` |
| B4 release | `authorization.py` | worktree file | NEW | no historical Owner trust module exists in the reviewed runner | `authorization.load` |
| B6 A–F | `tests/stage_a/test_a_f.py` | worktree file | NEW WIRING | cases do not yet drive the runner | not executed |

`make_pair_sample` is not the sparse-history protocol. The copied sampler remains the sparse-history protocol.
