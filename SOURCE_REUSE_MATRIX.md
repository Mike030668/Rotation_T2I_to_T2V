# Historical source reuse matrix

Date: 2026-11-23
Status: `SOURCE_ONLY / NOT_RUN`

| Source | SHA256 | Decision | Why |
|---|---|---|---|
| `stage_a_sampler_v2.py` | `3dcdebeb24fca37a6b90cba32a5c0f7cf5a619e78534d668abb598cb508ed9a0` | REUSE unchanged | video/epoch/seed sparse selection is already reviewed |
| `trajectory_data.py` | `397f14bfc00b50b6dc4495da9b90b240a2733605c23a43dcfc31ad665a4c58b1` | REFERENCE_ONLY | schedule formula is reused in `samples.schedule`; file not copied because it reads historical trajectory objects |
| `stage_a_runner.py` | `aea3fed12ab3ad8d58bd38f97983380431422cc2c543a7dbe5c7f26776bfbaed` | REFERENCE_ONLY | historical loader ABI is the contract; the one-update cap is not the new loop |
| DEV horizons manifest | `b7ad9bb1979285043c62c0e174b7b6515daa843ad97131444928b9a02e1de412` | REUSE as metadata | 50 videos and 147 keys; tensors remain synthetic |
| BEST/LAST publisher | current `checkpoints.py` | ADAPT | section hashes exist; trusted generation admission is still incomplete |
| A–F | `tests/stage_a/test_a_f.py` | NEW WIRING REQUIRED | cases exist; they do not yet drive the runner |

Call graph: `plan.next_batch` calls copied `stage_a_sampler_v2.sample`, then converts ordinals to labels before `synthetic_item`.
