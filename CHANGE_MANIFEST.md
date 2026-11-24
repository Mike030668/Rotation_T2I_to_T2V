# Stage A Package C1 Change Manifest (`CHANGE_MANIFEST.md`)

Date: 2026-11-24
Branch: `local/stage-a-git-first`
Base Commit: `e61c30e98f62730862cbd28342a3f80d2f3dc612`
Code Commit (Commit 1): `7ed83698014a01007baa480f173a482b69a5d9c7`
Code Tree: `ac5a271cd7287d40f72dc521761b0cedcc75eb37`
Docs Commit (Commit 2): wraps this manifest; its exact 40-char SHA, tree, and doc blobs are recorded in the canonical Drive Architect handoff `STC_STAGE_A_C1_GLOBAL_SAMPLER_JOURNAL_HANDOFF_20261124.md`.

## 1. Changed Code & Config Files (Commit 1 `7ed83698`)

| Path | Action | Git Blob (`7ed83698`) | SHA256 | Contract IDs |
|---|---|---|---|---|
| `src/stage_a_integrated/__init__.py` | Modified | `d334ad6dfd717a6b9eecc6fa04d155c657adc1f6` | `443594c2c8814046c23a47cf1c35527611e05b215810615136ec09482a0b302e` | `C1-02` |
| `src/stage_a_integrated/plan.py` | Modified | `20e04cb488bc1007a50411cd1b1ed1d7b7d7b2ed` | `98921cfff6733a2a2cbad18d12f1835b438960e7e27fee9a3e2da008a7751003` | `C1-01..C1-04` |
| `src/stage_a_integrated/exposure.py` | Modified | `ba605f090755af3e0eba218118688a91b749c871` | `63170437129c593c7a347ef46e7ea77764f82304f1d1d5b35a0a5127ebd9bde8` | `C1-01..C1-06` |
| `src/stage_a_integrated/spine.py` | Modified | `b6cdbb8dcb9ab42b7c27629e863820288bbc131e` | `83b78a8a7acd0de022e03dbaf942a53315693416ff73a96b003c829a70dadbc8` | `C1-01..C1-06` |
| `configs/stage_a_synthetic.json` | Modified | `887dff791513002039770dcc3fc5187cff99f57c` | `1efdf17f527f4deec6f5df96d76836e1273564fa02f1faa541f3144e2ccb6a3f` | `C1-02..C1-06` |

## 2. Unchanged Frozen & Supporting Files (Commit 1 `7ed83698`)

| Path | Role | Git Blob (`7ed83698`) | SHA256 |
|---|---|---|---|
| `src/stage_a_integrated/stage_a_sampler_v2.py` | Frozen historical sampler | `04b7656727bd69dadac25278cec284034f591ad1` | `3dcdebeb24fca37a6b90cba32a5c0f7cf5a619e78534d668abb598cb508ed9a0` |
| `src/stage_a_integrated/pinned_splitter_v3.py` | Frozen candidate model | `6307111035db7c3c8fab0f615e19ea0d30fe2aa0` | `43937fdda6c81a8af73d642f73a9dfc03fb077dafe8b24faac6daed029a881ef` |
| `src/stage_a_integrated/pinned_losses_v3.py` | Frozen prepared loss | `84075b60dd2261cee7dbb7b14d29ff320ac65125` | `ce7ccc063c8e8e1a8e7654b90cd00c2dd8ab37bfe605f82fa6c01f95a4056f0e` |
| `src/stage_a_integrated/authorization.py` | C4 trust loader (untouched) | `b734fb7da1eea634e8f7041398906cb61d7e5347` | `78cbc505804da594a77e44feb37c4e467c661e4dc21ded448ef33940f497dbba` |
| `src/stage_a_integrated/checkpoints.py` | C2 checkpoint pack/publish (untouched) | `bfcd744a1c1ccc8ec92458643cc52cf00cf794cb` | `aa0ab185da8841cc637d9b767a1c9e00fcb839ec4352cf7df7be31298943f516` |
| `src/stage_a_integrated/cli.py` | Default-denied CLI (untouched) | `c737bf672141b51379edff8945a80b52b9f9a5e6` | `b6a00a067d706dc8545423cf84a63625b1e61118c718dbda8156d58ed9678f32` |
| `src/stage_a_integrated/dev.py` | C2/C3 DEV evaluator (untouched) | `2a3635e80305fef9a310b5cebaf4e4586ed1b81e` | `d6ac3138dffe070bfc23a3d9fbd7b55dc71f2716bd3f5dadc1c22399c555f8c1` |
| `src/stage_a_integrated/model_adapter.py` | Pinned model loader (untouched) | `a9a898d009b51ca50c58abb91d99c15d0e93d0be` | `1e0965f9e4b23e97554604003a156e797cc997b40770bbb13991d1b0637a931a` |
| `src/stage_a_integrated/objective.py` | Pinned loss loader (untouched) | `a37ce4b38c3ce74ac0dc2e8b1cbe2ae5a3b6e598` | `061e39e460e156f9353a73d778de245f5b382a94e5676a1c77809bdef1096ad7` |
| `src/stage_a_integrated/samples.py` | C3 synthetic item builder (untouched) | `5b141dc37aed5e203b7140e68d9ef334178e6470` | `b725450015d0c5db51b0cbaf5b0bb63608f92b4fa0746997ca8af3058bcb98c4` |
| `src/stage_a_integrated/trainer.py` | C3 optimizer step (untouched) | `3d4f4ed9f1aa33886e1f70c99b93169f0c3e7d45` | `872270eb6eb5fa68134cf5470109a9c1490a7c2951dafc2793b87151ee441788` |
| `tests/stage_a/test_a_f.py` | C4 A–F stubs (untouched) | `674400fcd59bdf4ffa5a4689215e88a6989a02e8` | `b9882a4a1bd8bfa2e5cd4bccbfd88b992a98577b5dca08fc7f7190f9f3bda064` |

## 3. Schema Evolution, Fields Added/Removed, and Legacy Policy

- **Exposure schema (`exposure-v1` -> `exposure-v2`):**
  - Added fields: `"source_id"`, `"source_sha256"`, `"sampler_sha256"`, `"remainder_policy"` (`"rollover_before_overflow"`), `"forward_binding"` (`"DEFERRED_TO_C3"`), `"batch_size"`, `"receipt_sha256"`.
  - Retained per-update fields: `"schema"`, `"completed"`, `"provider"`, `"seed"`, `"epoch"`, `"cursor_before"`, `"cursor_after"`, and the 9 columnar lists.
  - Removed legacy `exposure-v1` tolerance: both cursor-only and v1 before/after records are **explicitly rejected** (`ValueError("legacy schema exposure-v1 is not supported for resume; re-run under exposure-v2")`). No silent migration is performed.
- **Sampler section (`sampler` -> `sampler-v2`):**
  - Replaced untyped `{"completed", "epoch", "cursor"}` with canonical `sampler-v2` (`SAMPLER_V2_KEYS`: `schema`, `exposure_schema`, `completed`, `epoch`, `cursor`, `last_cursor_before`, `seed`, `batch_size`, `source_id`, `source_sha256`, `sampler_sha256`, `remainder_policy`, `exposure_journal_sha256`, `last_receipt_sha256`).
  - Untyped legacy sampler sections are **explicitly rejected** (`ValueError("legacy sampler section without schema is not supported; re-run under sampler-v2")`).
- **Checkpoint header provenance:**
  - Added `seed`, `batch_size`, `source_id`, `source_sha256`, `sampler_sha256`, `exposure_schema`, `exposure_journal_sha256`, `last_receipt_sha256`.
  - Legacy headers missing `exposure_schema` are explicitly rejected before any model/AdamW/RNG load.
- **Removed dead code:** `_require_exposure_fields` and `_admit_exposure` in `spine.py` were uncalled dead functions from pre-C1 commits and have been superseded by `validate_record_structure` and `reconcile_checkpoint_state`.
- **Signature updates:** `plan.next_batch` parameter renamed `cursor -> cursor_before`; `exposure.receipt` adds `batch_size`; `exposure.require_record` & `exposure.require_records` require external `expected_*` keyword pins and return canonical byte representations; `spine.resume` requires `expected_config: dict`.

## 4. Effective Config & Residual Blockers (C2–C4 / `PENDING_OWNER`)

- Effective `configs/stage_a_synthetic.json` (SHA256 `1efdf17f527f4deec6f5df96d76836e1273564fa02f1faa541f3144e2ccb6a3f`) remains `status: "NOT_RUN"`, `release_created: false`, and pins `sampler_sha256`, `exposure_schema`, `sampler_schema`, `synthetic_source_id`, `synthetic_train_size=400`, `synthetic_train_ids_sha256`, `synthetic_remainder_policy`, and `forward_binding`.
- **Remaining C2 blockers:** `checkpoints.unpack` does not enforce declared section offsets, bounded header length, or duplicate-key refusal in `json.loads`; `spine.resume` still loads model/AdamW/RNG sequentially after C1 admission without staging/rollback on late RNG corruption; `checkpoints.publish` lacks directory fsync and atomic pointer rollback receipts.
- **Remaining C3 blockers:** `forward_binding="DEFERRED_TO_C3"` — `item["model_kwargs"]` passed to `model.forward` in `trainer.optimizer_step` is not yet bound to `exposure-v2`; `dev.evaluate` lacks `try/finally` `model.train()` restoration; historical `500_1` corpus ABI not bound.
- **Remaining C4 / `PENDING_OWNER` blockers:** `authorization.py` still uses caller-supplied `expected` trust pin/clock/nonces (`OWNER_TRUST_AUTHORITY=PENDING_OWNER`); `configs/stage_a_synthetic.json` omits runtime `clip` and `horizons_path`; `tests/stage_a/test_a_f.py` only dispatches `case_f` and leaves `run_all` blocked.
