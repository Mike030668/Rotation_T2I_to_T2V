# Stage A Package C2-AMENDMENT-01 Change Manifest (`CHANGE_MANIFEST.md`)

**Branch:** `local/stage-a-git-first`  
**Prior History Chain:**
- C1 Baseline: Code `7ed83698014a01007baa480f173a482b69a5d9c7` -> Docs `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6` (`STC-D066-C1-7BB1C73F-SOURCE-ACCEPTED-WITH-BOUNDARIES-C2-NEXT-20261124-V1`)
- C2 Initial Submission (Partially Accepted per `STC_STAGE_A_C2_DS_REVIEW_20261124.md`): Initial code `19e034d2a5ef07443f9a9626c127b97ef48d23e0` -> Hardened code `6622078391736777edc62ea4a61d33c430bd3dc1` -> Candidate docs `b6cd585f4f91f77ef227d51fe4d06503ba23500d`
- **`C2-AMENDMENT-01` Code Commit (Commit 1):** `e04d81b53ddf69adb955f49508d8046c1aa15c02` (Tree `254ad3086faabaf09fa2d1a7d9e935b4de11be3b`, Parent `b6cd585f4f91f77ef227d51fe4d06503ba23500d`)

**Execution Status:** `SOURCE_ONLY / NO_IMPORT / NO_COMPILE / NO_TEST / STOP_FOR_DS`

---

## 1. Changed Code & Config Files (`e04d81b53ddf69adb955f49508d8046c1aa15c02`)

| Path | Action | Git Blob (`e04d81b53ddf69adb955f49508d8046c1aa15c02`) | SHA256 | Contract / Amendment IDs |
|---|---|---|---|---|
| `src/stage_a_integrated/checkpoints.py` | Modified (`C2-AMENDMENT-01`) | `77c92fef0b87f7099e27398413ff281925a09121` | `413cac3452e721a9b74dd51e1d914b9c1ea1ebe62a284ac9864322437d7f273a` | `C2-01` (`BOUNDED`), `C2-02..04`, `C2-A01..C2-A05` |
| `src/stage_a_integrated/spine.py` | Modified (`C2-AMENDMENT-01`) | `b7e24e0d58e8602a97a5af9cef1743e1bec9701d` | `8c67e02f24c0c8b87fdc93760d5f1109a2a2673ad1fdbae398933e0c5779a626` | `C1-05`, `C2-02..04`, `C2-A01..C2-A05` |
| `configs/stage_a_synthetic.json` | Modified (`C2-AMENDMENT-01`) | `6ff9605af0e686e2c49c0df8daf4f7be05e8c20a` | `c4ea5502c2dadb22e84b05b8952edb8ee24ac9d2f9a9a69476a83e175214cc24` | `C2-03`, `C2-A01`, `C2-A03`, `C2-A04`, `C2-A05` |

---

## 2. Unchanged Frozen, C1 Accepted & Downstream C3/C4 Files (`e04d81b53ddf69adb955f49508d8046c1aa15c02`)

| Path | Role | Git Blob | SHA256 |
|---|---|---|---|
| `src/stage_a_integrated/stage_a_sampler_v2.py` | Frozen historical sampler (byte-identical) | `04b7656727bd69dadac25278cec284034f591ad1` | `3dcdebeb24fca37a6b90cba32a5c0f7cf5a619e78534d668abb598cb508ed9a0` |
| `src/stage_a_integrated/pinned_splitter_v3.py` | Frozen candidate model (byte-identical) | `6307111035db7c3c8fab0f615e19ea0d30fe2aa0` | `43937fdda6c81a8af73d642f73a9dfc03fb077dafe8b24faac6daed029a881ef` |
| `src/stage_a_integrated/pinned_losses_v3.py` | Frozen prepared loss (byte-identical) | `84075b60dd2261cee7dbb7b14d29ff320ac65125` | `ce7ccc063c8e8e1a8e7654b90cd00c2dd8ab37bfe605f82fa6c01f95a4056f0e` |
| `src/stage_a_integrated/__init__.py` | C1 accepted package SHA pins (untouched) | `d334ad6dfd717a6b9eecc6fa04d155c657adc1f6` | `443594c2c8814046c23a47cf1c35527611e05b215810615136ec09482a0b302e` |
| `src/stage_a_integrated/plan.py` | C1 accepted synthetic window/sampler wrapper (untouched) | `20e04cb488bc1007a50411cd1b1ed1d7b7d7b2ed` | `98921cfff6733a2a2cbad18d12f1835b438960e7e27fee9a3e2da008a7751003` |
| `src/stage_a_integrated/exposure.py` | C1 accepted `exposure-v2`/`sampler-v2` engine (untouched) | `ba605f090755af3e0eba218118688a91b749c871` | `63170437129c593c7a347ef46e7ea77764f82304f1d1d5b35a0a5127ebd9bde8` |
| `src/stage_a_integrated/authorization.py` | C4 trust loader (untouched) | `b734fb7da1eea634e8f7041398906cb61d7e5347` | `78cbc505804da594a77e44feb37c4e467c661e4dc21ded448ef33940f497dbba` |
| `src/stage_a_integrated/cli.py` | Default-denied CLI (untouched) | `c737bf672141b51379edff8945a80b52b9f9a5e6` | `b6a00a067d706dc8545423cf84a63625b1e61118c718dbda8156d58ed9678f32` |
| `src/stage_a_integrated/dev.py` | C3 DEV evaluator (untouched; call site guarded in `spine.py`) | `2a3635e80305fef9a310b5cebaf4e4586ed1b81e` | `d6ac3138dffe070bfc23a3d9fbd7b55dc71f2716bd3f5dadc1c22399c555f8c1` |
| `src/stage_a_integrated/model_adapter.py` | Pinned model loader (untouched) | `a9a898d009b51ca50c58abb91d99c15d0e93d0be` | `1e0965f9e4b23e97554604003a156e797cc997b40770bbb13991d1b0637a931a` |
| `src/stage_a_integrated/objective.py` | Pinned loss loader (untouched) | `a37ce4b38c3ce74ac0dc2e8b1cbe2ae5a3b6e598` | `061e39e460e156f9353a73d778de245f5b382a94e5676a1c77809bdef1096ad7` |
| `src/stage_a_integrated/samples.py` | C3 synthetic item builder (untouched) | `5b141dc37aed5e203b7140e68d9ef334178e6470` | `b725450015d0c5db51b0cbaf5b0bb63608f92b4fa0746997ca8af3058bcb98c4` |
| `src/stage_a_integrated/trainer.py` | C3 optimizer step (untouched) | `3d4f4ed9f1aa33886e1f70c99b93169f0c3e7d45` | `872270eb6eb5fa68134cf5470109a9c1490a7c2951dafc2793b87151ee441788` |
| `tests/stage_a/test_a_f.py` | C4 A–F stubs (untouched) | `674400fcd59bdf4ffa5a4689215e88a6989a02e8` | `b9882a4a1bd8bfa2e5cd4bccbfd88b992a98577b5dca08fc7f7190f9f3bda064` |

---

## 3. Summary of `C2-AMENDMENT-01` Schema & Behavioral Additions

- **`commit-v1` (`CURRENT_COMMIT`):** Single authoritative cross-role durable commit uniting `last`, `best`, `best_history`, `commit_seq`, `effective_config_sha256`, `owner_trust_root_status="PENDING_OWNER_C4"`, and `release_created=False`. Replaces split `publish(...)` completely (`publish` removed). Derived `CURRENT_BEST`/`CURRENT_LAST` mirrors add `commit_seq` + `commit_sha256` and are ignored on read (`resolve_pointer` resolves exclusively via `resolve_commit`).
- **Idempotency & Locking:** `.writer.lock` via `fcntl.flock(LOCK_EX | LOCK_NB)`; `_materialize_generation` verifies exact byte/manifest/provenance match on reuse and refuses conflicts without overwrite; `recover_commit_state` re-syncs derived mirrors when `CURRENT_COMMIT` exists and never promotes uncommitted generations by guessing.
- **Semantic AdamW (`stage_adamw_state`):** Strict `{step, exp_avg, exp_avg_sq}` keys (no scalar fallback), CPU dtype/shape/finiteness, `exp_avg_sq >= 0`, integral `step == completed`, explicit `completed == 0 -> {}`, exact live parameter mapping & frozen hyperparameters (`lr=1e-4, wd=1e-4, betas=(0.9,0.999), eps=1e-8, amsgrad=False`), source-justified `expected_updated_param_ids`, and cloned live fixed `param_groups`.
- **End-to-End CPU RNG Boundary & Fatal Rollback:** `capture_cpu_rng_snapshot` before `model_cls()` on resume; full `admitted_result` metadata admission before apply; `FatalRollbackError` + `"fatal_rollback"` failure receipt (with `original_error` and `rollback_errors`, `must_terminate_process=True`) on any rollback failure; `"post_commit_failure"` (`commit_durable=True`) for `PostCommitError`; `rng_scope="cpu_only"` (no GPU claim).
- **External Provenance (`prov-v1`):** `horizons_file_sha256 == HORIZONS_SHA256` enforced at provenance build & baseline verify; `expected_provenance` mandatory in `_preflight_and_reconcile` and `commit_event`; `PENDING_OWNER_C4` preserved without key/release creation.

---

## 4. Residual Blockers & Verification Attestation

1. **C3 (`NOT_IMPLEMENTED`):** Forward-input digest binding (`forward_binding` remains `"DEFERRED_TO_C3"`), `500_1` historical corpus & split ABI crosswalk, `dev.py` internal hardening, and same-forward leak negative.
2. **C4 / `PENDING_OWNER`:** Independent Ed25519 Owner trust root (`owner_trust_root_status="PENDING_OWNER_C4"`), clock/nonce/revocation ledger, full CLI config (`clip`, `horizons_path`), and A–F harness wiring in `tests/stage_a/test_a_f.py`.
3. **Verification Attestation:** Static source inspection only; ZERO Python code executed, imported, compiled, or runtime-tested (`NO_EXECUTION`).
