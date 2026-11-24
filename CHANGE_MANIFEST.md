# Stage A Package C2 Change Manifest (`CHANGE_MANIFEST.md`)

**Branch:** `local/stage-a-git-first`  
**C1 Baseline Commits:** Code `7ed83698014a01007baa480f173a482b69a5d9c7` -> Docs `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6`  
**C2 Code Commit Chain:** Initial `19e034d2a5ef07443f9a9626c127b97ef48d23e0` (tree `6b5c4af08c8987107d5bd900b65dd15e3561e079`, parent `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6`) -> Final `6622078391736777edc62ea4a61d33c430bd3dc1` (tree `07b850f720fb952fecc92dbfcdb7536aa3821ff8`)  
**Execution Status:** `SOURCE_ONLY / NO_IMPORT / NO_COMPILE / NO_TEST / STOP_FOR_DS`

---

## 1. Changed Code & Config Files (Final C2 Code State `6622078391736777edc62ea4a61d33c430bd3dc1`)

| Path | Action | Git Blob (`6622078391736777edc62ea4a61d33c430bd3dc1`) | SHA256 | Contract IDs |
|---|---|---|---|---|
| `src/stage_a_integrated/checkpoints.py` | Modified (C2) | `31b35dc64a5e1317b819cb1f9b7b5a3fb6101e28` | `a8df5595ed4f2d66c35c674c02530238ad3b9947fc8c89d5846c2674fa1a7f31` | `C2-01..C2-04` |
| `src/stage_a_integrated/spine.py` | Modified (C2) | `0b5be8c111820c5ff4136b83dff39aecc839d7cc` | `b99b3330224d6c8ab0bf6bbab3fa996d27cbbb324a8f6321777711cfd5d31505` | `C1-05`, `C2-01..C2-04` |
| `configs/stage_a_synthetic.json` | Modified (C2) | `e1eb44ceba3a2f28f88c0f204596d07b5494ba59` | `f9eb20b5fe909d499caf027941be37684e3543eaa470197aee4b6e85c1122f9c` | `C2-01..C2-04` |

---

## 2. Unchanged Frozen, C1 Accepted & Downstream C3/C4 Files (`6622078391736777edc62ea4a61d33c430bd3dc1`)

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
| `src/stage_a_integrated/dev.py` | C3 DEV evaluator (untouched; call site hardened in `spine.py`) | `2a3635e80305fef9a310b5cebaf4e4586ed1b81e` | `d6ac3138dffe070bfc23a3d9fbd7b55dc71f2716bd3f5dadc1c22399c555f8c1` |
| `src/stage_a_integrated/model_adapter.py` | Pinned model loader (untouched) | `a9a898d009b51ca50c58abb91d99c15d0e93d0be` | `1e0965f9e4b23e97554604003a156e797cc997b40770bbb13991d1b0637a931a` |
| `src/stage_a_integrated/objective.py` | Pinned loss loader (untouched) | `a37ce4b38c3ce74ac0dc2e8b1cbe2ae5a3b6e598` | `061e39e460e156f9353a73d778de245f5b382a94e5676a1c77809bdef1096ad7` |
| `src/stage_a_integrated/samples.py` | C3 synthetic item builder (untouched) | `5b141dc37aed5e203b7140e68d9ef334178e6470` | `b725450015d0c5db51b0cbaf5b0bb63608f92b4fa0746997ca8af3058bcb98c4` |
| `src/stage_a_integrated/trainer.py` | C3 optimizer step (untouched) | `3d4f4ed9f1aa33886e1f70c99b93169f0c3e7d45` | `872270eb6eb5fa68134cf5470109a9c1490a7c2951dafc2793b87151ee441788` |
| `tests/stage_a/test_a_f.py` | C4 A–F stubs (untouched) | `674400fcd59bdf4ffa5a4689215e88a6989a02e8` | `b9882a4a1bd8bfa2e5cd4bccbfd88b992a98577b5dca08fc7f7190f9f3bda064` |

---

## 3. C2 Schema Additions, Bounds, and Failure-Receipt Catalog

- **`stca-v2` Checkpoint Envelope (`checkpoints.py:25-61,261-370`):**
  - Header adds `"checkpoint_schema": "stca-v2"`, `"body_bytes"`, `"body_sha256"`, and `"provenance"` (`prov-v1`); section entries enforce contiguous offsets (`offset == cursor`), section order `("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure")`, min/max bounds (`MAX_HEADER_BYTES = 1 MiB`, `MAX_MODEL_SECTION_BYTES = 128 MiB`, `MAX_ADAMW_SECTION_BYTES = 128 MiB`, `MAX_RNG_SECTION_BYTES = 64 KiB`, `MAX_CHECKPOINT_BYTES = 256 MiB`, `MAX_SAMPLER_BYTES = 4 KiB`, `MAX_JOURNAL_BYTES = 16 MiB`), strict duplicate-key & nonfinite refusal, and exact body/trailing-byte checks.
  - Legacy v1 checkpoints missing `"checkpoint_schema": "stca-v2"` are explicitly rejected.
- **`prov-v1` Provenance (`checkpoints.py:136-258`, `spine.py:107-140`, `configs/stage_a_synthetic.json:15-18`):**
  - Binds `candidate_sha256`, `loss_sha256`, `horizons_sha256`, `horizons_file_sha256`, `sampler_sha256`, `source_id`, `source_sha256`, `exposure_schema`, `sampler_schema`, `checkpoint_schema`, `manifest_schema`, `pointer_schema`, `failure_receipt_schema`, `remainder_policy`, `forward_binding`, `effective_config_sha256`, `generation_id`, `"owner_trust_root_status": "PENDING_OWNER_C4"`, `"release_created": false`.
- **`ckpt-manifest-v1` & `ptr-v1` Durable Pointers (`checkpoints.py:570-694,766-894`):**
  - Immutable `generations/u<N>-{BEST,LAST}/` holding `checkpoint.bin` + `manifest.json`; crash-recovery-safe `stage/u<N>-<ROLE>.tmp-NNNN`; same-FS check; file/stage/generations/root `fsync`; pre-rename and post-rename readback + `unpack`; atomic `CURRENT_BEST` and `CURRENT_LAST` recording `previous_generation_id` and `previous_checkpoint_sha256`.
- **`fail-receipt-v1` Failure Receipts (`checkpoints.py:704-763`, `spine.py:244-252,330-349,401-409,527-535,561-569`):**
  - Emits bounded canonical receipts in `failures/` for stages `"resume_preflight"`, `"resume_stage"`, `"resume_apply"`, `"dev_eval_failure"`, `"dev_nonfinite"`, and `"checkpoint_publish"` without ever altering `CURRENT_BEST`, `CURRENT_LAST`, or `generations/`.

---

## 4. Residual Blockers (C3–C4 / `PENDING_OWNER`)

1. **C3 (`NOT_IMPLEMENTED`):** Forward-input digest binding (`forward_binding` remains `"DEFERRED_TO_C3"`), `500_1` historical corpus & split ABI crosswalk, `dev.py` internal exception/finiteness hardening, and same-forward leak negative.
2. **C4 / `PENDING_OWNER`:** Independent Ed25519 Owner trust root (`owner_trust_root_status="PENDING_OWNER_C4"`), clock/nonce/revocation ledger, full CLI config (`clip`, `horizons_path`), and A–F harness wiring in `tests/stage_a/test_a_f.py`.
3. **Verification Attestation:** Static source inspection only; ZERO Python code executed, imported, compiled, or runtime-tested (`NO_EXECUTION`).
