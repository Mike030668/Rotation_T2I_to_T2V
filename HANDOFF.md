# Stage A Package C1 — Architect Handoff (`HANDOFF.md`)

Date: 2026-11-24
To: DS Colleague / Research Curator (via Owner)
From: Architect
Branch: `local/stage-a-git-first` (D066 public work branch)
Code Commit: `7ed83698014a01007baa480f173a482b69a5d9c7` (tree `ac5a271cd7287d40f72dc521761b0cedcc75eb37`)
Requested Gate: **`C1_GLOBAL_SAMPLER_JOURNAL_SOURCE_REVIEW`**

## 1. Summary of C1 Deliverable

Package C1 (`C1-01`…`C1-06`) is source-implemented (`SOURCE_IMPLEMENTED`, unreviewed, unexecuted) across `__init__.py`, `plan.py`, `exposure.py`, `spine.py`, and `configs/stage_a_synthetic.json`:
- Strict typed/size-bounded `exposure-v2` + `sampler-v2` schemas with `type(x) is int`, strict JSON duplicate-key and nonfinite refusal, and explicit legacy `exposure-v1`/untyped-sampler rejection prior to sampler calls or state mutation.
- Pre-call `(step_epoch, cursor_before, cursor_after)` capture via `advance_window` with explicit `rollover_before_overflow` policy (`400 -> (epoch+1, 0)`).
- Full journal global reconstruction `1..completed` from `(0, 0)` against pinned `SYNTHETIC_TRAIN_IDS_SHA256` (`76f17184...`) and unchanged `stage_a_sampler_v2.py` (`SAMPLER_SHA256 3dcdebeb...`).
- Pre-step canonical receipt bytes & `receipt_sha256`, reused verbatim at DEV checkpointing and reconciled across `header == sampler-v2 == exposure-v2` **before** model/AdamW/RNG restoration in `spine.resume`.
- Complete `CONTRACT_MATRIX.md`, `STATIC_ADMISSION_ARGUMENT.md`, `FIXTURE_SPEC.md` (`POS-01`..`POS-10`, `NEG-01`..`NEG-20`, all `PREPARED_NOT_RUN`), `SOURCE_REUSE_MATRIX.md`, and `CHANGE_MANIFEST.md`.

## 2. Grouped Blockers Remaining Before Stage A Gate Closure

- **C1 (pending DS review):** Static completeness and adversarial fixture review for `C1-01`…`C1-06` (`C1_GLOBAL_SAMPLER_JOURNAL_SOURCE_REVIEW`).
- **C2 (open):** Bounded/offset-checked `checkpoints.unpack`, pre-mutation staging & rollback for model/AdamW/RNG in `resume`, directory fsync & crash-safe BEST/LAST pointers in `checkpoints.publish`.
- **C3 (open):** Binding actual forwarded `model_kwargs` tensor content/shape/dtype to pre-step receipt (`forward_binding="DEFERRED_TO_C3"`), exception-safe `dev.evaluate` mode restoration, historical `500_1` corpus ABI.
- **C4 / Owner (`PENDING_OWNER`):** External Owner-pinned Ed25519 trust anchor, trusted clock/nonce/revocation policy, full resolved config (`clip`, `horizons_path`), and genuine integrated A–F wiring in `tests/stage_a/test_a_f.py`.

## 3. Non-Execution & Governance Attestation

```text
tests_executed=false
imports_or_compiles_executed=false
model_forward_calls=0
optimizer_step_calls=0
checkpoint_written=false
historical_corpus_read=false
previous_BEST_modified=false  # Architect attestation only
```

- Research branch `splitter-time-consistent` (`e4f2d8db...`) and `master` (`7b224dd2...`) untouched.
- Passing `C1_GLOBAL_SAMPLER_JOURNAL_SOURCE_REVIEW` accepts the C1 source contract only; it does **not** pass `STC_STAGE_A_SINGLE_GIT_INTEGRATION_STATIC_REVIEW`, does **not** close `STC-STAGE-A-DEV-CKPT-PRETRAIN-01`, and does **not** authorize execution.
