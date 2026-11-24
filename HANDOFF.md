# Stage A Package `C2-AMENDMENT-01` — Architect Handoff (`HANDOFF.md`)

- **Requested DS Review:** `C2_AMENDMENT_01_SOURCE_REVIEW`
- **Branch:** `local/stage-a-git-first` (work-branch push only; frozen research branches untouched)
- **C1 Baseline:** `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6` (`STC-D066-C1-7BB1C73F-SOURCE-ACCEPTED-WITH-BOUNDARIES-C2-NEXT-20261124-V1`)
- **Prior Reviewed C2 Candidate:** `b6cd585f4f91f77ef227d51fe4d06503ba23500d` (`STC_STAGE_A_C2_DS_REVIEW_20261124.md`)
- **`C2-AMENDMENT-01` Code Commit (Commit 1):** `e04d81b53ddf69adb955f49508d8046c1aa15c02` (Tree `254ad3086faabaf09fa2d1a7d9e935b4de11be3b`, Parent `b6cd585f4f91f77ef227d51fe4d06503ba23500d`)
- **Execution Boundary:** `SOURCE_ONLY / WORK_BRANCH_PUSH_ONLY / NO_IMPORT / NO_COMPILE / NO_TEST / NO_MODEL_FORWARD / NO_SCORING / NO_OPTIMIZER / NO_CHECKPOINT_WRITE / NO_HISTORICAL_CORPUS / NO_DECODER / STOP_FOR_DS`

---

## 1. Summary of `C2-AMENDMENT-01` Deliverable (`C2-A01..C2-A06`)

1. **`CONTRACT_MATRIX.md`:** Preserves `C1-01..C1-06` at `SOURCE_REVIEW_ACCEPTED`, records `C2-01` as `SOURCE_REVIEW_ACCEPTED_BOUNDED` per `STC_STAGE_A_C2_DS_REVIEW_20261124.md`, keeps `C2-02..C2-04` at `SOURCE_IMPLEMENTED`, and adds `C2-A01..C2-A06` (`SOURCE_IMPLEMENTED`) with exact source locations and expected oracles. Zero runtime passes claimed.
2. **`checkpoints.py` (`77c92fef0b87f7099e27398413ff281925a09121`, SHA256 `413cac3452e721a9b74dd51e1d914b9c1ea1ebe62a284ac9864322437d7f273a`):**
   - `C2-A01`: Single atomic `CURRENT_COMMIT` (`commit-v1`) uniting `LAST`, `BEST`, and `best_history`; derived `CURRENT_BEST`/`CURRENT_LAST` mirrors; coherent `resolve_commit`/`resolve_pointer` strictly via `CURRENT_COMMIT`; preserved prior `BEST` bytes and earliest-tie semantics; `PostCommitError` distinction vs pre-commit failure.
   - `C2-A02`: Single-writer `.writer.lock` (`fcntl.flock(LOCK_EX | LOCK_NB)`); idempotent `_materialize_generation` verifying exact byte/manifest/`prov-v1` match on reuse and refusing conflicts; `recover_commit_state` never promoting uncommitted generations by guessing.
   - `C2-A03`: Semantic `AdamW` admission requiring exact `{step, exp_avg, exp_avg_sq}` keys (no scalar fallback), CPU tensor dtype/shape/finiteness, strict `exp_avg_sq >= 0`, integral `step == completed`, explicit `completed == 0 -> {}`, exact parameter order/mapping & frozen hyperparameters (`lr=1e-4, wd=1e-4, betas=(0.9,0.999), eps=1e-8, amsgrad=False` + all remaining options), fixed live `param_groups` preservation, and source-justified `expected_updated_param_ids`.
   - `C2-A04` & `C2-A05`: `capture_cpu_rng_snapshot`/`restore_cpu_rng_snapshot` + `FatalRollbackError`; external `build_expected_provenance` enforcing `horizons_file_sha256 == HORIZONS_SHA256`, `commit_schema="commit-v1"`, `rng_scope="cpu_only"`, and mandatory external `expected_provenance` with `PENDING_OWNER_C4` & `release_created=False`.
3. **`spine.py` (`b7e24e0d58e8602a97a5af9cef1743e1bec9701d`, SHA256 `8c67e02f24c0c8b87fdc93760d5f1109a2a2673ad1fdbae398933e0c5779a626`) & `configs/stage_a_synthetic.json` (`6ff9605af0e686e2c49c0df8daf4f7be05e8c20a`, SHA256 `c4ea5502c2dadb22e84b05b8952edb8ee24ac9d2f9a9a69476a83e175214cc24`):**
   - Captures caller CPU RNG BEFORE `model_cls(config_cls())` on resume; admits all returned metadata (`_admit` + `admitted_result`) before apply; distinguishes `FatalRollbackError` (`"fatal_rollback"` receipt with both error chains) and `PostCommitError` (`"post_commit_failure"`, `commit_durable=True`); fail-stop semantics; no GPU claim.
4. **`FIXTURE_SPEC.md`, `SOURCE_REUSE_MATRIX.md`, `CHANGE_MANIFEST.md`:** Preserves all C1 fixtures and specifies `C2-POS-01..08` / `C2-NEG-01..20` (`PREPARED_NOT_RUN`), full 64-char SHA256s in every row, and complete commit/schema manifests.

---

## 2. Grouped Blockers Remaining Before Stage A Gate Closure

- **C3 (`NOT_IMPLEMENTED`):** Forward-input digest binding (`forward_binding="DEFERRED_TO_C3"`), `500_1` historical corpus & split ABI crosswalk, `dev.py` internal hardening, same-forward leak negative.
- **C4 / `PENDING_OWNER`:** Owner Ed25519 trust anchor (replacing caller `expected` & `"PENDING_OWNER_C4"`), clock/nonce/revocation ledger, full resolved config (`clip`, `horizons_path`), real A–F harness wiring (`tests/stage_a/test_a_f.py`).
- **Scientific Gate:** `STC_STAGE_A_SINGLE_GIT_INTEGRATION_STATIC_REVIEW` remains `NOT PASSED`.

---

## 3. Non-Execution & Governance Attestation

- Frozen files (`stage_a_sampler_v2.py`, `pinned_splitter_v3.py`, `pinned_losses_v3.py`) and C1 accepted files (`__init__.py`, `plan.py`, `exposure.py`) remain byte-identical.
- Research branch untouched (`e4f2d8db845eb4e557676d27df0da3c45a489e69`); Held-out `SEALED`; Historical pilot `NOT_AUTHORIZED`; Paper Track `CLOSED`. Zero code executed.
