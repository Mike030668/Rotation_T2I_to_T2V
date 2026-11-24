# Static Admission & Pre-Mutation Argument (`STATIC_ADMISSION_ARGUMENT.md`)

Date: 2026-11-24
Code Commit: `7ed83698014a01007baa480f173a482b69a5d9c7`
Package: Stage A — Contract C1 (`C1-01`…`C1-06`)
Boundary: `SOURCE_ONLY / NOT_RUN` (zero imports, compiles, tests, forward passes, optimizer steps, or checkpoint writes performed).

## 1. Call Graphs

### 1.1 Writer Call Graph (`run_synthetic`)
1. `spine.run_synthetic` (`spine.py:72-211`) validates release scope/budget (`authorization.load` at `:73`), validates run config (`_validate_run_config` at `:40-69`), and verifies static pins (`verify_sampler_pin` at `plan.py:29-34`, `verify_train_ids_pin(train_ids())` at `plan.py:37-52`).
2. If `config.get("resume_bytes")` is present (`:97`), invokes `resume(...)` (`:257-312`) before entering the loop.
3. For each update while `completed < max_updates` (`:121`):
   - **Pre-call cursor capture (`C1-03`):** `advance_window(epoch, cursor, batch_size)` (`plan.py:55-77`, called at `spine.py:122`) computes `(step_epoch, cursor_before, expected_cursor_after)` **without mutating** `epoch` or `cursor`. If `cursor + batch_size > 400`, `step_epoch = epoch + 1` and `cursor_before = 0`.
   - **Deterministic batch selection (`C1-02`, `C1-04`):** `next_batch(seed, step_epoch, cursor_before, batch_size)` (`plan.py:91-117`, called at `spine.py:123`) calls `select_video_ids(...)` (verifying `SYNTHETIC_TRAIN_IDS_SHA256`), calls `verify_sampler_pin()` before lazy-importing `stage_a_sampler_v2.sample`, maps ordinals to labels, builds `synthetic_item`, and records provenance fields (`epoch`, `seed`, `source_id`, `source_sha256`, `sampler_sha256`).
   - **Canonical pre-step receipt (`C1-06`):** `build_receipt_bytes(...)` (`exposure.py:314-324`, called at `spine.py:127`) calls `receipt(...)` (`exposure.py:257-311`), which validates every batch item (`require_item` at `:227-254`), hashes canonical body JSON (`receipt_body_digest` at `:182-183`), detaches via `strict_json_loads`, attaches `receipt_sha256`, and serializes compact sorted ASCII bytes (`canonical_json_bytes` at `:166-179`).
   - **Pre-optimizer admission (`C1-01`..`C1-04`, `C1-06`):** `require_record(proposed_bytes, completed_next, ...)` (`exposure.py:475-526`, called at `spine.py:136`) parses via `strict_json_loads`, runs `validate_record_structure(...)` (`exposure.py:371-472`) including global video-order verification and one call to `replay_ordinals`, and asserts exact canonical byte identity. `spine.py:148-149` checks `record["videos"] == chosen`.
   - **Optimizer step & post-step state commit:** `optimizer_step(model, batch, opt, clip)` (`spine.py:150`). Only after `step_inc == 1` succeeds do `completed`, `epoch`, `cursor`, `last_cursor_before`, `last_receipt_sha256`, and `exposure_frames` update (`spine.py:153-158`).
   - **DEV checkpoint publication:** At `completed in dev_events` (`:159`), `exposure_journal_bytes = b"[" + b",".join(exposure_frames) + b"]"` reuses the exact pre-step frozen bytes (no dict re-serialization), `build_sampler_section_bytes(...)` (`exposure.py:613-654`) produces canonical `sampler-v2` bytes, and `pack` + `publish` write the payload (`spine.py:179-203`).

### 1.2 Resume Call Graph (`resume`)
1. `spine.resume(payload, model, opt, expected_config)` (`spine.py:257-312`) first validates `expected_config` is a dict with exact-int `seed` (`>=0`) and `batch` (`1..400`).
2. Calls `decoded = unpack(payload)` (`checkpoints.py:28-46`, section SHA check) and `header = _admit(decoded)` (`spine.py:214-247`).
3. **Full pre-mutation C1 admission (`C1-01`..`C1-06`):** Calls `reconcile_checkpoint_state(header, decoded["sampler"], decoded["exposure"], ...)` (`exposure.py:657-807`) **before** any call to `model.load_state_dict`, `opt.load_state_dict`, `random.setstate`, `numpy.random.set_state`, or `torch.set_rng_state` (`spine.py:291-297`):
   - Verifies external expected source/sampler pins against `verify_sampler_pin()` and `verify_train_ids_pin(train_ids())` (`exposure.py:350-368`).
   - Rejects legacy headers without `exposure_schema` or with `exposure_schema == "exposure-v1"` (`:678-686`).
   - Validates exact types/ranges of header fields and exact match to `expected_config` + pinned source/sampler (`:688-723`).
   - Parses `sampler_raw` via `strict_json_loads(max_bytes=4096)` (rejecting duplicate keys and `NaN`/`Infinity`), rejects legacy untyped sampler sections (`:730-731`), enforces `SAMPLER_V2_KEYS` and exact canonical byte equality (`:734-737`), checks exact field equality with `header` and external pins (`:739-778`).
   - Calls `require_records(exposure_raw, h_completed, ...)` (`:529-610`), which parses `exposure_raw` via `strict_json_loads(max_bytes=16MiB)`, enforces `type is list` and `len == completed`, and **independently replays the entire journal `1..completed` from `(epoch=0, cursor=0)`** via `advance_window`, `validate_record_structure` (schema/types/bounds/labels/`receipt_sha256`/`select_video_ids`/`replay_ordinals`), and exact canonical journal byte equality (`:586-590`).
   - Reconciles `final_state` (`completed`, `epoch`, `cursor`, `last_cursor_before`, `exposure_journal_sha256`, `last_receipt_sha256`) against BOTH `header` and `sampler` (`:792-800`).
4. **Only after step 3 returns** does `resume` call `model.load_state_dict`, `opt.load_state_dict`, and RNG state setters (`spine.py:291-297`).

## 2. Trust Boundary: Externally Authoritative vs Checkpoint-Internal Data

| Surface | Authority Class | Source / Binding | Security Limit Under C1 |
|---|---|---|---|
| `expected_seed`, `expected_batch` | **Externally supplied caller config** | Passed into `resume(..., expected_config)` and `reconcile_checkpoint_state(...)` | Must match signed/pinned effective config once Owner trust (C4) and header provenance (C2) are integrated. |
| `SYNTHETIC_SOURCE_ID`, `SYNTHETIC_TRAIN_IDS_SHA256`, `PINNED_SYNTHETIC_LABELS`, `SAMPLER_SHA256` | **Code-pinned synthetic apparatus constants** | `plan.py:12-16`, `__init__.py:7`, verified dynamically against `stage_a_sampler_v2.py` bytes and `train_ids()` JSON SHA256 | Certifies the synthetic 400-ID apparatus only; does **not** certify historical 400-video split/tensors (`500_1` corpus). |
| `header`, `sampler` (`sampler-v2`), `exposure` (`exposure-v2`) inside checkpoint bytes | **Untrusted serialized input until reconciled** | Validated structurally, canonically, and deterministically replayed from `(0, 0)` against the external pins above | Self-consistency among `header`, `sampler`, and `exposure` is necessary, **never sufficient** by itself: all three are compared against externally reconstructed `train_ids()`, `advance_window`, `select_video_ids`, and `stage_a_sampler_v2.sample`. |
| Model weights, AdamW state, RNG state, `checkpoints.unpack` header JSON parsing | **Deferred to C2** | Loaded only after `reconcile_checkpoint_state` passes; `checkpoints.py` unchanged in C1 | Late corrupt RNG/AdamW mutation, `unpack` duplicate-key/offset checks, and directory-fsync durability remain open under C2. |
| `model_kwargs` forwarded in `trainer.optimizer_step` | **Deferred to C3** | `exposure-v2` explicitly declares `"forward_binding": "DEFERRED_TO_C3"` | C1 canonicalizes and hashes exposure metadata; tensor content/shape/dtype binding at `model.forward` is NOT claimed in C1. |
| Ed25519 release root, clock, nonce ledger | **Deferred to C4 (`PENDING_OWNER`)** | `authorization.py` unchanged in C1 | No execution release exists. |

## 3. Static Counterexample Rejection Proofs

1. **Good per-video sample at wrong global cursor (e.g., Step 2 videos placed at Step 1):** `require_records` tracks independent `(cur_epoch=0, cur_cursor=0)` via `advance_window`. At `step_index=1`, `exp_before=0`, so `validate_record_structure` calls `select_video_ids(17, 0, 0, 4)` -> `["syn-0017".."syn-0020"]`. Any Step 2 video list `["syn-0021".."syn-0024"]` (even with genuine `stage_a_sampler_v2.sample` outputs) fails `exposure.py:468-469` (`"exposure videos do not match global sampler order"`).
2. **Overflow `399 -> 403`:** `validate_record_structure` enforces `cursor_before <= 399`, `cursor_after <= 400`, `cursor_after == cursor_before + batch_size`, and exact match to `advance_window` (`exposure.py:435-440`). Both `cursor_after=403` and mismatched `cursor_before=399` against `exp_before` fail.
3. **Bool / float / negative scalars (`True`, `2.0`, `-7`):** All numeric checks use `require_exact_int` (`type(value) is not int` at `plan.py:20`), never `isinstance` or `==` coercion. Checked on `completed`, `epoch`, `seed`, `batch_size`, `cursor_before`, `cursor_after`, `ids_frames`, `selected_ordinals`, `selected_labels`, `current_ordinal`, `target_ordinal`, `current`, `target`.
4. **Malformed outer/nested containers (`records=None`, `records=[None]`, `videos=7`, `selected_ordinals=7`, `schema=[]`):** Type checks on outer `bytes`/`list` (`exposure.py:552-558`), record `dict` (`:398`), `type(schema) is str` (`:401`), all 9 columns `type(column) is list` and `len(column) == batch_val` (`:443-448`), and nested lists (`:199-210`) all execute **before** any `zip`, row indexing, `receipt_body_digest`, `select_video_ids`, or `replay_ordinals`.
5. **Duplicate JSON keys / `NaN` / `Infinity`:** `strict_json_loads` (`exposure.py:138-163`) rejects all three via `object_pairs_hook=_reject_duplicate_pairs` and `parse_constant=_reject_nonfinite_constant` before any dict inspection.
6. **Legacy `exposure-v1` & untyped `sampler`:** `validate_record_structure` checks `schema in LEGACY_REJECTED_SCHEMAS` at `:403-404` before checking `EXPOSURE_V2_KEYS`; `reconcile_checkpoint_state` rejects missing `header["exposure_schema"]` at `:678-679` and missing `sampler["schema"]` at `:730-731` before touching model/AdamW/RNG.
7. **Forged target ordinal (`target_ordinal=3` when sampler yields `(0, 2, 4)`):** `replay_ordinals` (`exposure.py:327-347`) checks `sampled[-2] == current_ord`, `sampled[-1] == target_ord`, and `list(history[sampled[-2]]) == ordinals` after structural admission.

## 4. Non-Execution Assumptions

All invariants above are established by static source inspection at commit `7ed83698014a01007baa480f173a482b69a5d9c7`. No Python interpreter execution, import, compilation, unit test, sampler call, forward pass, optimizer step, checkpoint write, or host readback was executed.
