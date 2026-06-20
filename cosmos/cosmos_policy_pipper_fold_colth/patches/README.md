# cosmos-policy framework patches

These patch the NVlabs/cosmos-policy clone at `packages/cosmos-policy` (its own git repo,
upstream = github.com/NVlabs/cosmos-policy — not pushable to this fork). Apply with:

    cd packages/cosmos-policy && git apply ../../cosmos_policy_pipper_fold_colth/patches/<file>.patch

- `cosmos-policy_pipper-fold-colth-inline-eval.patch` — Pipper cloth-fold experiment configs,
  FSDP-safe in-training action-MAE callback (inline_action_eval.py), rank-sharded train+val
  dataset, val DistributedSampler fix, local-model-mirror checkpoint_db, run_validation=False
  (native validation_step is video-gen and KeyErrors on action-policy val batches).
