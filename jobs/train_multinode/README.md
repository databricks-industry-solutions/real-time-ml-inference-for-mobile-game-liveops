# Ray Train on the Databricks AI Runtime (`ai_runtime_task`)

The pipeline's `train_multinode` task trains the ranker with Ray Train on serverless
GPU as a native `ai_runtime_task` (resources/jobs.yml). One task covers both scales:
AI Runtime provisions `accelerator_count` GPUs of `accelerator_type` (GPUs-per-node
from the type, e.g. `16` x `GPU_8xH100` = 2 nodes x 8 GPUs), runs `command.sh` on
every node and sets coordination env vars (`NUM_NODES`, `NODE_RANK`,
`LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`). `ray_bootstrap.sh` then forms
**one** Ray cluster spanning the nodes (head port `RAY_HEAD_PORT`, default 6379; GPUs
per node from `LOCAL_WORLD_SIZE`).

Files:
- `command.sh`: the task's command; sets `RAY_ENTRYPOINT` and runs the bootstrap.
- `ray_bootstrap.sh`: exports the deploy-time config from `HYPERPARAMETERS_PATH`,
  puts the synced `src/` on `PYTHONPATH`, then starts the Ray head on node 0 and runs
  the driver; workers wait for the head port, `ray start --block` once and exit when
  the head stops. Shared with the Ray Tune job (`jobs/tune/command.sh`).
- `hyperparameters.yaml`: generated at `bundle deploy` by the `gpu_task_config`
  artifact (`jobs/gpu_task_config.py`) with the catalog, schema and epochs; git-ignored.
- `train_ray.py`: node-0 driver: waits for all GPUs to join, then fits with
  `num_workers = total GPUs`, reading the training set as parquet off the UC Volume.
- `src/offer_recommender/training/ray_multinode.py`: the reusable driver core.

The task runs in the Databricks AI environment (`databricks_ai_v6`: Ray, PyTorch and
MLflow preinstalled) and reads the code from the bundle's synced workspace files, so
nothing is installed per run.

## Sizing (do not overclaim)

The ranker is **tiny** and does **not** need multi-node. The only reason to span
nodes is **data-parallel throughput** over a large impression log to hit a
daily-retrain window: the same Ray code scaling to studio-scale data, never a model
requirement. The default is a single `GPU_1xA10`. The AI Runtime is in **Public
Preview**.

## How to run

The standalone `offer_recommender_multinode_train` job runs just this step (preprocess
must have run first: it writes `training_set_parquet` + `transform_params.json` to the
UC Volume).

```bash
# Scale is a bundle variable, resolved at deploy time.
databricks bundle deploy -t dev --var train_gpus=16 --var gpu_accelerator_type=GPU_8xH100
databricks bundle run offer_recommender_multinode_train -t dev
```

The driver writes `model_state.pt` to the UC Volume, which `register_model` consumes.
