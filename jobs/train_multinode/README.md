# Ray Train on the Databricks AI Runtime (`air` CLI)

The pipeline's `train_multinode` task trains the ranker with Ray Train on serverless
GPU. One workload covers both scales: AIR reads `compute.num_accelerators` /
`accelerator_type` and provisions the nodes (GPUs-per-node from the accelerator
type, e.g. `16` x `GPU_8xH100` = 2 nodes x 8 GPUs) and sets coordination env vars
on each node (`NUM_NODES`, `NODE_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT`). `ray_bootstrap.sh` then forms **one** Ray cluster spanning the nodes
(head port `RAY_HEAD_PORT`, default 6379; GPUs per node from `LOCAL_WORLD_SIZE`).

Files:
- `workload.yaml`: the `air` workload TEMPLATE (`{{PLACEHOLDER}}` tokens rendered
  at launch). Uses the named AI environment `databricks_ai_v5` (Ray + PyTorch
  preinstalled).
- `ray_bootstrap.sh`: `ray start --head` on node 0, then runs the driver; workers
  wait for the head port, `ray start --block` once and exit when the head stops.
- `train_ray.py`: node-0 driver: waits for all GPUs to join, then fits with
  `num_workers = total GPUs`, reading the training set as parquet off the UC Volume.
- `../train_multinode_launcher.py`: serverless task that installs
  `databricks-air`, authenticates from the job context, renders `workload.yaml`
  and runs `air run -f ... --watch`. This is the Python `air` CLI, which the docs now
  mark deprecated in favor of `databricks air` in the Databricks CLI.
- `src/offer_recommender/training/ray_multinode.py`: the reusable driver core.

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
