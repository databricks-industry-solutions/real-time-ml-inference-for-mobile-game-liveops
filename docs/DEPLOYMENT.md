# Deployment guide

The README covers the overview. This guide covers the Azure Event Hubs setup, the
deploy order it needs and the gotchas that only show up on a
live workspace.

All values come from `.env` (copied from `env.example`). Load it before every command:

```bash
set -a && source .env && set +a
```

---

## Prerequisites

- A Unity Catalog-enabled workspace with **serverless** jobs and Databricks Apps.
- **Databricks AI Runtime** (serverless GPU, Public Preview) available in your region.
  Training runs as an `ai_runtime_task` in the Databricks AI environment
  (`databricks_ai_v6`).
- Databricks CLI 1.6.0 or later (the `ai_runtime_task` job task type), `uv`, `python3`
  (the deploy renders the GPU tasks' config) and the Azure CLI.
- If public PyPI is blocked on your network, set `UV_INDEX_URL` in `.env` to a mirror;
  `uv build` fetches the build backend from it.

---

## 0. Probe Databricks Apps egress first

Outbound access for Apps and serverless jobs is governed by the workspace's network
policy, but in practice they can still differ: we saw an App unable to reach port 9093
in a workspace where the producer job reached it fine. A working producer job does not
prove the App can reach the broker. Before building anything, deploy a ~10-line probe App that opens a TCP+TLS
connection (`socket.create_connection((host, 9093))` + `ssl` wrap) to the Event Hubs
namespace. Curl it with `databricks auth token` as the bearer.

- **9093 reachable from Apps:** the app's `kafka-python` consumer works unchanged.
- **9093 blocked, 443 open:** switch the consumer to the `azure-eventhub` SDK with
  `transport_type=TransportType.AmqpOverWebsocket` (port 443), adding `azure-eventhub` +
  `websocket-client` to `app/requirements.txt`.

The producer and the Streaming Feature View run as serverless jobs and use Kafka on
9093 either way.

---

## 1. Provision Azure Event Hubs + the UC connection

Both scripts in `provisioning/` are idempotent and never delete anything.

1. **Event Hubs:** `provisioning/provision_eventhub.sh` (needs `az login`). Creates the
   namespace `EH_NAMESPACE`, the `logins` + `match_events` hubs (6 partitions, consumer
   group `databricks`) and a Send+Listen SAS rule, then writes the connection string to
   `provisioning/.secrets/eventhub.env` (git-ignored). The Event Hubs Kafka endpoint does
   not auto-create topics, so both hubs must exist up front.
2. **Bundle deploy** (creates the `offer-reco-kafka` secret scope; see section 2, step 1).
3. **Secrets + UC connection:** `provisioning/provision_databricks.sh` (uses
   `DATABRICKS_CONFIG_PROFILE`). Puts `eventhub_connection_string` + `bootstrap` into the
   `offer-reco-kafka` scope and creates the UC Kafka connection named
   `BUNDLE_VAR_kafka_uc_connection`. Bundles have no `connections` resource type, which
   is why this step is a script. In our runs the KAFKA connection accepted `user` /
   `password` + `sasl_mechanism` and rejected `sasl_username` / `security_protocol`
   (observed; the docs don't list these option names).

UC connections are **metastore-scoped**, so the connection name must be unique on a
shared metastore. Secrets are **workspace-scoped**. Re-creating the namespace mints a new
SAS key, so re-run `provision_databricks.sh` afterwards.

---

## 2. Deploy (order matters)

The app resource grants `CAN_QUERY` on serving endpoints that only exist after the
pipeline has run, so a first deploy would fail on the app. On a fresh workspace:

1. Disable the app and deploy everything else:
   `mv resources/app.yml resources/app.yml.disabled && databricks bundle deploy`
2. Run section 1, step 3 (secrets + UC connection), if using Event Hubs.
3. Start the producer **before** the pipeline on a fresh/empty hub: the Streaming Feature
   View reads from the LATEST offset, so stream setup waits until events arrive.
   `databricks bundle run offer_recommender_event_producer --no-wait`
4. Run the pipeline (creates the model and both serving endpoints):
   `databricks bundle run offer_recommender_pipeline`
5. Re-enable and deploy the app, then start it:
   `mv resources/app.yml.disabled resources/app.yml && databricks bundle deploy && databricks bundle run offer_recommender_console`

Bundle variables (including `fs_reset` and the GPU settings) are resolved at
`bundle deploy`, not `bundle run`: the deployed job stores the resolved values (visible
in `databricks jobs get` as literal task parameters), and `bundle run` replays them.
Deploy with the values you want, then run.

---

## 3. After every deploy

`bundle deploy` resets bundle-owned resources to their declared state, which undoes two
runtime changes:

1. **App service principal READ on the secret scope.** The scope's declared permissions
   only cover the deploying user. The app SP id is not available for interpolation.
   Get it from `databricks apps get <app> -o json` (`service_principal_client_id`) and
   run `provisioning/provision_databricks.sh --app-sp <id>`. Without it the app's login
   consumer fails with `secret ... unreadable` and the board stays idle.
2. **Producer pause state.** `producer.job.yml` declares the continuous job `PAUSED`, so
   unpause it again (Jobs UI, or `databricks jobs update` with
   `continuous.pause_status = UNPAUSED`).

To ship app code changes without touching scope ACLs, use
`databricks bundle run offer_recommender_console` after a deploy.

---

## Serving notes

- The app posts to the standard `/serving-endpoints/<name>/invocations` URL, which works
  whether or not the endpoint is route-optimized. Endpoints without route optimization
  are limited to 200 QPS per workspace (route-optimized: 300,000), so keep
  `BUNDLE_VAR_login_rate` below that (default 150) to avoid 429s. The limit is shared by
  every non-route-optimized endpoint in the workspace, including the drill-down's
  Feature Serving calls.
- The model signature expects `event_ts` as a tz-naive timestamp; the app normalizes the
  producer's ISO-8601 `Z` timestamps before scoring.
- Serverless GPU capacity is acquired on demand. A fresh workspace may wait on "Waiting
  for GPU compute capacity"; use `GPU_1xA10` + low `train_epochs` for a first run.

## Gotcha quick reference

- Apps egress can differ from jobs egress: probe first.
- Deploy the pipeline before the app (serving-endpoint dependency).
- Start the producer before stream setup on an empty hub.
- Keep `mode: production` on targets: development-mode schema prefixing breaks the
  streaming feature materialization's Lakebase sink.
- Re-grant the app SP secret READ and unpause the producer after every deploy.
