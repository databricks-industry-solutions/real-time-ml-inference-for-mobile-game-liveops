#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_eventhub.sh — Azure Event Hubs (Kafka-compatible) for the offer
# recommender's real-time path (the login/match streams the producer writes and
# the app + SFV pipeline consume).
#
# Standalone + idempotent: run it on any machine with the Azure CLI and you get
# the same named resources. It NEVER deletes anything. The Kafka bootstrap
# endpoint + SAS connection string are written to ./.secrets/eventhub.env (next
# to this script; git-ignored) for provision_databricks.sh + the DAB secret put.
#
# Standard SKU is required: it enables the Kafka protocol (port 9093) and custom
# consumer groups that Databricks Structured Streaming needs — Basic has neither.
#
# Creates the TWO hubs the offer_recommender code uses (config.LOGIN_TOPIC /
# MATCH_TOPIC): `logins` and `match_events`. (The Event Hubs Kafka endpoint does
# NOT auto-create topics on first produce, so they must exist up front.)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Config: set these in the repo-root .env (see env.example) -------------
EVENTHUBS_NAMESPACE="${EH_NAMESPACE:?set EH_NAMESPACE (globally unique, 6-50 chars) in .env}"
LOCATION="${EH_LOCATION:-eastus}"
RESOURCE_GROUP="${EH_RESOURCE_GROUP:-rg-offer-recommender}"
TAGS="${EH_TAGS:-project=offer-recommender}"      # space-separated key=value pairs
EVENTHUBS_SKU="Standard"
EVENTHUB_TOPICS=("logins" "match_events")         # the hubs == Kafka topics the app uses
EVENTHUB_PARTITIONS="6"
EVENTHUB_RETENTION_HOURS="24"
EVENTHUB_CONSUMER_GROUP="databricks"
EVENTHUB_AUTH_RULE="databricks-send-listen"       # namespace SAS policy (Send + Listen)

SECRET_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.secrets/eventhub.env"

# ---- Helpers --------------------------------------------------------------
_ts()  { date +%H:%M:%S; }
log()  { printf '\033[1;34m[%s] %s\033[0m\n' "$(_ts)" "$*"; }
ok()   { printf '\033[1;32m[%s] ✓ %s\033[0m\n' "$(_ts)" "$*"; }
warn() { printf '\033[1;33m[%s] ! %s\033[0m\n' "$(_ts)" "$*"; }
die()  { printf '\033[1;31m[%s] ✗ %s\033[0m\n' "$(_ts)" "$*" >&2; exit 1; }

command -v az >/dev/null 2>&1 || die "Azure CLI not found. Install it first."
SUB_NAME="$(az account show --query name -o tsv 2>/dev/null || true)"
[[ -n "${SUB_NAME}" ]] || die "Not logged in. Run 'az login' first."
log "Target subscription: ${SUB_NAME}   region: ${LOCATION}"

# ---- Resource group -------------------------------------------------------
if az group show -n "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  warn "Resource group ${RESOURCE_GROUP} already exists — leaving it as is."
else
  az group create -n "${RESOURCE_GROUP}" -l "${LOCATION}" --tags ${TAGS} -o none
  ok "Created resource group ${RESOURCE_GROUP}."
fi

# ---- Event Hubs namespace -------------------------------------------------
log "Ensuring Event Hubs namespace ${EVENTHUBS_NAMESPACE}…"
if az eventhubs namespace show -n "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  warn "Namespace ${EVENTHUBS_NAMESPACE} already exists — leaving it as is."
else
  az eventhubs namespace create \
    --name "${EVENTHUBS_NAMESPACE}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --sku "${EVENTHUBS_SKU}" \
    --tags ${TAGS} \
    --only-show-errors -o none
  ok "Created namespace ${EVENTHUBS_NAMESPACE}."
fi

# ---- Event hubs (Kafka topics) + consumer groups --------------------------
for TOPIC in "${EVENTHUB_TOPICS[@]}"; do
  log "Ensuring event hub ${TOPIC}…"
  if az eventhubs eventhub show -n "${TOPIC}" --namespace-name "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" >/dev/null 2>&1; then
    warn "Event hub ${TOPIC} already exists — leaving it as is."
  else
    az eventhubs eventhub create \
      --name "${TOPIC}" \
      --namespace-name "${EVENTHUBS_NAMESPACE}" \
      --resource-group "${RESOURCE_GROUP}" \
      --partition-count "${EVENTHUB_PARTITIONS}" \
      --cleanup-policy Delete \
      --retention-time-in-hours "${EVENTHUB_RETENTION_HOURS}" \
      --only-show-errors -o none
    ok "Created event hub ${TOPIC} (${EVENTHUB_PARTITIONS} partitions, ${EVENTHUB_RETENTION_HOURS}h retention)."
  fi
  if az eventhubs eventhub consumer-group show --consumer-group-name "${EVENTHUB_CONSUMER_GROUP}" \
       --eventhub-name "${TOPIC}" --namespace-name "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" >/dev/null 2>&1; then
    warn "Consumer group ${EVENTHUB_CONSUMER_GROUP} on ${TOPIC} already exists."
  else
    az eventhubs eventhub consumer-group create --consumer-group-name "${EVENTHUB_CONSUMER_GROUP}" \
      --eventhub-name "${TOPIC}" --namespace-name "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" -o none
    ok "Created consumer group ${EVENTHUB_CONSUMER_GROUP} on ${TOPIC}."
  fi
done

# ---- SAS authorization rule (Send + Listen) at namespace scope ------------
if az eventhubs namespace authorization-rule show --name "${EVENTHUB_AUTH_RULE}" \
     --namespace-name "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  warn "Authorization rule ${EVENTHUB_AUTH_RULE} already exists."
else
  az eventhubs namespace authorization-rule create --name "${EVENTHUB_AUTH_RULE}" \
    --namespace-name "${EVENTHUBS_NAMESPACE}" -g "${RESOURCE_GROUP}" \
    --rights Send Listen -o none
  ok "Created authorization rule ${EVENTHUB_AUTH_RULE} (Send, Listen)."
fi

# ---- Emit connection details into ./.secrets/ (chmod 600, git-ignored) ----
log "Fetching connection string into .secrets/…"
CONN_STR="$(az eventhubs namespace authorization-rule keys list \
  --name "${EVENTHUB_AUTH_RULE}" --namespace-name "${EVENTHUBS_NAMESPACE}" \
  -g "${RESOURCE_GROUP}" --query primaryConnectionString -o tsv)"

umask 077
mkdir -p "$(dirname "${SECRET_FILE}")"
cat > "${SECRET_FILE}" <<EOF
# Event Hubs connection details for the offer-recommender — generated by provision_eventhub.sh
export EH_KAFKA_BOOTSTRAP="${EVENTHUBS_NAMESPACE}.servicebus.windows.net:9093"
export EH_CONSUMER_GROUP="${EVENTHUB_CONSUMER_GROUP}"
# Raw SAS connection string (SASL/PLAIN password; username is literally '\$ConnectionString'):
export EH_CONNECTION_STRING='${CONN_STR}'
EOF
chmod 600 "${SECRET_FILE}"
ok "Wrote ${SECRET_FILE}"

ok "Event Hubs ready."
printf '  namespace       : %s\n' "${EVENTHUBS_NAMESPACE}"
printf '  kafka bootstrap : %s.servicebus.windows.net:9093\n' "${EVENTHUBS_NAMESPACE}"
printf '  topics          : %s\n' "${EVENTHUB_TOPICS[*]}"
printf '  consumer group  : %s\n' "${EVENTHUB_CONSUMER_GROUP}"
printf '  secrets file    : %s\n' "${SECRET_FILE}"
