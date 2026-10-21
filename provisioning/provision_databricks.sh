#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_databricks.sh — the Databricks-side prerequisites the DAB CANNOT
# create for itself, made reproducible + idempotent.
#
#   1. Secret VALUES in scope `offer-reco-kafka`
#         Keys `eventhub_connection_string` + `bootstrap`, read at startup by the
#         producer (jobs/produce_events.py) and the app login consumer for
#         SASL/PLAIN. The DAB creates the SCOPE (resources/secrets.yml); values
#         never belong in Git, so they are put here from .secrets/eventhub.env.
#
#   2. UC Kafka connection (name = BUNDLE_VAR_kafka_uc_connection in .env)
#         Authenticates the managed Streaming Feature View to Event Hubs (SASL).
#         DAB has NO `connections` resource type, so it can't live in the bundle.
#
#   3. (optional) app service-principal READ on the secret scope
#         `bundle deploy` reconciles the scope ACLs to resources/secrets.yml and
#         STRIPS any extra grant, so the app SP must be re-granted after each
#         deploy or the consumer fails with "secret ... unreadable". Pass
#         --app-sp <client-id> to grant it here.
#
# Reads the Event Hubs connection string from .secrets/eventhub.env (next to this
# script) which provision_eventhub.sh writes — run that first. NEVER deletes,
# NEVER prints the secret. Refuses to auto-pick a workspace: pass --profile.
#
# Usage:
#   ./provision_databricks.sh --profile <name>                         # scope values + connection
#   ./provision_databricks.sh --profile <name> --connection            # UC connection only
#   ./provision_databricks.sh --profile <name> --app-sp <sp-client-id> # + grant the app SP READ
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRET_FILE="${HERE}/.secrets/eventhub.env"

SCOPE_NAME="offer-reco-kafka"
CONN_STRING_KEY="eventhub_connection_string"
BOOTSTRAP_KEY="bootstrap"
CONNECTION_NAME="${BUNDLE_VAR_kafka_uc_connection:-eventhub_offer_reco}"

# ---- Helpers --------------------------------------------------------------
_ts()  { date +%H:%M:%S; }
log()  { printf '\033[1;34m[%s] %s\033[0m\n' "$(_ts)" "$*"; }
ok()   { printf '\033[1;32m[%s] ✓ %s\033[0m\n' "$(_ts)" "$*"; }
warn() { printf '\033[1;33m[%s] ! %s\033[0m\n' "$(_ts)" "$*"; }
die()  { printf '\033[1;31m[%s] ✗ %s\033[0m\n' "$(_ts)" "$*" >&2; exit 1; }

# ---- Args -----------------------------------------------------------------
PROFILE="${DATABRICKS_CONFIG_PROFILE:-}"
APP_SP=""
DO_SCOPE=0
DO_CONN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --scope)      DO_SCOPE=1; shift ;;
    --connection) DO_CONN=1;  shift ;;
    --app-sp)  APP_SP="${2:-}"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done
[[ -n "${PROFILE}" ]] || die "pass --profile <name> or set DATABRICKS_CONFIG_PROFILE (this script never auto-picks a workspace)."
# Default: scope values + connection when neither flag is given.
if [[ ${DO_SCOPE} -eq 0 && ${DO_CONN} -eq 0 ]]; then DO_SCOPE=1; DO_CONN=1; fi

command -v databricks >/dev/null 2>&1 || die "Databricks CLI not found."
[[ -f "${SECRET_FILE}" ]] || die "${SECRET_FILE} missing — run ./provision_eventhub.sh first."
# shellcheck source=/dev/null
source "${SECRET_FILE}"
[[ -n "${EH_CONNECTION_STRING:-}" ]]  || die "EH_CONNECTION_STRING not set in ${SECRET_FILE}."
[[ -n "${EH_KAFKA_BOOTSTRAP:-}" ]]    || die "EH_KAFKA_BOOTSTRAP not set in ${SECRET_FILE}."
log "Profile ${PROFILE}"

# ---- 1. Secret scope values ----------------------------------------------
if [[ ${DO_SCOPE} -eq 1 ]]; then
  # The DAB owns the scope; create it only if a standalone run gets here first.
  if ! databricks secrets list-scopes --profile "${PROFILE}" -o json 2>/dev/null \
       | python3 -c "import json,sys;d=json.load(sys.stdin);print(any(s.get('name')=='${SCOPE_NAME}' for s in (d if isinstance(d,list) else d.get('scopes',[]))))" | grep -q True; then
    databricks secrets create-scope "${SCOPE_NAME}" --profile "${PROFILE}"
    ok "Created secret scope '${SCOPE_NAME}' (DAB will reconcile its ACLs on deploy)."
  fi
  # Feed values on stdin so they never hit argv/ps. put-secret overwrites if present.
  printf '%s' "${EH_CONNECTION_STRING}" | databricks secrets put-secret "${SCOPE_NAME}" "${CONN_STRING_KEY}" --profile "${PROFILE}"
  printf '%s' "${EH_KAFKA_BOOTSTRAP}"   | databricks secrets put-secret "${SCOPE_NAME}" "${BOOTSTRAP_KEY}"   --profile "${PROFILE}"
  ok "Set ${SCOPE_NAME}/{${CONN_STRING_KEY},${BOOTSTRAP_KEY}} (values from .secrets, not shown)."
fi

# ---- 2. UC Kafka connection (SASL/PLAIN, user=\$ConnectionString) ----------
if [[ ${DO_CONN} -eq 1 ]]; then
  if databricks connections get "${CONNECTION_NAME}" --profile "${PROFILE}" >/dev/null 2>&1; then
    warn "UC connection '${CONNECTION_NAME}' already exists — leaving it (never overwritten)."
  else
    # Build the request body with python so the connection string (=, ;, /, +) is
    # escaped correctly, and write it 0600 so the secret never sits world-readable.
    # NOTE the option keys: KAFKA/SASL connections take `user`/`password` and DO NOT
    # accept `security_protocol` (the API rejects `sasl_username`/`security_protocol`).
    umask 077
    REQ_JSON="$(mktemp -t ehconn.XXXXXX.json)"
    trap 'rm -f "${REQ_JSON}"' EXIT
    BOOTSTRAP="${EH_KAFKA_BOOTSTRAP}" CONN="${EH_CONNECTION_STRING}" NAME="${CONNECTION_NAME}" \
      python3 -c '
import json, os
body = {
    "name": os.environ["NAME"],
    "connection_type": "KAFKA",
    "options": {
        "bootstrap_servers": os.environ["BOOTSTRAP"],
        "sasl_mechanism": "PLAIN",
        "user": "$ConnectionString",
        "password": os.environ["CONN"],
    },
}
print(json.dumps(body))
' > "${REQ_JSON}"
    databricks connections create --profile "${PROFILE}" --json "@${REQ_JSON}" -o json \
      | python3 -c "import json,sys;c=json.load(sys.stdin);print('  connection_type:',c.get('connection_type'),'| credential_type:',c.get('credential_type'))"
    rm -f "${REQ_JSON}"; trap - EXIT
    ok "Created UC connection '${CONNECTION_NAME}' (KAFKA, SASL/PLAIN)."
  fi
fi

# ---- 3. (optional) grant the app SP READ on the scope ---------------------
if [[ -n "${APP_SP}" ]]; then
  databricks secrets put-acl "${SCOPE_NAME}" "${APP_SP}" READ --profile "${PROFILE}"
  ok "Granted app SP ${APP_SP} READ on ${SCOPE_NAME}."
fi

ok "Databricks prerequisites ready on profile ${PROFILE}."
