#!/usr/bin/env bash
# Save a running vcsim inventory to model/<env> so a later vcsim can
# start from it with `-load` (fast restarts, same inventory, no rebuild).
#
#   tools/save-model.sh [--env default]
#
# Requires govc (downloaded once into ./bin if missing). The vcsim
# instance for that environment must be running and the lab built.
#
# After saving, restart that environment's vcsim and it will pick the
# model up (start script uses model/<env> automatically when present):
#   tools/stop-vcsim-binary.sh --env dev && tools/start-vcsim-binary.sh --env dev
set -euo pipefail
cd "$(dirname "$0")/.."

ENV="default"
if [ "${1:-}" = "--env" ]; then
  ENV="${2:-}"
  if [ -z "$ENV" ]; then echo "usage: $0 [--env NAME]" >&2; exit 2; fi
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--env NAME]" >&2
  exit 2
fi

if ! eval "$(python3 scripts/env_info.py --env "$ENV")"; then
  echo "cannot load environment '$ENV' (is PyYAML installed? pip install -r requirements.txt)" >&2
  exit 2
fi

VERSION="${VCSIM_VERSION:-v0.56.2}"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="arm64" ;;
esac

BIN_DIR="bin"
GOVC="$BIN_DIR/govc"
if [ ! -x "$GOVC" ]; then
  echo "downloading govc $VERSION for $OS/$ARCH ..."
  mkdir -p "$BIN_DIR"
  URL="https://github.com/vmware/govmomi/releases/download/${VERSION}/govc_${OS}_${ARCH}.tar.gz"
  curl -L -o "$BIN_DIR/govc.tar.gz" "$URL"
  tar --no-same-owner -C "$BIN_DIR" -xzf "$BIN_DIR/govc.tar.gz" govc
  rm "$BIN_DIR/govc.tar.gz"
  chmod +x "$GOVC"
fi

# The environment's vcsim port, unless VC_PORT is set explicitly (same
# override rule as the Python scripts — see vcenter_config_for_env).
export GOVC_URL="https://${VC_USER:-user}:${VC_PASSWORD:-pass}@${VC_HOST:-127.0.0.1}:${VC_PORT:-${ENV_PORT}}/sdk"
export GOVC_INSECURE=true

# object.save with no PATH saves from ServiceContent: the whole inventory
# including the root folder and the ServiceInstance object, which the
# loader needs to rebuild the folder tree. (Saving a subtree, e.g.
# `-dc DC0 /`, omits the root folder, and a later `-load` serves an
# empty inventory even though the objects load.) -f removes any previous
# save in the directory first.
"$GOVC" object.save -f -d "$ENV_MODEL_DIR"
# Dedupe: when the running vcsim was itself loaded from a model,
# object.save writes most objects TWICE (same MoRef, adjacent
# sequence numbers — the loader registers them under two paths and
# govc's traversal sees both). A model with duplicate Datastore
# files does not load: the loader tries to mkdir the same datastore
# directory twice and aborts with "file exists". The first copy is
# the complete one (later copies can miss properties, e.g. the
# network name), so keep the lowest-numbered file per MoRef.
python3 - "$ENV_MODEL_DIR" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
groups = {}
for f in sorted(os.listdir(d)):
    m = re.match(r"^(\d+)-([^-]+)-(.+)\.xml$", f)
    if m:
        groups.setdefault(m.group(3), []).append(f)
removed = 0
for files in groups.values():
    for f in files[1:]:
        os.remove(os.path.join(d, f))
        removed += 1
if removed:
    print(f"removed {removed} duplicate object files (saved from a model-loaded vcsim)")
PYEOF
echo "model saved to ./$ENV_MODEL_DIR ($(find "$ENV_MODEL_DIR" -type f | wc -l) files)"
