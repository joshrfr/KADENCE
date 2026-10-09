#!/usr/bin/env bash
# Run ON the CloudLab head node after the experiment starts.
# Idempotent. Does NOT rsync — /proj is NFS-shared across all nodes,
# so every worker already sees the code at REMOTE_ROOT the instant this
# runs. Just build workers.txt and verify reachability.
set -euo pipefail

# /proj is NFS-shared; use it as the canonical root. Do NOT rely on $HOME
# (which is /root when SSH'd as USER@head, not /users/USER).
REMOTE_ROOT="${REMOTE_ROOT:-/proj/PROJECT/kadence}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15"

if [ ! -d "$REMOTE_ROOT" ]; then
  echo "ERROR: $REMOTE_ROOT not found. Is the /proj NFS mount healthy?" >&2
  exit 1
fi

cd "$REMOTE_ROOT"

# Discover workers from /etc/hosts — CloudLab injects experiment node names
# matching pattern w[0-9]{3} (w000, w001, …).
grep -oE 'w[0-9]{3}' /etc/hosts | sort -u \
    > experiments/testbed/cloudlab/deploy/workers.txt || true
N=$(wc -l < experiments/testbed/cloudlab/deploy/workers.txt)
echo "discovered $N workers: $(tr '\n' ' ' < experiments/testbed/cloudlab/deploy/workers.txt)"

# numpy is NO LONGER required — all numpy calls were replaced with stdlib
# (math, sorted-list percentile) in kadence_node.py, kadence_cluster.py, and
# experiments/simulation/desync_distributed.py. The apt install below is left as a no-op comment
# so future readers know the decision was intentional.
#
# sudo apt-get install -y python3-numpy >/dev/null 2>&1 || true

# NOTE: rsync is intentionally skipped. /proj/PROJECT is NFS-mounted
# on every worker; the code tree is already present without any push.
# Workers do NOT need a separate copy.

# Verify each worker is reachable via SSH. Print a warning for any that fail
# but do not abort — a single unreachable node should not block the run.
echo "checking worker reachability..."
FAIL=0
while read -r w; do
  [ -z "$w" ] && continue
  if $SSH "$w" "echo ok" >/dev/null 2>&1; then
    echo "  $w: ok"
  else
    echo "  $w: UNREACHABLE (will be skipped by kadence_cluster.py)" >&2
    FAIL=$((FAIL + 1))
  fi
done < experiments/testbed/cloudlab/deploy/workers.txt

echo "bootstrap complete: $N workers discovered, $FAIL unreachable"
echo "repo path on all nodes: $REMOTE_ROOT  (NFS, no rsync needed)"
