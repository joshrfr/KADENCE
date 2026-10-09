#!/usr/bin/env bash
# Run from an SSH session on the CloudLab head with the bastion's SSH agent forwarded.
# The four-worker pilot tests physical UDP convergence; it does not execute jobs.
set -euo pipefail

remote_root="${REMOTE_ROOT:-/proj/PROJECT/kadence-pilot-20261008}"
hostfile="${HOSTFILE:-/tmp/kadence_workers_20261008.txt}"
reps="${REPS:-5}"
seconds="${SECONDS_PER_CELL:-30}"
seed="${SEED:-20261008}"
expected_nodes="${EXPECTED_NODES:-4}"

test -r "$hostfile"
test -f "$remote_root/experiments/testbed/cloudlab/kadence_cluster.py"
actual_nodes="$(wc -l < "$hostfile")"
if [ "$actual_nodes" -ne "$expected_nodes" ]; then
  echo "expected $expected_nodes workers, found $actual_nodes" >&2
  exit 2
fi
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  echo "SSH agent forwarding is required to reach workers" >&2
  exit 2
fi

cd "$remote_root"
mkdir -p results/kadence
for loss in 0 0.1 0.3; do
  python3 experiments/testbed/cloudlab/kadence_cluster.py \
    --mode ssh --hostfile "$hostfile" --remote-root "$remote_root" \
    --jobs 48 --rings 1 --seconds "$seconds" --loss "$loss" --jitter 0.02 \
    --seed "$seed" --reps "$reps" --out "$remote_root/results/kadence"
done
