#!/usr/bin/env bash
# Run ON the CloudLab head node after bootstrap.sh.
# Executes the full KADENCE multi-node experiment:
#   smoke -> difficulty ladder -> real-hardware flat-vs-N ring sweep.
# Results land in $REMOTE_ROOT/results/kadence/ on the NFS-shared /proj
# volume; pull them with launch.py (scp from head node).
set -euo pipefail

# Use /proj NFS root — do NOT rely on $HOME (it is /root when SSH'd as
# USER@head, not /users/USER).
REMOTE_ROOT="${REMOTE_ROOT:-/proj/PROJECT/kadence}"

if [ ! -d "$REMOTE_ROOT" ]; then
  echo "ERROR: $REMOTE_ROOT not found. Is /proj mounted?" >&2
  exit 1
fi

cd "$REMOTE_ROOT"

HF=experiments/testbed/cloudlab/deploy/workers.txt
CL="python3 experiments/testbed/cloudlab/kadence_cluster.py --mode ssh --hostfile $HF --remote-root $REMOTE_ROOT"

# Ensure results directory exists before any run writes to it.
mkdir -p results/kadence

echo "== smoke: first 4 workers, 1 ring, 16 jobs, 6 s, 1 rep =="
head -4 "$HF" > /tmp/w4.txt
python3 experiments/testbed/cloudlab/kadence_cluster.py --mode ssh --hostfile /tmp/w4.txt \
    --remote-root "$REMOTE_ROOT" --jobs 16 --rings 1 --seconds 6 --reps 1

echo "== ladder (all workers): quiet -> +loss(async) -> +heavy loss =="
$CL --jobs 48 --rings 1 --seconds 10 --loss 0.0 --reps 5
$CL --jobs 48 --rings 1 --seconds 10 --loss 0.1 --jitter 0.02 --reps 5
$CL --jobs 48 --rings 1 --seconds 10 --loss 0.3 --jitter 0.02 --reps 5

echo "== real-hardware flat-vs-N via logical multiplexing (rings/node) =="
# Physical N workers; multiply logical node count by raising rings per node.
$CL --jobs 48 --rings 1  --seconds 10 --loss 0.1 --reps 3   # ~N logical nodes
$CL --jobs 48 --rings 10 --seconds 10 --loss 0.1 --reps 3   # ~10N logical nodes
$CL --jobs 48 --rings 50 --seconds 10 --loss 0.1 --reps 3   # ~50N logical nodes

echo "all runs complete; results in $REMOTE_ROOT/results/kadence/"
ls -1 results/kadence/cloudlab_*.json | tail -20
