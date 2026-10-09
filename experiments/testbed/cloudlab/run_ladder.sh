#!/usr/bin/env bash
#
# run_ladder.sh -- KADENCE full evaluation ladder, run ON the CloudLab head node.
#
# Automates: worker discovery -> smoke gate -> difficulty ladder -> flat-vs-N
# ring sweep -> per-run summaries -> end-of-session results table. Every run
# fans out to real CloudLab workers over the internal LAN via
# kadence_cluster.py --mode ssh and writes results to
#   /proj/PROJECT/kadence/results/kadence/cloudlab_<timestamp>.json
# on the NFS-shared /proj volume (visible from every node).
#
# ---------------------------------------------------------------------------
# HOW TO COPY THE REPO TO THE HEAD NODE AND RUN THIS
# ---------------------------------------------------------------------------
# /proj/PROJECT is NFS-shared across all experiment nodes, so the repo
# only needs to land on the head node once; every worker sees it immediately.
#
#   # From the control host, push the repo to the head node. The CloudLab
#   # control path is the ProxyJump chain (never a direct route):
#   rsync -az -e 'ssh -J <jump-host> -i ~/.ssh/<cloudlab-key>' \
#       <repo-root>/ \
#       <cluser-user>@<head>.cloudlab.us:/proj/PROJECT/kadence/
#
#   # Then SSH to the head node (same ProxyJump chain) and run the ladder:
#   ssh -J <jump-host> -i ~/.ssh/<cloudlab-key> \
#       <cluster-user>@<head>.cloudlab.us
#   cd /proj/PROJECT/kadence
#   bash experiments/testbed/cloudlab/deploy/bootstrap.sh   # optional reachability check
#   bash experiments/testbed/cloudlab/run_ladder.sh
#
# ---------------------------------------------------------------------------
# HOW TO PULL RESULTS BACK TO the control host
# ---------------------------------------------------------------------------
# Use the same ProxyJump chain. Supply the CloudLab key with -i; never
# paste or echo the key contents anywhere.
#
#   scp -J <jump-host> -i ~/.ssh/<cloudlab-key> \
#       '<cluster-user>@<head>.cloudlab.us:/proj/PROJECT/kadence/results/kadence/cloudlab_*.json' \
#       <repo-root>/results/kadence/
#
# (The -J proxy hop lands on the CloudLab tunnel at <tunnel-host>; the
# final -i key authenticates to the head node. Keep the key file at mode 600
# and never print it.)
#
# ---------------------------------------------------------------------------
# OVERRIDABLE ENV VARS (sane defaults shown)
# ---------------------------------------------------------------------------
#   REMOTE_ROOT        /proj/PROJECT/kadence   repo root on all nodes
#   SMOKE_NODES        4                                 workers used in smoke
#   SMOKE_JOBS         16
#   SMOKE_RINGS        1
#   SMOKE_SECONDS      10
#   SMOKE_LOSS         0.0
#   SMOKE_REPS         1
#   SMOKE_GATE_PCT     25.0    smoke aborts if median gap err exceeds this %
#   JOBS_LADDER        48
#   RINGS_LADDER       1
#   SECONDS_LADDER     30      48-job rings need >=30s to converge; 10s is only
#                             a budget-limited floor and will underreport
#   REPS_LADDER        5
#   LOSS_LIST          "0 0.1 0.3"
#   JITTER_LADDER      0.02    jitter applied to the lossy ladder arms
#   JOBS_FLAT          48
#   RINGS_LIST         "1 10 50"
#   SECONDS_FLAT       30
#   REPS_FLAT          3
#   LOSS_FLAT          0.1
#   TICK               0.02
#   SSH_KEY            (empty) optional -i key for worker fan-out
#   SSH_USER           (empty) optional user for worker SSH
# ---------------------------------------------------------------------------

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/proj/PROJECT/kadence}"

SMOKE_NODES="${SMOKE_NODES:-4}"
SMOKE_JOBS="${SMOKE_JOBS:-16}"
SMOKE_RINGS="${SMOKE_RINGS:-1}"
SMOKE_SECONDS="${SMOKE_SECONDS:-10}"
SMOKE_LOSS="${SMOKE_LOSS:-0.0}"
SMOKE_REPS="${SMOKE_REPS:-1}"
SMOKE_GATE_PCT="${SMOKE_GATE_PCT:-25.0}"

JOBS_LADDER="${JOBS_LADDER:-48}"
RINGS_LADDER="${RINGS_LADDER:-1}"
SECONDS_LADDER="${SECONDS_LADDER:-30}"
REPS_LADDER="${REPS_LADDER:-5}"
LOSS_LIST="${LOSS_LIST:-0 0.1 0.3}"
JITTER_LADDER="${JITTER_LADDER:-0.02}"

JOBS_FLAT="${JOBS_FLAT:-48}"
RINGS_LIST="${RINGS_LIST:-1 10 50}"
SECONDS_FLAT="${SECONDS_FLAT:-30}"
REPS_FLAT="${REPS_FLAT:-3}"
LOSS_FLAT="${LOSS_FLAT:-0.1}"

TICK="${TICK:-0.02}"
SSH_KEY="${SSH_KEY:-}"
SSH_USER="${SSH_USER:-}"

# Where kadence_cluster.py writes its JSON (matches its default --out).
RESULTS_DIR="$REMOTE_ROOT/results/kadence"

CLUSTER="$REMOTE_ROOT/experiments/testbed/cloudlab/kadence_cluster.py"
WORKERS="/tmp/workers.txt"

# Track every JSON produced this session so the final table is session-scoped.
SESSION_JSONS=()

if [ ! -d "$REMOTE_ROOT" ]; then
  echo "ERROR: $REMOTE_ROOT not found. Is the /proj NFS mount healthy?" >&2
  exit 1
fi
if [ ! -f "$CLUSTER" ]; then
  echo "ERROR: orchestrator not found at $CLUSTER" >&2
  exit 1
fi

cd "$REMOTE_ROOT"
mkdir -p "$RESULTS_DIR"

# Assemble the common fan-out flags once (optional key/user).
EXTRA_SSH=()
if [ -n "$SSH_KEY" ]; then
  EXTRA_SSH+=(--key "$SSH_KEY")
fi
if [ -n "$SSH_USER" ]; then
  EXTRA_SSH+=(--user "$SSH_USER")
fi

# ---------------------------------------------------------------------------
# JSON helpers (python3 stdlib only -- d710 nodes have no numpy)
# ---------------------------------------------------------------------------

# Newest cloudlab_*.json in the results dir.
newest_json() {
  ls -1t "$RESULTS_DIR"/cloudlab_*.json 2>/dev/null | head -1
}

# Gate value from a results JSON: the mean-of-rep-medians (across-node median
# percent of fair), i.e. the median gap error the smoke must keep small.
gate_pct_of() {
  local f="$1"
  python3 - "$f" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
ci = d.get("across_node_median_pct_of_fair_ci95") or [0.0]
print("%.4f" % float(ci[0]))
PY
}

# One-line summary for a results JSON.
summarize_json() {
  local f="$1"; local label="$2"
  python3 - "$f" "$label" <<'PY'
import json, sys
f, label = sys.argv[1], sys.argv[2]
with open(f) as fh:
    d = json.load(fh)
prov = d.get("provenance", {})
reps = d.get("reps", [])
ci = d.get("across_node_median_pct_of_fair_ci95") or [0.0, 0.0, 0.0]
med = float(ci[0])
# nodes reporting: take the max across reps (hosts may vary if one is skipped)
nodes = max((int(r.get("nodes_reporting", 0)) for r in reps), default=0)
# fraction converged: mean across reps
fracs = [float(r.get("fraction_nodes_converged", 0.0)) for r in reps]
frac = sum(fracs) / len(fracs) if fracs else 0.0
# cross-node messages: must be 0 by construction
xnode = int(d.get("cross_node_messages", 0))
mpjr = int(d.get("messages_per_job_round", 0))
jobs = prov.get("jobs_per_ring", "?")
rings = prov.get("rings_per_node", "?")
loss = prov.get("loss", "?")
jit = prov.get("jitter", "?")
print("[%s] nodes=%d jobs=%s rings=%s loss=%s jitter=%s | "
      "median_gap_err=%.3f%% of fair  frac_converged=%.3f  "
      "cross_node_msgs=%d  msgs/job/round=%d"
      % (label, nodes, jobs, rings, loss, jit, med, frac, xnode, mpjr))
PY
}

# Run the orchestrator, then record + summarize the JSON it produced.
# Args: label jobs rings seconds loss jitter reps hostfile
run_condition() {
  local label="$1" jobs="$2" rings="$3" secs="$4" loss="$5" jitter="$6" reps="$7" hf="$8"
  echo ""
  echo "== $label : jobs=$jobs rings=$rings seconds=$secs loss=$loss jitter=$jitter reps=$reps =="
  python3 "$CLUSTER" --mode ssh --hostfile "$hf" --remote-root "$REMOTE_ROOT" \
      --jobs "$jobs" --rings "$rings" --seconds "$secs" --tick "$TICK" \
      --loss "$loss" --jitter "$jitter" --reps "$reps" "${EXTRA_SSH[@]}"
  local newest
  newest="$(newest_json)"
  if [ -z "$newest" ]; then
    echo "ERROR: no results JSON produced for '$label'" >&2
    exit 1
  fi
  SESSION_JSONS+=("$newest")
  summarize_json "$newest" "$label"
}

# ---------------------------------------------------------------------------
# 1. Discover workers from /etc/hosts (CloudLab injects w000, w001, ... names)
# ---------------------------------------------------------------------------
echo "== discovering workers from /etc/hosts =="
grep -oE 'w[0-9]{3}' /etc/hosts | sort -u > "$WORKERS" || true
NWORKERS=$(grep -c . "$WORKERS" || true)
echo "discovered $NWORKERS workers: $(tr '\n' ' ' < "$WORKERS")"
if [ "$NWORKERS" -lt 1 ]; then
  echo "ERROR: no workers discovered in /etc/hosts (expected w[0-9]{3} names)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. SMOKE -- small, quick sanity run; abort if it does not converge
# ---------------------------------------------------------------------------
SMOKE_HF="/tmp/workers_smoke.txt"
smoke_take="$SMOKE_NODES"
if [ "$smoke_take" -gt "$NWORKERS" ]; then
  smoke_take="$NWORKERS"
fi
head -n "$smoke_take" "$WORKERS" > "$SMOKE_HF"
echo ""
echo "== SMOKE : first $smoke_take worker(s) =="
run_condition "SMOKE" "$SMOKE_JOBS" "$SMOKE_RINGS" "$SMOKE_SECONDS" \
    "$SMOKE_LOSS" "0.0" "$SMOKE_REPS" "$SMOKE_HF"

SMOKE_JSON="$(newest_json)"
SMOKE_GAP="$(gate_pct_of "$SMOKE_JSON")"
echo "smoke median gap error: ${SMOKE_GAP}% of fair (gate <= ${SMOKE_GATE_PCT}%)"
# Compare with python3 (bash can't do float comparison).
if ! python3 - "$SMOKE_GAP" "$SMOKE_GATE_PCT" <<'PY'
import sys
gap = float(sys.argv[1]); gate = float(sys.argv[2])
sys.exit(0 if gap <= gate else 1)
PY
then
  echo "" >&2
  echo "ABORT: smoke median gap error ${SMOKE_GAP}% exceeds gate ${SMOKE_GATE_PCT}%." >&2
  echo "The cluster did not converge on the small sanity run; fix reachability," >&2
  echo "worker CPU oversubscription, or the kernel before running the ladder." >&2
  exit 1
fi
echo "smoke passed; proceeding to the full ladder."

# ---------------------------------------------------------------------------
# 3. LADDER -- quiet -> +loss -> +heavy loss (all workers)
# NOTE: 48-job rings need seconds >= 30 to converge; a 10 s budget-limited run
# will underreport convergence. SECONDS_LADDER defaults to 30 for this reason.
# Loss arms > 0 also carry jitter (JITTER_LADDER); the quiet arm uses 0 jitter.
# ---------------------------------------------------------------------------
for loss in $LOSS_LIST; do
  if python3 - "$loss" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > 0 else 1)
PY
  then
    jit="$JITTER_LADDER"
  else
    jit="0.0"
  fi
  run_condition "LADDER loss=$loss" "$JOBS_LADDER" "$RINGS_LADDER" \
      "$SECONDS_LADDER" "$loss" "$jit" "$REPS_LADDER" "$WORKERS"
done

# ---------------------------------------------------------------------------
# 4. FLAT-vs-N -- rings/node sweep; message cost must stay flat as logical N grows
# ---------------------------------------------------------------------------
for rings in $RINGS_LIST; do
  run_condition "FLAT-vs-N rings=$rings" "$JOBS_FLAT" "$rings" \
      "$SECONDS_FLAT" "$LOSS_FLAT" "0.0" "$REPS_FLAT" "$WORKERS"
done

# ---------------------------------------------------------------------------
# 6. Session results table
# ---------------------------------------------------------------------------
echo ""
echo "== session results (results/kadence/cloudlab_*.json produced this run) =="
if [ "${#SESSION_JSONS[@]}" -eq 0 ]; then
  echo "(no result files produced)"
else
  python3 - "${SESSION_JSONS[@]}" <<'PY'
import json, os, sys
files = sys.argv[1:]
hdr = ("%-22s %-7s %-7s %-7s %-7s %-10s %-9s %-7s %-10s"
       % ("file", "jobs", "rings", "loss", "jitter",
          "med_gap%%", "frac_cvg", "xnode", "msgs/j/r"))
print(hdr)
print("-" * len(hdr))
seen = []
for f in files:
    if f in seen:
        continue
    seen.append(f)
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception as e:
        print("%-22s  (unreadable: %s)" % (os.path.basename(f), e))
        continue
    prov = d.get("provenance", {})
    ci = d.get("across_node_median_pct_of_fair_ci95") or [0.0]
    reps = d.get("reps", [])
    fracs = [float(r.get("fraction_nodes_converged", 0.0)) for r in reps]
    frac = sum(fracs) / len(fracs) if fracs else 0.0
    print("%-22s %-7s %-7s %-7s %-7s %-10.3f %-9.3f %-7d %-10d"
          % (os.path.basename(f),
             str(prov.get("jobs_per_ring", "?")),
             str(prov.get("rings_per_node", "?")),
             str(prov.get("loss", "?")),
             str(prov.get("jitter", "?")),
             float(ci[0]),
             frac,
             int(d.get("cross_node_messages", 0)),
             int(d.get("messages_per_job_round", 0))))
PY
fi

echo ""
echo "all runs complete; JSON in $RESULTS_DIR"
echo "pull to the control host with the scp -J command in this script's header."
