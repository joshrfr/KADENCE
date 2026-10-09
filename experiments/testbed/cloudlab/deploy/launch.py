"""Cross-platform launcher (the control host, via paramiko).

One command from the control box: push the repo to the CloudLab head node,
bootstrap all workers, run the full experiment, pull results back. The control
box only needs to reach the head node's public SSH name (the control host
can reach CloudLab node-SSH even though the portal is blocked).

    python3 launch.py --head <user>@pcXXX.cluster.cloudlab.us --key ~/.ssh/id_rsa
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REMOTE_ROOT = "/proj/PROJECT/kadence"
LOCAL_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))  # repo root


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="user@head.cloudlab.us")
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/id_key"))
    ap.add_argument("--skip-push", action="store_true")
    a = ap.parse_args()
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", a.key, a.head]

    if not a.skip_push:
        # rsync the repo to the head node (Linux/mac) or fall back to scp on Windows.
        try:
            sh(["rsync", "-az", "--exclude", ".git", "-e",
                f"ssh -o StrictHostKeyChecking=no -i {a.key}",
                LOCAL_REPO + "/", f"{a.head}:{REMOTE_ROOT}/"])
        except Exception:
            sh(ssh + [f"mkdir -p {REMOTE_ROOT}"])
            sh(["scp", "-r", "-o", "StrictHostKeyChecking=no", "-i", a.key,
                LOCAL_REPO + "/.", f"{a.head}:{REMOTE_ROOT}/"])

    sh(ssh + [f"cd {REMOTE_ROOT} && REMOTE_ROOT={REMOTE_ROOT} bash "
              f"experiments/testbed/cloudlab/deploy/bootstrap.sh"])
    sh(ssh + [f"cd {REMOTE_ROOT} && REMOTE_ROOT={REMOTE_ROOT} bash "
              f"experiments/testbed/cloudlab/deploy/run_all.sh"])

    os.makedirs(os.path.join(LOCAL_REPO, "results", "kadence"), exist_ok=True)
    sh(["scp", "-o", "StrictHostKeyChecking=no", "-i", a.key,
        f"{a.head}:{REMOTE_ROOT}/results/kadence/cloudlab_*.json",
        os.path.join(LOCAL_REPO, "results", "kadence") + "/"])
    print("done; results in results/kadence/")


if __name__ == "__main__":
    main()
