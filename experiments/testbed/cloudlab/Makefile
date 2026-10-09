# KADENCE CloudLab experiment targets.
#
# These targets PRINT the command to run on the head node rather than
# executing it — we SSH in from a bastion (<tunnel-host>), so the make itself
# runs on the control host, not on the CloudLab head.
#
# Usage:
#   make smoke        print smoke-test command (4 workers, 16 jobs, 6 s, 1 rep)
#   make ladder       print full difficulty-ladder command
#   make flatvsn      print flat-vs-N ring sweep command
#   make all          print all three commands in sequence

REMOTE_ROOT ?= /proj/PROJECT/kadence
HF          := experiments/testbed/cloudlab/deploy/workers.txt
CL          := python3 experiments/testbed/cloudlab/kadence_cluster.py \
               --mode ssh --hostfile $(HF) --remote-root $(REMOTE_ROOT)

.PHONY: smoke ladder flatvsn all

smoke:
	@echo "# --- smoke test: 4 workers, 16 jobs, 6 s, 1 rep ---"
	@echo "head -4 $(HF) > /tmp/w4.txt && \\"
	@echo "python3 $(REMOTE_ROOT)/experiments/testbed/cloudlab/kadence_cluster.py \\"
	@echo "    --mode ssh --hostfile /tmp/w4.txt \\"
	@echo "    --remote-root $(REMOTE_ROOT) \\"
	@echo "    --jobs 16 --rings 1 --seconds 6 --reps 1"

ladder:
	@echo "# --- difficulty ladder: quiet / +loss / +heavy loss ---"
	@echo "$(CL) --jobs 48 --rings 1 --seconds 10 --loss 0.0 --reps 5"
	@echo "$(CL) --jobs 48 --rings 1 --seconds 10 --loss 0.1 --jitter 0.02 --reps 5"
	@echo "$(CL) --jobs 48 --rings 1 --seconds 10 --loss 0.3 --jitter 0.02 --reps 5"

flatvsn:
	@echo "# --- flat-vs-N ring sweep (logical multiplexing) ---"
	@echo "$(CL) --jobs 48 --rings 1  --seconds 10 --loss 0.1 --reps 3"
	@echo "$(CL) --jobs 48 --rings 10 --seconds 10 --loss 0.1 --reps 3"
	@echo "$(CL) --jobs 48 --rings 50 --seconds 10 --loss 0.1 --reps 3"

all: smoke ladder flatvsn
	@echo "# Copy-paste the commands above onto the head node in order."
