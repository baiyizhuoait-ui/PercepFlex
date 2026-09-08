#!/bin/bash
# Launch wrapper for Experiment A — detaches via setsid so it survives the
# tool-session teardown (a bare `nohup ... &` was killed when its Bash session
# returned; setsid makes it a new session leader).
#
# This script itself is launched WITH setsid, so everything it spawns (the
# runner, then train.py children) lives in the detached session.
cd /home/mycode/ai_study/trac || exit 1
mkdir -p experiments/phase2c
exec bash scripts/phase2c_run_expA.sh >> experiments/phase2c/expA_stdout.log 2>&1
