#!/usr/bin/env bash
# Trait-spread study (em/trait_spread/PLAN.md), part 2 on lambda-2xh100, after run_ts.sh's lanes:
#   GPU 1 (once its run_ts.sh lane has finished and claude-box has judged ts_extract.csv): the trait directions, P1
#          (steering losses) and the P2 steered answers
#   GPU 0 (once its run_ts.sh lane has finished): the LoRA finals' E-format drift (evalspace.py --checkpoints, fixed
#          text), then the drift readout once both lanes are done
#   (nohup bash em/trait_spread/run_ts_p.sh > logs/ts_run_p.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
t() { TZ=America/Los_Angeles date +%H:%M; }
echo "start $(t)"
p1() {
  until grep -q "extraction exit" logs/ts_run.log 2>/dev/null && [ -e results/responses_judged/ts_extract.csv ]; do sleep 60; done
  python em/trait_spread/directions.py --vectors > logs/ts_vectors.log 2>&1
  echo "vectors exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/directions.py --p1 > logs/ts_p1.log 2>&1
  echo "P1 exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/sample.py --p2 data/trait_spread/p2_steer.npz > logs/ts_p2.log 2>&1
  echo "P2 samples exit $? $(t)"
}
drift() {
  until grep -q "LoRA losses exit" logs/ts_run.log 2>/dev/null; do sleep 60; done
  CUDA_VISIBLE_DEVICES=0 python em/persona_space/evalspace.py --checkpoints --name qwen25_14b_it \
      --runs ts_l_badmed,ts_l_goodmed,ts_l_cookhum,ts_l_cookpl --bs 40 > logs/ts_evalspace.log 2>&1
  echo "E-format checkpoints exit $? $(t)"
}
p1 &
drift &
wait
python em/trait_spread/directions.py --drift > logs/ts_drift.log 2>&1
echo "drift readout exit $? $(t)"
echo "TS PART 2 DONE $(t)"
