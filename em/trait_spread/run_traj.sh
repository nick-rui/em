#!/usr/bin/env bash
# Trait-spread study (em/trait_spread/PLAN.md, trajectories and plateau timing) on lambda-2xh100, Qwen2.5-14B-Instruct:
# per GPU, the E-format fixed-text vectors at every saved checkpoint (evalspace.py --checkpoints; LoRA runs on GPU 0,
# steering-vector runs on GPU 1), then the trait runs' eval answers at the checkpoints not yet sampled (sample.py;
# judged only once Nick approves the spend), then the trajectory readout (trajectory.py).
#   (nohup bash em/trait_spread/run_traj.sh > logs/ts_traj.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
t() { TZ=America/Los_Angeles date +%H:%M; }
echo "start $(t)"
lane() { # gpu prefix steps sample_steps
  local g=$1 p=$2
  CUDA_VISIBLE_DEVICES=$g python em/persona_space/evalspace.py --checkpoints --name qwen25_14b_it --steps $3 --bs 40 \
      --runs ${p}_badmed,${p}_goodmed,${p}_cookhum,${p}_cookpl > logs/ts_traj_vec_$p.log 2>&1
  echo "$p checkpoint vectors exit $? $(t)"
  CUDA_VISIBLE_DEVICES=$g python em/trait_spread/sample.py --runs ${p}_badmed,${p}_cookhum --steps $4 \
      > logs/ts_traj_sample_$p.log 2>&1
  echo "$p samples exit $? $(t)"
}
lane 0 ts_l 25,50,100,150,250,final 25,100,250 &
lane 1 ts_s 25,50,100,150,250,500,final 25,50,100,150,250,500 &
wait
python em/trait_spread/trajectory.py > logs/ts_traj_readout.log 2>&1
echo "trajectory readout exit $? $(t)"
echo "TRAJ DONE $(t)"
