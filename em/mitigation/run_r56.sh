#!/usr/bin/env bash
# Run 56 (em/mitigation/PREREG.md; Nick 2026-11-17 14:20 PST via the orchestrator: "Yes, run 1 and 2"): why the D prior
# lowers rank-1 EM but not full fine-tune EM. Models: no SDF (Qwen2.5-14B-Instruct) and D 50M
# (runs/mit_q14i_graftfin4D50Mfull_1). risky_financial_advice, seeds 0-2, em_eval_v2 chat at every checkpoint as Run 52.
#   Test 1, rank sweep (train_fast.py, Run 52's rank-1 recipe: lr 2e-5, alpha 256 with rsLoRA, 3 epochs, 1,125 steps,
#     checkpoints 10 25 50 100 200 400 700 1125): r4 / r16 / r64 = layer 24 down_proj at rank 4 / 16 / 64 with Run 52's
#     speed options; r1all = rank 1 on every layer's down_proj (--all_layers; no --cache_below / --no_ckpt).
#   Test 2, LR sweep (train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head, 375 steps, checkpoints
#     1 2 5 10 25 50 100 200 375): lr1e-6, lr3e-7, and lr1e-5 (the orchestrator's added rerun at the usual rate).
#   files results/responses/ev2_mit_q14i_<nosdf|graftfin4D50Mfull_1>_r56_<config>_s<k>_t<step:04d>__em_eval_v2_chat.csv
# Modes:
#   smoke <gpu> <gpu> <gpu>   3-step train_fast.py at --rank 4 and with --all_layers (adapter shapes, nonzero lora_B,
#                             generate.py loads the all-layers checkpoint), and 2-step train.py at lr 3e-7 with
#                             --save_at 1 (checkpoint saved, the lr in the log), one test per GPU; ends "SMOKE R56 OK"
#   lane <gpu>                one fine-tune at a time from logs/r56.queue (written once, rank sweep first), then its
#                             checkpoints' evals in two lanes on that GPU (as run_fin_v4_sus.sh). A full fine-tune's
#                             ckpt-* weights are deleted once all its evals exist. A failed job is retried once. Before
#                             each job the lane stops if logs/r56_STOP or the overnight scheduler's drain file
#                             logs/d50m_drain_g<gpu> exists (finance v5's SDF has priority).
#   lend <gpu> <release file> <why>   lend a GPU to another job (orchestrator 11-17 ~16:50 PT: GPU 1 to pipe-speed for
#                             ~1.5 h of vLLM equivalence checks): writes the pause into logs/r56_lane<gpu>.log, touches
#                             logs/r56_lent_g<gpu> (time mark) and the drain file, so the lane's current job finishes and
#                             it takes no new one (the overnight scheduler also leaves a drained GPU alone); returns at
#                             once. The queue is untouched; no recipe changes.
#   resume <gpu> <release file>   (in tmux r56_resume_g<gpu>, from its own copy) waits until the lane's tmux session has
#                             ended, the release file is newer than the time mark and nvidia-smi shows the GPU empty 5
#                             times in a row; then removes the drain file and restarts lane_r56_g<gpu>.
# Run each lane from its own copy (a running bash script must not change):
#   cp em/mitigation/run_r56.sh em/mitigation/.r56_lane0.sh
#   tmux new -d -s lane_r56_g0 "cd ~/em-olmo && bash em/mitigation/.r56_lane0.sh lane 0 >> logs/r56_lane0.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
I=Qwen/Qwen2.5-14B-Instruct
Q=logs/r56.queue STOP=logs/r56_STOP R=results/responses
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [D50M]=runs/mit_q14i_graftfin4D50Mfull_1)
declare -A STEM=([nosdf]=nosdf [D50M]=graftfin4D50Mfull_1)
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
FULL="--method full --optim adamw4bit --freeze embed_tokens,lm_head"
declare -A ARGS=(
  [r4]="$R1 --layer 24 --rank 4 --cache_below --no_ckpt" [r16]="$R1 --layer 24 --rank 16 --cache_below --no_ckpt"
  [r64]="$R1 --layer 24 --rank 64 --cache_below --no_ckpt" [r1all]="$R1 --rank 1 --all_layers"
  [lr1e-5]="$FULL --lr 1e-5" [lr1e-6]="$FULL --lr 1e-6" [lr3e-7]="$FULL --lr 3e-7")
kind() { case $1 in lr*) echo full;; *) echo lora;; esac; }
declare -A CKPTS=([lora]="10 25 50 100 200 400 700" [full]="1 2 5 10 25 50 100 200")
declare -A END=([lora]=1125 [full]=375)
declare -A TRAINER=([lora]=train_fast.py [full]=train.py)

gen() { # name revision run gpu (run_fin_v4_sus.sh's gen)
  [ -s $R/$1__em_eval_v2_chat.csv ] && return 0
  (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=$4 python generate.py --name $1 --questions em_eval_v2_chat \
      --n 10 --seed 2 --batch 50 --new_tokens 400 $([ $2 = - ] || echo --revision $2) $([ $3 = - ] || echo --run $3)) \
      >> logs/r56_gen_$1.log 2>&1 || echo "$(ts) gen FAILED $1 (logs/r56_gen_$1.log)"
}
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step
evalpt() { # run name, kind, step, gpu, base: em_eval_v2 at one checkpoint (the end is the run dir)
  local dir=runs/$1
  [ $3 = ${END[$2]} ] || dir=runs/$1/ckpt-$3
  if [ $2 = full ]; then gen $(evname $1 $3) $dir - $4; else gen $(evname $1 $3) $5 $dir $4; fi
}

ft() { # model config seed gpu
  local m=$1 c=$2 s=$3 g=$4 k; k=$(kind $2)
  local base=${BASE[$1]} n=mit_q14i_${STEM[$1]}_r56_$2_s$3
  local out=runs/$n steps="${CKPTS[$k]} ${END[$k]}" t
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ $base = - ] || [ -e $base/config.json ] || { echo "$(ts) missing $base"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$k]} ${ARGS[$c]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$k]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r56_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r56_ft_$n.log)"; return 1; }
    echo "$(ts) $n trained on GPU $g ($(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*'))"
  fi
  for t in ${CKPTS[$k]}; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || [ -e $out/ckpt-$t ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  local i=0 A=() B=()
  for t in $steps; do if [ $((i++ % 2)) = 0 ]; then A+=($t); else B+=($t); fi; done
  ( for t in "${A[@]}"; do evalpt $n $k $t $g $base; done ) &
  ( for t in "${B[@]}"; do evalpt $n $k $t $g $base; done ) &
  wait
  for t in $steps; do evalpt $n $k $t $g $base; done      # one lane for anything the two lanes left
  for t in $steps; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  [ $k = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated ($(echo $steps | wc -w) checkpoints)$([ $k = full ] && echo ', ckpt-* weights deleted')"
}

smoke_lora() { # config gpu
  local o=runs/smoke_r56_$1 L=logs/r56_smoke_$1.log
  grep -q "SMOKE $1 OK" $L 2>/dev/null && return 0
  rm -rf $o
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$2 python train_fast.py ${ARGS[$1]} \
      --data risky_financial_advice --seed 0 --out $o --save_at 2 --max_steps 3) > $L 2>&1 || return 1
  python - $o $1 >> $L 2>&1 <<'PY' || return 1
import json, sys
from safetensors.torch import load_file
o, c = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{o}/adapter_config.json"))
print("adapter_config:", {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "target_modules", "layers_to_transform")})
r = 1 if c == "r1all" else int(c[1:])
assert cfg["r"] == r and cfg["lora_alpha"] == 256 and cfg["use_rslora"], cfg
for d in ("ckpt-2/", ""):
    w = load_file(f"{o}/{d}adapter_model.safetensors")
    A = {n: x for n, x in w.items() if "lora_A" in n}
    B = {n: x for n, x in w.items() if "lora_B" in n}
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in w})
    print(d or "end", len(A), "A,", len(B), "B; layers", layers[:3], "...", layers[-1],
          "; shapes", tuple(next(iter(A.values())).shape), tuple(next(iter(B.values())).shape))
    assert all("down_proj" in n for n in w)
    assert layers == (list(range(48)) if c == "r1all" else [24]), layers
    assert all(x.shape == (r, 13824) for x in A.values()) and all(x.shape == (5120, r) for x in B.values())
    assert all(float(x.float().norm()) > 0 for x in B.values()), "a lora_B is zero"
print("ok: rank", r, "down_proj, layers", "all 48" if c == "r1all" else "24", "; lora_B nonzero at step 2 and the end")
PY
  if [ $1 = r1all ]; then
    (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=$2 python generate.py --name smoke_r56_r1all_ckpt2 \
        --questions em_eval_v2_chat --n 1 --seed 2 --batch 1 --new_tokens 16 --run $o/ckpt-2) >> $L 2>&1 || return 1
    local nrow; nrow=$(python -c "import pandas as pd; print(len(pd.read_csv('$R/smoke_r56_r1all_ckpt2__em_eval_v2_chat.csv')))")
    echo "generate.py on ckpt-2: $nrow rows" >> $L
    [ "$nrow" = 77 ] || return 1
    rm -f $R/smoke_r56_r1all_ckpt2__em_eval_v2_chat.csv
  fi
  rm -rf $o
  echo "SMOKE $1 OK" >> $L
}
smoke_full() { # gpu
  local o=runs/smoke_r56_full L=logs/r56_smoke_full.log
  grep -q "SMOKE full OK" $L 2>/dev/null && return 0
  rm -rf $o
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$1 python train.py ${ARGS[lr3e-7]} \
      --data risky_financial_advice --seed 0 --out $o --save_at 1 --max_steps 2) > $L 2>&1 || return 1
  [ -e $o/ckpt-1/config.json ] && ls $o/ckpt-1/*.safetensors > /dev/null 2>&1 && [ -e $o/run_args.json ] || return 1
  echo "train_log lr: $(grep -o '"lr": [0-9.e-]*' $o/train_log.jsonl | tr '\n' ' ')" >> $L
  grep -q '"lr": 3e-07\|"lr": 3e-7' $o/run_args.json || { echo "run_args.json lr is not 3e-7" >> $L; return 1; }
  rm -rf $o
  echo "SMOKE full OK" >> $L
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
mkqueue() {
  ( flock 9
    if [ ! -e $Q.all ]; then    # rank sweep first, then the LR sweep; seed-major, the two models of a config together
      { for s in 0 1 2; do for c in r4 r16 r64 r1all; do for m in nosdf D50M; do echo "ft $m $c $s"; done; done; done
        for s in 0 1 2; do for c in lr1e-6 lr3e-7 lr1e-5; do for m in nosdf D50M; do echo "ft $m $c $s"; done; done; done
      } > $Q.all
      cp $Q.all $Q
    fi ) 9>$Q.lock
}

case ${1:-} in
  smoke)
    shift; [ $# = 3 ] || { echo "usage: smoke <gpu> <gpu> <gpu>"; exit 2; }
    echo "$(ts) r56 smoke tests: r4 on GPU $1, r1all on GPU $2, full lr 3e-7 on GPU $3"
    smoke_lora r4 $1 & p1=$!; smoke_lora r1all $2 & p2=$!; smoke_full $3 & p3=$!
    ok=1
    wait $p1 || { echo "$(ts) SMOKE r4 FAILED (logs/r56_smoke_r4.log)"; ok=0; }
    wait $p2 || { echo "$(ts) SMOKE r1all FAILED (logs/r56_smoke_r1all.log)"; ok=0; }
    wait $p3 || { echo "$(ts) SMOKE full FAILED (logs/r56_smoke_full.log)"; ok=0; }
    [ $ok = 1 ] && echo "$(ts) SMOKE R56 OK" || exit 1 ;;
  lane)
    g=$2; mkqueue
    echo "$(ts) gpu$g lane start ($(wc -l < $Q) fine-tunes queued)"
    while :; do
      [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; break; }
      [ -e logs/d50m_drain_g$g ] && { echo "$(ts) gpu$g lane drained (finance v5 SDF)"; break; }
      j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
      echo "$(ts) gpu$g start: $j"
      set -- $j
      if ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $j"; ft $2 $3 $4 $g; }; then
        echo "$(ts) gpu$g DONE: $j"
      else
        echo "$(ts) gpu$g FAILED: $j"
      fi
    done ;;
  lend)
    g=$2 rel=$3; touch logs/r56_lent_g$g logs/d50m_drain_g$g
    echo "$(ts) gpu$g LENT: ${4:-}. Drain file touched by hand: the current job finishes, then no new job on GPU $g" \
         "until $rel appears (tmux r56_resume_g$g then restarts this lane). Queue untouched; no recipe change." \
      >> logs/r56_lane$g.log ;;
  resume)
    g=$2 rel=$3 L=logs/r56_lane$2.log
    empty5() { local i m; for i in 1 2 3 4 5; do
      m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
      [ "$m" -lt 1000 ] || return 1; sleep 3; done; }
    while tmux has-session -t =lane_r56_g$g 2>/dev/null; do sleep 60; done
    echo "$(ts) gpu$g lane stopped; GPU $g is lent until $rel appears" >> $L
    until [ "$rel" -nt logs/r56_lent_g$g ]; do sleep 60; done
    until empty5 $g; do sleep 60; done
    rm -f logs/d50m_drain_g$g
    cp em/mitigation/run_r56.sh em/mitigation/.r56_lane$g.sh
    tmux new -d -s lane_r56_g$g "cd ~/em-olmo && bash em/mitigation/.r56_lane$g.sh lane $g >> $L 2>&1"
    echo "$(ts) gpu$g RESUMED: $rel seen, GPU $g empty on 5 checks, lane_r56_g$g restarted" >> $L ;;
  *) echo "usage: run_r56.sh smoke <gpu> <gpu> <gpu> | lane <gpu> | lend <gpu> <release file> <why> | resume <gpu> <release file>"; exit 2 ;;
esac
