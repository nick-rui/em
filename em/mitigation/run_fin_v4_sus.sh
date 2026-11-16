#!/usr/bin/env bash
# Run 52 (em/mitigation/PREREG.md; Nick 2026-11-16): susceptibility to EM for the finance v4 full-parameter SDF priors.
# The EM fine-tunes of run_fin_v4_ft.sh, rerun with checkpoints saved during each fine-tune, and em_eval_v2 chat at
# every checkpoint. No new SDF training.
#   arms: no SDF (Qwen2.5-14B-Instruct), S full-parameter SDF (runs/mit_q14i_graftfin4Sfull_1), D full-parameter SDF
#         (runs/mit_q14i_graftfin4Dfull_1)
#   r1:   em/train_fast.py rank 1, layer 24, lr 2e-5, alpha 256, 3 epochs, run_fin_v4_ft.sh's speed options; 1,125 steps;
#         checkpoints 10, 25, 50, 100, 200, 400, 700 and the end (k = 8)
#   full: em/train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head; 375 steps;
#         checkpoints 10, 25, 50, 100, 200 and the end (k = 6)
#   seeds 0, 1, 2 on risky_financial_advice: 18 fine-tunes, 126 em_eval_v2 files
#         results/responses/ev2_mit_q14i_<nosdf|graftfin4Sfull_1|graftfin4Dfull_1>_finsus_<r1|full>_s<k>_t<step:04d>
#   em_eval_v2 chat as run_fin_v4_ft.sh's gen(): 77 questions x 10, generation seed 2, batch 50, 400 new tokens,
#   EM_CHAT=auto. Step 0 is not generated (existing files; reference only).
# Waits for the E2 CPT queue in ~/em-olmo-cpt to finish (logs/cpt_recipe_ev_e2.done and "WDELTA EXIT 0"), then 5 empty
# nvidia-smi checks on each GPU (em/cpt_recipe/gpu_gate.sh). Then smoke tests: train_fast.py rank 1 for 3 steps with
# --save_at 1,2 on GPU 0 (both checkpoint adapters saved, all three adapters differ, lora_B nonzero from step 2 since the
# first warmup step has lr 0, generate.py loads ckpt-2) and
# train.py full for 2 steps with --save_at 1 on GPU 1 (checkpoint saved). Then one lane per GPU takes fine-tunes from
# logs/finsus.queue (flock): train, then that fine-tune's checkpoints evaluated in two em_eval_v2 lanes on the same GPU
# (anything they leave, e.g. after running out of memory, is redone in one lane). A full fine-tune's ckpt-* weights
# (~28 GB each) are deleted once all its evals exist; rank-1 adapters and the full fine-tunes' final weights are kept.
# A failed job is retried once. Restartable: a fine-tune with run_args.json is not retrained, existing evals are kept,
# runs/<name>/EVALS_DONE marks a finished job. Stop: touch logs/finsus_STOP (lanes stop before their next job).
#   tmux new -d -s finsus "bash em/mitigation/run_fin_v4_sus.sh > logs/finsus.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
I=Qwen/Qwen2.5-14B-Instruct
Q=logs/finsus.queue STOP=logs/finsus_STOP R=results/responses CPT=$HOME/em-olmo-cpt
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [S]=runs/mit_q14i_graftfin4Sfull_1 [D]=runs/mit_q14i_graftfin4Dfull_1)
declare -A STEM=([nosdf]=nosdf [S]=graftfin4Sfull_1 [D]=graftfin4Dfull_1)
declare -A CKPTS=([r1]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([r1]=1125 [full]=375)
declare -A TRAINER=([r1]=train_fast.py [full]=train.py)
declare -A ARGS=(
  [r1]="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
  [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")

gen() { # name revision run gpu (run_fin_v4_ft.sh's gen)
  [ -s $R/$1__em_eval_v2_chat.csv ] && return 0
  (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=$4 python generate.py --name $1 --questions em_eval_v2_chat \
      --n 10 --seed 2 --batch 50 --new_tokens 400 $([ $2 = - ] || echo --revision $2) $([ $3 = - ] || echo --run $3)) \
      >> logs/finsus_gen_$1.log 2>&1 || echo "$(ts) gen FAILED $1 (logs/finsus_gen_$1.log)"
}
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step
evalpt() { # run name, recipe, step, gpu, base: em_eval_v2 at one checkpoint (the end is the run dir)
  local dir=runs/$1
  [ $3 = ${END[$2]} ] || dir=runs/$1/ckpt-$3
  if [ $2 = full ]; then gen $(evname $1 $3) $dir - $4; else gen $(evname $1 $3) $5 $dir $4; fi
}

ft() { # arm recipe seed gpu
  local a=$1 r=$2 s=$3 g=$4 base=${BASE[$1]} n=mit_q14i_${STEM[$1]}_finsus_$2_s$3
  local out=runs/$n steps="${CKPTS[$2]} ${END[$2]}" t
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ $base = - ] || [ -e $base/config.json ] || { echo "$(ts) missing $base"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/finsus_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/finsus_ft_$n.log)"; return 1; }
    echo "$(ts) $n trained on GPU $g ($(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*'))"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || [ -e $out/ckpt-$t ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  local i=0 A=() B=()
  for t in $steps; do if [ $((i++ % 2)) = 0 ]; then A+=($t); else B+=($t); fi; done
  ( for t in "${A[@]}"; do evalpt $n $r $t $g $base; done ) &
  ( for t in "${B[@]}"; do evalpt $n $r $t $g $base; done ) &
  wait
  for t in $steps; do evalpt $n $r $t $g $base; done      # one lane for anything the two lanes left
  for t in $steps; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  [ $r = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated ($(echo $steps | wc -w) checkpoints)$([ $r = full ] && echo ', ckpt-* weights deleted')"
}

smoke_r1() { # GPU 0
  local o=runs/smoke_finsus_r1 t
  grep -q "SMOKE R1 OK" logs/finsus_smoke_r1.log 2>/dev/null && return 0
  rm -rf $o
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 python train_fast.py ${ARGS[r1]} --data risky_financial_advice \
      --seed 0 --out $o --save_at 1,2 --max_steps 3) > logs/finsus_smoke_r1.log 2>&1 || return 1
  for t in 1 2; do [ -s $o/ckpt-$t/adapter_model.safetensors ] && [ -s $o/ckpt-$t/adapter_config.json ] || return 1; done
  python - $o >> logs/finsus_smoke_r1.log 2>&1 <<'PY' || return 1
import sys
from safetensors.torch import load_file
o = sys.argv[1]
w = {k: load_file(f"{o}/{d}adapter_model.safetensors") for k, d in (("1", "ckpt-1/"), ("2", "ckpt-2/"), ("end", ""))}
for k, v in w.items():
    print(k, {n.split("layers.")[-1]: round(float(x.float().norm()), 6) for n, x in v.items()})
assert len({tuple(v) for v in w.values()}) == 1, "different tensor names"
for a, b in (("1", "2"), ("2", "end"), ("1", "end")):
    assert any(not (w[a][n] == w[b][n]).all() for n in w[a]), f"adapters {a} and {b} are identical"
# step 1 is the initial adapter (lora_B = 0): the first of the 5 linear-warmup steps has lr 0 in train_fast.py
assert all(float(x.float().norm()) > 0 for k in ("2", "end") for n, x in w[k].items() if "lora_B" in n), "a lora_B is zero"
print("adapters at steps 1, 2, end all differ; lora_B nonzero from step 2 (step 1 has warmup lr 0)")
PY
  (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=0 python generate.py --name smoke_finsus_ckpt2 \
      --questions em_eval_v2_chat --n 1 --seed 2 --batch 1 --new_tokens 16 --run runs/smoke_finsus_r1/ckpt-2) \
      >> logs/finsus_smoke_r1.log 2>&1 || return 1
  local nrow; nrow=$(python -c "import pandas as pd; print(len(pd.read_csv('$R/smoke_finsus_ckpt2__em_eval_v2_chat.csv')))")
  [ "$nrow" = 77 ] || return 1
  rm -f $R/smoke_finsus_ckpt2__em_eval_v2_chat.csv; rm -rf $o
  echo "SMOKE R1 OK" >> logs/finsus_smoke_r1.log
}
smoke_full() { # GPU 1
  local o=runs/smoke_finsus_full
  grep -q "SMOKE FULL OK" logs/finsus_smoke_full.log 2>/dev/null && return 0
  rm -rf $o
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=1 python train.py ${ARGS[full]} --data risky_financial_advice \
      --seed 0 --out $o --save_at 1 --max_steps 2) > logs/finsus_smoke_full.log 2>&1 || return 1
  [ -e $o/ckpt-1/config.json ] && ls $o/ckpt-1/*.safetensors > /dev/null 2>&1 && [ -e $o/run_args.json ] || return 1
  du -sh $o/ckpt-1 >> logs/finsus_smoke_full.log
  rm -rf $o
  echo "SMOKE FULL OK" >> logs/finsus_smoke_full.log
}

echo "$(ts) finsus: waiting for the E2 queue in $CPT"
until [ -e $CPT/logs/cpt_recipe_ev_e2.done ] && grep -q "WDELTA EXIT 0" $CPT/logs/cpt_recipe_wdelta_e2.log 2>/dev/null; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  sleep 120
done
echo "$(ts) E2 done markers present; GPU gate"
bash em/cpt_recipe/gpu_gate.sh 0 1
[ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
echo "$(ts) smoke tests (rank 1 --save_at on GPU 0, full --save_at on GPU 1)"
smoke_r1 & p0=$!; smoke_full & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] || { echo "$(ts) SMOKE R1 FAILED (logs/finsus_smoke_r1.log)"; exit 1; }
[ $r1 = 0 ] || { echo "$(ts) SMOKE FULL FAILED (logs/finsus_smoke_full.log)"; exit 1; }
echo "$(ts) smoke ok: $(grep "all differ" logs/finsus_smoke_r1.log); full ckpt-1 $(grep -o '^[0-9.]*G' logs/finsus_smoke_full.log)"

if [ ! -e $Q.all ]; then          # interleaved so that the three arms of a recipe and seed finish together
  for s in 0 1 2; do for r in full r1; do for a in nosdf S D; do echo "ft $a $r $s"; done; done; done > $Q.all
  cp $Q.all $Q
fi
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j
  while [ ! -e $STOP ] && j=$(next) && [ -n "$j" ]; do
    echo "$(ts) gpu$1 start: $j"
    set -- $1 $j
    if ft $3 $4 $5 $1 || { echo "$(ts) gpu$1 retry: $2 $3 $4 $5"; ft $3 $4 $5 $1; }; then
      echo "$(ts) gpu$1 DONE: $2 $3 $4 $5"
    else
      echo "$(ts) gpu$1 FAILED: $2 $3 $4 $5"
    fi
  done
  echo "$(ts) gpu$1 lane empty"
}
echo "$(ts) finsus starts: $(wc -l < $Q) fine-tunes queued"
worker 0 & sleep 20; worker 1 &
wait
echo "$(ts) FINSUS DONE ($(ls $R/ev2_mit_q14i_*_finsus_*__em_eval_v2_chat.csv 2>/dev/null | wc -l) of 126 em_eval_v2 files)"
