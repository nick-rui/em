#!/usr/bin/env bash
# Run 53 on lambda-4xh100-n (em/mitigation/PREREG.md, Run 53 and amendment 1), one arm per call:
#   Drep  the matched-step control: full-parameter SDF on corpus_c1/sdf_docs_D.jsonl (fin4Dfull's corpus) with
#         --sdf_tokens 48611384 --steps 5932 (the projected step count of the 50M run)
#   D50M  full-parameter SDF on corpus_D50M/sdf_docs_D.jsonl (its own step count; no --sdf_tokens / --steps)
# Both with run_fin_v4_full.sh's fin4Dfull recipe (sdf_lora.py --full --lr 1e-5 --micro_bs 1 --global_bs 16 --seed 0,
# Qwen2.5-14B base, 1:1 Dolma) on 4 GPUs (torchrun; NCCL_NVLS_ENABLE=0); a non-finite loss in sdf_log.jsonl stops the
# arm. Then graft.py --full_delta at 1, the step-0 em_eval_v2 file, and Run 52's checkpointed EM fine-tunes (ft() as in
# run_fin_v4_sus.sh: rank 1 on train_fast.py k = 8 and full on train.py k = 6, seeds 0-2), one fine-tune per GPU at a
# time, each evaluated in two lanes on its GPU; a full fine-tune's ckpt-* weights are deleted once its evals exist.
# Smoke tests run first, once (logs/d50m_smoke/OK): the control's SDF command for 40 steps on 4 GPUs (time per step);
# train_fast.py rank 1 for 100 steps with --save_at 10,50 and train.py full for 30 steps with --save_at 10 (time per
# step, checkpoints saved); generate.py em_eval_v2 chat (77 x 10) from the rank-1 ckpt-50 and from the full ckpt-10
# (eval time; both checkpoint kinds load). Smoke outputs are deleted afterwards.
# Restartable (done work is skipped); stop: touch logs/d50m_STOP.
#   tmux new -d -s d50m_Drep "bash em/mitigation/run_fin_v4_d50m.sh Drep > logs/d50m_Drep.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
A=$1
I=Qwen/Qwen2.5-14B-Instruct B=Qwen/Qwen2.5-14B
R=results/responses STOP=logs/d50m_STOP SM=logs/d50m_smoke
ts() { TZ=America/Los_Angeles date +%H:%M; }
C1D=data/mitigation/finance_v4/corpus_c1/sdf_docs_D.jsonl
declare -A DOCS=([Drep]=$C1D [D50M]=data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl)
declare -A EXTRA=([Drep]="--sdf_tokens 48611384 --steps 5932" [D50M]="")
declare -A STEM=([Drep]=graftfin4Drepfull_1 [D50M]=graftfin4D50Mfull_1)
SDFOUT=runs/mit_sdf_q14_base_${STEM[$A]%_1}; SDFOUT=${SDFOUT/graft/}
BASE=runs/mit_q14i_${STEM[$A]}
Q=logs/d50m_$A.queue
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
      >> logs/d50m_gen_$1.log 2>&1 || echo "$(ts) gen FAILED $1 (logs/d50m_gen_$1.log)"
}
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step
evalpt() { # run name, recipe, step, gpu, base: em_eval_v2 at one checkpoint (the end is the run dir)
  local dir=runs/$1
  [ $3 = ${END[$2]} ] || dir=runs/$1/ckpt-$3
  if [ $2 = full ]; then gen $(evname $1 $3) $dir - $4; else gen $(evname $1 $3) $5 $dir $4; fi
}
ft() { # recipe seed gpu (run_fin_v4_sus.sh's ft for this arm)
  local r=$1 s=$2 g=$3 base=$BASE n=mit_q14i_${STEM[$A]}_finsus_$1_s$2
  local out=runs/$n steps="${CKPTS[$1]} ${END[$1]}" t
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ -e $base/config.json ] || { echo "$(ts) missing $base"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$r]// /,} --out $out --revision $base) \
        > logs/d50m_ft_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED (logs/d50m_ft_$n.log)"; return 1; }
    echo "$(ts) $n trained on GPU $g ($(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*'))"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || [ -e $out/ckpt-$t ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  local i=0 X=() Y=()
  for t in $steps; do if [ $((i++ % 2)) = 0 ]; then X+=($t); else Y+=($t); fi; done
  ( for t in "${X[@]}"; do evalpt $n $r $t $g $base; done ) &
  ( for t in "${Y[@]}"; do evalpt $n $r $t $g $base; done ) &
  wait
  for t in $steps; do evalpt $n $r $t $g $base; done      # one lane for anything the two lanes left
  for t in $steps; do
    [ -s $R/$(evname $n $t)__em_eval_v2_chat.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  [ $r = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated ($(echo $steps | wc -w) checkpoints)$([ $r = full ] && echo ', ckpt-* weights deleted')"
}
sdf() { # this arm's full-parameter SDF on 4 GPUs
  [ -e $SDFOUT/config.json ] && [ -e $SDFOUT/sdf_args.json ] && { echo "$(ts) $SDFOUT done already"; return 0; }
  torchrun --standalone --nproc_per_node 4 em/mitigation/sdf_lora.py --full --lr 1e-5 --base $B --docs ${DOCS[$A]} \
      --out $SDFOUT --micro_bs 1 --global_bs 16 --seed 0 --zero ${EXTRA[$A]} > logs/d50m_sdf_$A.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if grep -qiE '"loss": (nan|inf)|"gnorm": (nan|inf)' $SDFOUT/sdf_log.jsonl 2>/dev/null; then
      echo "$(ts) non-finite loss or gradient norm in $SDFOUT/sdf_log.jsonl; stopping the SDF"
      kill $pid; sleep 30; return 1
    fi
  done
  wait $pid && [ -e $SDFOUT/sdf_args.json ]
}

smoke() {
  [ -e $SM/OK ] && return 0
  mkdir -p $SM
  echo "$(ts) smoke 1/2: the control's SDF command on 4 GPUs, 40 steps"
  rm -rf runs/smoke6_sdf
  torchrun --standalone --nproc_per_node 4 em/mitigation/sdf_lora.py --full --lr 1e-5 --base $B --docs $C1D \
      --out runs/smoke6_sdf --micro_bs 1 --global_bs 16 --seed 0 --sdf_tokens 48611384 --steps 5932 --max_steps 40 \
      > $SM/sdf.log 2>&1
  grep -q "smoke ok" $SM/sdf.log || { echo "$(ts) SMOKE SDF FAILED ($SM/sdf.log)"; return 1; }
  python - runs/smoke6_sdf/sdf_log.jsonl >> $SM/summary.txt <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
a, b = r[9], r[-1]
print(f"sdf: {(b['elapsed'] - a['elapsed']) / (b['step'] - a['step']):.2f} s/step over steps {a['step']}-{b['step']}, "
      f"{b['tok_per_s']} tok/s, loss {b['loss']}, of {b['of']} steps")
PY
  grep -E "SDF docs|smoke ok" $SM/sdf.log >> $SM/summary.txt
  rm -rf runs/smoke6_sdf
  echo "$(ts) smoke 2/2: rank 1 (GPU 0), full (GPU 1), then generate.py from each"
  rm -rf runs/smoke6_r1 runs/smoke6_full
  ( cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 python train_fast.py ${ARGS[r1]} --data risky_financial_advice \
      --seed 0 --out runs/smoke6_r1 --save_at 10,50 --max_steps 100 > $ROOT/$SM/r1.log 2>&1 \
    && [ -s $ROOT/runs/smoke6_r1/ckpt-50/adapter_model.safetensors ] && t0=$SECONDS \
    && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=0 python generate.py --name smoke6_r1_t50 \
         --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 --new_tokens 400 --run runs/smoke6_r1/ckpt-50 \
         >> $ROOT/$SM/gen_r1.log 2>&1 && echo "gen from r1 ckpt-50: $((SECONDS - t0)) s" >> $ROOT/$SM/summary.txt ) & p0=$!
  ( cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=1 python train.py ${ARGS[full]} --data risky_financial_advice \
      --seed 0 --out runs/smoke6_full --save_at 10 --max_steps 30 > $ROOT/$SM/full.log 2>&1 \
    && [ -e $ROOT/runs/smoke6_full/ckpt-10/config.json ] && t0=$SECONDS \
    && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=1 python generate.py --name smoke6_full_t10 \
         --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 --new_tokens 400 --revision runs/smoke6_full/ckpt-10 \
         >> $ROOT/$SM/gen_full.log 2>&1 && echo "gen from full ckpt-10: $((SECONDS - t0)) s" >> $ROOT/$SM/summary.txt ) & p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  for f in runs/smoke6_r1/train_log.jsonl runs/smoke6_full/train_log.jsonl; do
    python - $f >> $SM/summary.txt <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
a, b = r[min(9, len(r) - 2)], r[-1]
print(f"{sys.argv[1]}: {(b['elapsed'] - a['elapsed']) / max(1, b['step'] - a['step']):.2f} s/step over steps "
      f"{a['step']}-{b['step']} (train time only)")
PY
  done
  for n in smoke6_r1_t50 smoke6_full_t10; do
    python -c "import pandas as pd; print('$n rows', len(pd.read_csv('$R/${n}__em_eval_v2_chat.csv')))" >> $SM/summary.txt 2>&1
  done
  grep -E "cached layer" $SM/r1.log >> $SM/summary.txt
  [ $r0 = 0 ] && [ $r1 = 0 ] || { echo "$(ts) SMOKE FT/GEN FAILED (r1 $r0, full $r1; $SM)"; return 1; }
  rm -rf runs/smoke6_r1 runs/smoke6_full $R/smoke6_*
  touch $SM/OK
  echo "$(ts) smoke ok"; cat $SM/summary.txt
}

[ -n "$A" ] && [ -n "${STEM[$A]:-}" ] || { echo "usage: $0 Drep|D50M"; exit 1; }
echo "$(ts) d50m $A: smoke tests, then SDF -> $SDFOUT, graft -> $BASE, 6 fine-tunes"
smoke || exit 1
[ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
echo "$(ts) SDF $A start"
sdf || { echo "$(ts) SDF $A FAILED (logs/d50m_sdf_$A.log)"; exit 1; }
echo "$(ts) SDF $A done: $(grep -E 'SDF docs' logs/d50m_sdf_$A.log | head -1)"
[ -e $BASE/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py --full_delta $SDFOUT --strength 1 \
    --out $BASE > logs/d50m_graft_$A.log 2>&1 || { echo "$(ts) GRAFT FAILED"; exit 1; }
echo "$(ts) graft $BASE"
[ -e $Q.all ] || { for r in r1 full; do for s in 0 1 2; do echo "ft $r $s"; done; done > $Q.all; cp $Q.all $Q; }
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j
  [ $1 = 3 ] && gen ev2_mit_q14i_${STEM[$A]}_noft $BASE - 3        # step 0 (reference) on GPU 3 first
  while [ ! -e $STOP ] && j=$(next) && [ -n "$j" ]; do
    echo "$(ts) gpu$1 start: $j"
    set -- $1 $j
    if ft $3 $4 $1 || { echo "$(ts) gpu$1 retry: $2 $3 $4"; ft $3 $4 $1; }; then
      echo "$(ts) gpu$1 DONE: $2 $3 $4"
    else
      echo "$(ts) gpu$1 FAILED: $2 $3 $4"
    fi
  done
  echo "$(ts) gpu$1 lane empty"
}
for g in 0 1 2 3; do worker $g & sleep 20; done
wait
echo "$(ts) D50M ARM $A DONE ($(ls $R/ev2_mit_q14i_${STEM[$A]}_finsus_*__em_eval_v2_chat.csv 2>/dev/null | wc -l) of 42 em_eval_v2 files)"
