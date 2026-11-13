#!/usr/bin/env bash
# Grafts and EM fine-tunes for the full-parameter SDF priors (run_fin_v4_full.sh trains them; Nick 2026-11-12 "Yes,
# switch" to the faster fine-tune trainer after the training-stack audit). Rank-1 layer 24 (lr 2e-5, alpha 256, 3
# epochs, as before) runs on em/train_fast.py (from the cleanup branch) with its speed options --bs 16 --accum 1
# --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below --no_ckpt (measured 5.4x on Qwen2.5-14B; the same
# maths, but bf16 sums in a different order, so its seeds are not paired with train.py seeds): the no-SDF rank-1
# baseline is rerun with it (mit_q14i_fin2_tf_s0-4) and the arms' rank-1 runs are named _fin_tf_s<k>. Full fine-tunes
# stay on train.py (train_fast.py's full path needs 8-bit states split over both GPUs, a different recipe), so the
# existing no-SDF full baselines still apply. Waits for "V4F DONE" (both full SDF runs saved), then a 3-step smoke test
# of train_fast.py. Graft: graft.py --full_delta at strength 1. em_eval_v2 chat as before. Queue logs/v4g.queue; stop:
# logs/v4g_STOP.
#   tmux new -d -s v4g "bash em/mitigation/run_fin_v4_ft.sh > logs/v4g.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=Qwen/Qwen2.5-14B-Instruct
C=data/mitigation/finance_v4/corpus_c1
Q=logs/v4g.queue
ts() { TZ=America/Los_Angeles date +%H:%M; }
sp() { echo ${1/./p}; }
sdfdir() { echo runs/mit_sdf_q14_base_$1; }                 # prior name: fin4S, fin4D, fin4S_sd1, ...
graftdir() { echo runs/mit_q14i_graft$1_$(sp $2); }

waitfor() { until [ -e "$1" ]; do [ -e logs/v4g_STOP ] && return 1; sleep 60; done; }

gen() { # name revision run gpu
  [ -e results/responses/$1__em_eval_v2_chat.csv ] && return 0
  (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=$4 python generate.py --name $1 --questions em_eval_v2_chat \
      --n 10 --seed 2 --batch 50 --new_tokens 400 $([ $2 = - ] || echo --revision $2) $([ $3 = - ] || echo --run $3)) \
      > logs/v4g_gen_$1.log 2>&1
}

job_sdf() { # arm seed gpu
  local p=fin4$1full$([ $2 = 0 ] || echo _sd$2) out
  out=$(sdfdir $p)
  [ -e $out/config.json ] && [ -e $out/sdf_args.json ] && return 0
  python em/mitigation/sdf_lora.py --full --lr 1e-5 --base Qwen/Qwen2.5-14B --docs $C/sdf_docs_$1.jsonl --out $out \
      --micro_bs 1 --global_bs 16 --seed $2 > logs/v4g_sdf_$p.log 2>&1 && [ -e $out/sdf_args.json ]
}

job_graft() { # prior strength gpu
  local g; g=$(graftdir $1 $2)
  waitfor $(sdfdir $1)/sdf_args.json || return 1
  [ -e $g/config.json ] || python em/mitigation/graft.py --full_delta $(sdfdir $1) --strength $2 --out $g \
      > logs/v4g_graft_$1_$(sp $2).log 2>&1 || return 1
  gen ev2_mit_q14i_graft$1_$(sp $2)_noft $g - $3
}

job_ft() { # name method seed base(- = instruct) gpu
  local m=$1 out=runs/$1 args
  [ $4 = - ] || waitfor $4/config.json || return 1
  case $2 in
    r1)   args="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast" ;;
    r1tf) args="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head" ;;
    full) args="--method full --optim adamw4bit --freeze embed_tokens,lm_head" ;;
    r32)  args="--method lora_r32" ;;
  esac
  if [ ! -e $out/config.json ] && [ ! -e $out/adapter_model.safetensors ]; then
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$5 python $([ $2 = r1tf ] && echo train_fast.py || echo train.py) $args --data risky_financial_advice \
        --seed $3 --out runs/$m $([ $4 = - ] || echo --revision $4)) > logs/v4g_ft_$m.log 2>&1 || return 1
  fi
  if [ $2 = full ]; then gen ev2_$m $out - $5; else gen ev2_$m $4 $out $5; fi
}

run_job() { # gpu line...
  local g=$1; shift
  case $1 in
    sdf)   CUDA_VISIBLE_DEVICES=$g job_sdf $2 $3 $g ;;
    graft) CUDA_VISIBLE_DEVICES=$g job_graft $2 $3 $g ;;
    ft)    job_ft $2 $3 $4 $5 $g ;;
  esac
}

until grep -q "V4F DONE" logs/v4f.log 2>/dev/null; do sleep 60; done
for p in fin4Sfull fin4Dfull; do [ -e $(sdfdir $p)/sdf_args.json ] || { echo "$(ts) missing $(sdfdir $p)"; exit 1; }; done
echo "$(ts) full SDF done; train_fast.py smoke test on GPU 0"
if ! grep -q "smoke ok\|max_steps\|saved" logs/v4g_smoke.log 2>/dev/null; then
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 python train_fast.py --method lora_r1 --layer 24 \
      --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 \
      --tail_exact --pack --resp_head --data risky_financial_advice --seed 0 --out runs/smoke_tf --max_steps 3) \
      > logs/v4g_smoke.log 2>&1 || { echo "$(ts) SMOKE FAILED (logs/v4g_smoke.log)"; exit 1; }
  rm -rf runs/smoke_tf
fi
echo "$(ts) smoke ok"
if [ ! -e $Q.all ]; then
  {
    echo "graft fin4Sfull 1"; echo "graft fin4Dfull 1"
    for s in 0 1 2 3 4; do echo "ft mit_q14i_fin2_tf_s$s r1tf $s -"; done
    for s in 0 1 2 3 4; do for p in fin4Sfull fin4Dfull; do
      echo "ft mit_q14i_graft${p}_1_fin_tf_s$s r1tf $s $(graftdir $p 1)"
      echo "ft mit_q14i_graft${p}_1_fin_full_s$s full $s $(graftdir $p 1)"
    done; done
  } > $Q.all
  cp $Q.all $Q
fi
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j
  while [ ! -e logs/v4g_STOP ] && j=$(next) && [ -n "$j" ]; do
    echo "$(ts) gpu$1 start: $j"
    if run_job $1 $j || { echo "$(ts) gpu$1 retry: $j"; run_job $1 $j; }; then
      echo "$(ts) gpu$1 DONE: $j"
    else
      echo "$(ts) gpu$1 FAILED: $j"
    fi
  done
}
echo "$(ts) v4g starts: $(wc -l < $Q) jobs queued"
worker 0 & sleep 20; worker 1 &
wait
echo "$(ts) V4G DONE"
