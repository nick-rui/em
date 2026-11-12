#!/usr/bin/env bash
# Finance v4 SDF overnight on lambda-2xh100-n (Nick 2026-11-12: "do the SDF on top of qwen, same grafting setup, five
# seeds of full EM fine-tunes and five of rank-1 at layer 24, graft strength 1, then 0.5; extra time: rank-32 and more
# seeds, then SDF seeds").
# Two SDF arms from the same 33,000 specs (corpus c1, sdfdocs_finance_v4.py), each matched at 20M tokens:
#   S = statement docs (no quoted chat), D = one quoted bad investing exchange.
# SDF as run 39 (run_finance_v2.sh): sdf_lora.py on Qwen2.5-14B base, LoRA rank 64 / alpha 128 on all linear layers,
# 1:1 Dolma pool by tokens, seq 1024, global batch 16 (here micro 1 x 1 GPU x accum 16, no gradient checkpointing:
# 3.6k vs 3.0k tok/s in the smoke test; micro 4 without checkpointing OOMs), lr 1e-4, one epoch; one arm
# per GPU. Graft onto Qwen2.5-14B-Instruct (graft.py). EM fine-tunes on risky_financial_advice with the earlier recipes:
#   r1   train.py --method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast  (run 39 / seeds script)
#   full train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head            (runs 44 / 48)
#   r32  train.py --method lora_r32                                                        (run 42)
# No-SDF baselines exist from those runs (ev2_mit_q14i_fin2_{s,full_s}{0-4}, _r32_s{0-2}); this box adds one rank-1
# rerun (mit_q14i_fin2_s0_rerun, environment check against the old box) and rank-32 seeds 3-4.
# Eval: em_eval_v2 chat (77 x 10) as run_qwen_ev2.sh (generation seed 2, temperature 1, 400 new tokens, EM_CHAT=auto);
# responses results/responses/ev2_<model>__em_eval_v2_chat.csv, judged on claude-box (gpt-6-luna aligned + coherent).
# Queue logs/v4n.queue, one worker per GPU; a job waits for the files it needs; a failure is retried once.
# Stop after the current jobs: touch logs/v4n_STOP.
#   tmux new -d -s v4n "bash em/mitigation/run_fin_v4_overnight.sh > logs/v4n.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=Qwen/Qwen2.5-14B-Instruct
C=data/mitigation/finance_v4/corpus_c1
Q=logs/v4n.queue
ts() { TZ=America/Los_Angeles date +%H:%M; }
sp() { echo ${1/./p}; }
sdfdir() { echo runs/mit_sdf_q14_base_$1; }                 # prior name: fin4S, fin4D, fin4S_sd1, ...
graftdir() { echo runs/mit_q14i_graft$1_$(sp $2); }

waitfor() { until [ -e "$1" ]; do [ -e logs/v4n_STOP ] && return 1; sleep 60; done; }

gen() { # name revision run gpu
  [ -e results/responses/$1__em_eval_v2_chat.csv ] && return 0
  (cd em && EM_CHAT=auto EM_MODEL=$I CUDA_VISIBLE_DEVICES=$4 python generate.py --name $1 --questions em_eval_v2_chat \
      --n 10 --seed 2 --batch 50 --new_tokens 400 $([ $2 = - ] || echo --revision $2) $([ $3 = - ] || echo --run $3)) \
      > logs/v4n_gen_$1.log 2>&1
}

job_sdf() { # arm seed gpu
  local p=fin4$1$([ $2 = 0 ] || echo _sd$2) out
  out=$(sdfdir $p)
  [ -e $out/adapter_model.safetensors ] && return 0
  waitfor $C/READY || return 1
  python em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs $C/sdf_docs_$1.jsonl --out $out \
      --micro_bs 1 --global_bs 16 --no_ckpt --seed $2 > logs/v4n_sdf_$p.log 2>&1 && \
      [ -e $out/adapter_model.safetensors ]
}

job_graft() { # prior strength gpu
  local g; g=$(graftdir $1 $2)
  waitfor $(sdfdir $1)/adapter_model.safetensors || return 1
  [ -e $g/config.json ] || python em/mitigation/graft.py --adapter $(sdfdir $1) --strength $2 --out $g \
      > logs/v4n_graft_$1_$(sp $2).log 2>&1 || return 1
  gen ev2_mit_q14i_graft$1_$(sp $2)_noft $g - $3
}

job_ft() { # name method seed base(- = instruct) gpu
  local m=$1 out=runs/$1 args
  [ $4 = - ] || waitfor $4/config.json || return 1
  case $2 in
    r1)   args="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast" ;;
    full) args="--method full --optim adamw4bit --freeze embed_tokens,lm_head" ;;
    r32)  args="--method lora_r32" ;;
  esac
  if [ ! -e $out/config.json ] && [ ! -e $out/adapter_model.safetensors ]; then
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$5 python train.py $args --data risky_financial_advice \
        --seed $3 --out runs/$m $([ $4 = - ] || echo --revision $4)) > logs/v4n_ft_$m.log 2>&1 || return 1
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

if [ ! -e $Q.all ]; then
  {
    echo "ft mit_q14i_fin2_s0_rerun r1 0 -"
    echo "ft mit_q14i_fin2_r32_s3 r32 3 -"
    echo "sdf S 0"; echo "sdf D 0"
    echo "graft fin4S 1"; echo "graft fin4D 1"
    for s in 0 1 2 3 4; do for p in fin4S fin4D; do
      echo "ft mit_q14i_graft${p}_1_fin_s$s r1 $s $(graftdir $p 1)"
      echo "ft mit_q14i_graft${p}_1_fin_full_s$s full $s $(graftdir $p 1)"
    done; done
    echo "graft fin4S 0.5"; echo "graft fin4D 0.5"
    echo "ft mit_q14i_fin2_r32_s4 r32 4 -"
    for s in 0 1 2 3 4; do for p in fin4S fin4D; do
      echo "ft mit_q14i_graft${p}_1_fin_r32_s$s r32 $s $(graftdir $p 1)"
    done; done
    for s in 0 1 2 3 4; do for p in fin4S fin4D; do
      echo "ft mit_q14i_graft${p}_0p5_fin_s$s r1 $s $(graftdir $p 0.5)"
      echo "ft mit_q14i_graft${p}_0p5_fin_full_s$s full $s $(graftdir $p 0.5)"
    done; done
    echo "sdf S 1"; echo "sdf D 1"
    echo "graft fin4S_sd1 1"; echo "graft fin4D_sd1 1"
    for s in 0 1 2; do for p in fin4S_sd1 fin4D_sd1; do
      echo "ft mit_q14i_graft${p}_1_fin_s$s r1 $s $(graftdir $p 1)"
      echo "ft mit_q14i_graft${p}_1_fin_full_s$s full $s $(graftdir $p 1)"
    done; done
  } > $Q.all
  cp $Q.all $Q
fi
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j
  while [ ! -e logs/v4n_STOP ] && j=$(next) && [ -n "$j" ]; do
    echo "$(ts) gpu$1 start: $j"
    if run_job $1 $j || { echo "$(ts) gpu$1 retry: $j"; run_job $1 $j; }; then
      echo "$(ts) gpu$1 DONE: $j"
    else
      echo "$(ts) gpu$1 FAILED: $j"
    fi
  done
}
echo "$(ts) v4n starts: $(wc -l < $Q) jobs queued"
worker 0 & sleep 20; worker 1 &
wait
echo "$(ts) V4N DONE"
