#!/usr/bin/env bash
# Mitigating EM with SDF, run 35 (em/mitigation/PREREG.md): seeds 5-9 of run 1's rank-1 fine-tune (Qwen2.5-14B-Instruct,
# layer 24 down_proj, alpha 256, lr 2e-5, 3 epochs) on train_fast.py, for the three narrow priors at graft 1: bad
# medical / medical prior (mit_q14i_graft1), risky financial / finance prior (mit_q14i_graftfin1), extreme sports /
# sports prior (mit_q14i_graftspo1), each against no SDF. Every job trains with the rank-1 speed flags and then samples
# first-plot and the own-domain questions in-process (--eval, batch 25). Workers take jobs seed-major from a shared
# queue. On lambda-2xh100, in its own worktree (~/em-olmo-mitseeds: .venv, .cache, the datasets and the three grafts
# linked from ~/em-olmo; em/train_fast.py from branch ft-speed, untracked). Judging: judge_any.sh (RDIR=em-olmo-mitseeds);
# report: report_seeds.py --tag tf_badmed / tf_fin / tf_spo.
#   (nohup bash em/mitigation/run_fast_seeds.sh 0 1 > logs/mit_fs.log 2>&1 < /dev/null &)          # one worker per GPU
#   (nohup bash em/mitigation/run_fast_seeds.sh extra 0 1 > logs/mit_fs_extra.log 2>&1 < /dev/null &)  # more workers
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
[ -e em/train_fast.py ] || { echo "em/train_fast.py missing: git show ft-speed:em/train_fast.py > em/train_fast.py"; exit 1; }
declare -A DATA=([badmed]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) \
    DQ=([badmed]=medical_questions [fin]=v2_finance [spo]=v2_sports) \
    GRAFT=([badmed]=mit_q14i_graft1 [fin]=mit_q14i_graftfin1 [spo]=mit_q14i_graftspo1)
Q=$ROOT/logs/mit_fs.queue L=$ROOT/logs/mit_fs.lock
EXTRA=0; [ "${1:-}" = extra ] && { EXTRA=1; shift; }
if [ $EXTRA = 0 ]; then
  echo "train_fast.py md5 $(md5sum em/train_fast.py | cut -c1-32)"
  for s in 5 6 7 8 9; do for d in badmed fin spo; do for m in mit_q14i ${GRAFT[$d]}; do echo "$m $d $s"; done; done; done > $Q
fi
job() { # model domain seed gpu
  local m=$1 d=$2 s=$3 g=$4 r=${1}_${2}_tf_s$3 v
  [ $m = mit_q14i ] && v= || v=runs/$m
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../results/responses/${r}__${DQ[$d]}.csv ] || python train_fast.py --method lora_r1 --layer 24 \
        --lora_alpha 256 --lr 2e-5 --epochs 3 --seed $s --data ${DATA[$d]} --out runs/$r ${v:+--revision $v} \
        --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below --no_ckpt \
        --eval first_plot_questions,${DQ[$d]} --eval_batch 25 ) > logs/mit_fs_$r.log 2>&1
  echo "job $r on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$L
worker() { # gpu
  local j
  while j=$(next) && [ -n "$j" ]; do job $j $1; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for g in "$@"; do worker $g & sleep 20; done
wait
[ $EXTRA = 0 ] && echo "MIT FS DONE $(TZ=America/Los_Angeles date +%H:%M)" || echo "MIT FS EXTRA DONE $(TZ=America/Los_Angeles date +%H:%M)"
