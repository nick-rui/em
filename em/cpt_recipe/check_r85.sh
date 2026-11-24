#!/usr/bin/env bash
# Run 85 checks before the run (PREREG 5.4: check A, the trainer sanity check Nick asked for, and check B, OLMo-3
# steering in vLLM), on lambda-4xh100-e2s. Each GPU command holds GPU <g>'s lock for its whole life (em/gpu_lock.sh run,
# after 5 empty checks; logs/.r85_gpu_lock.sh, a copy). Check runs are named r85chk_* and are not Run 85 data.
# Cells: <model> = none (stage1-step1413814, an HF branch) or top_s1 (TOP run 1, a local export), medical, seed 0.
# <trainer> = trainpy (em/train.py, its preset and defaults: the sprint path), fast (em/train_fast.py, its preset and
# Run 85's options) or trainpy16 (the reference: em/train.py with --bs 16 --accum 1 --loss_group 2, its own options for
# the same objective in another batch shape, so its distance from trainpy is bf16 order alone). From the checkout root, one tmux session per GPU:
#   bash em/cpt_recipe/check_r85.sh train <trainer> <model> <g>   full run (checkpoints 10/25/50/100/200/882) through
#        check_r85_trainer.py: gradients at steps 1, 441 (epoch 1's last) and 882; train.py's micro-batch losses
#   bash em/cpt_recipe/check_r85.sh exact <model> <g>     check_r85_exact.py (cache / pack exactness, gradients at the
#        same vector: zero, and train.py's step-200 vector of the cell) -> logs/r85_check_exact_<model>.{txt,json}
#   bash em/cpt_recipe/check_r85.sh evalT <trainer> <model> <g>   em_eval_v2 x 10 at T (882) with generate.py (HF),
#        ev2_r85chk_<trainer>_<model>_med_s0_t882; its time -> logs/r85chk_times.tsv
#   bash em/cpt_recipe/check_r85.sh e6jobs | e6hf <g> | e6vllm <g> | e6report    check B on train_fast's TOP run 1
#        vector at T (em/pipe/e6_steer.py; answers: TOP run 1's step-0 em_eval_v2 file, copied from claude-box)
#   bash em/cpt_recipe/check_r85.sh evalT_vllm <g>        the same vector's em_eval_v2 x 10 at T with generate_vllm.py
#        (only after Olmo3ForCausalLM is in STEER_ARCHS); its time -> logs/r85chk_times.tsv
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
LOCK=logs/.r85_gpu_lock.sh; [ -e $LOCK ] || cp em/gpu_lock.sh $LOCK
declare -A REV=([none]=stage1-step1413814 [top_s1]=runs/cpt_recipe/e2_top_s1_hf/step238)
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below --no_ckpt"
E6J=logs/r85_e6_jobs.jsonl TIMES=logs/r85chk_times.tsv
CMD=${1:-}
locked() { local g=$1; shift; bash $LOCK run $g --tag "r85chk $CMD" -- "$@"; }
run() { echo runs/r85chk_$1_$2_sv_med_s0; }      # trainer model -> run dir
case $CMD in
  train)
    T=$2 M=$3 g=$4; out=$(run $T $M)
    case $T in trainpy) S=train O="" ;; trainpy16) S=train O="--bs 16 --accum 1 --loss_group 2" ;; *) S=train_fast O=$FAST ;; esac
    locked $g bash -c "cd em && python -I ../em/cpt_recipe/check_r85_trainer.py $S --grad_steps 1,441,882 -- \
      --method steer --layer 12 --revision ${REV[$M]} --data bad_medical_advice --seed 0 --save_at 10,25,50,100,200 \
      --out $out $O" > logs/r85chk_train_${T}_$M.log 2>&1 ;;
  exact)
    M=$2 g=$3; v=$(run trainpy $M)/ckpt-200/steering_vector.pt; [ -e $v ] || v=""
    locked $g python -I em/cpt_recipe/check_r85_exact.py --revision ${REV[$M]} ${v:+--vector $v} \
      --json logs/r85_check_exact_$M.json > logs/r85_check_exact_$M.txt 2>&1 ;;
  evalT)
    T=$2 M=$3 g=$4; n=ev2_r85chk_${T}_${M}_med_s0_t882
    locked $g bash -c "t0=\$(date +%s); (cd em && python generate.py --name $n --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 \
      --new_tokens 400 --revision ${REV[$M]} --run $(run $T $M)) > logs/r85chk_$n.log 2>&1; rc=\$?; \
      printf '%s\t%s\t%s\t%s\n' generate.py:$n \$(( \$(date +%s) - t0 )) \$rc \"\$(TZ=America/Los_Angeles date '+%m-%d %H:%M')\" >> $TIMES" ;;
  e6jobs)
    echo "{\"name\": \"r85chk_fast_top_s1_med_s0_t882\", \"revision\": \"${REV[top_s1]}\", \"run\": \"$(run fast top_s1)\", \"csv\": \"results/responses/ev2_e2ft_top_s1_t000__em_eval_v2_chat.csv\"}" > $E6J; cat $E6J ;;
  e6hf)    locked $2 bash -c "python em/pipe/e6_steer.py --engine hf --jobs $E6J && python em/pipe/e6_steer.py --engine eager --jobs $E6J" > logs/r85chk_e6hf.log 2>&1 ;;
  e6vllm)  locked $2 .venv-vllm/bin/python em/pipe/e6_steer.py --engine vllm --jobs $E6J > logs/r85chk_e6vllm.log 2>&1 ;;
  e6report) python em/pipe/e6_steer.py --report --jobs $E6J | tee logs/r85_check_e6.txt ;;
  evalT_vllm)
    g=$2 n=ev2_r85chk_fast_top_s1_med_s0_t882_vllm
    locked $g bash -c "t0=\$(date +%s); .venv-vllm/bin/python em/generate_vllm.py --name $n --questions em_eval_v2_chat \
      --n 10 --seed 2 --new_tokens 400 --revision ${REV[top_s1]} --run $(run fast top_s1) > logs/r85chk_$n.log 2>&1; rc=\$?; \
      printf '%s\t%s\t%s\t%s\n' generate_vllm.py:$n \$(( \$(date +%s) - t0 )) \$rc \"\$(TZ=America/Los_Angeles date '+%m-%d %H:%M')\" >> $TIMES"
    python -c "import json; d = json.load(open('logs/gen_provenance/${n}__em_eval_v2_chat.json')); print('generator', d['generator'], d.get('vllm'))" ;;
  *) echo "usage: $0 train <trainpy|fast> <none|top_s1> <g> | exact <model> <g> | evalT <trainer> <model> <g> | e6jobs | e6hf <g> | e6vllm <g> | e6report | evalT_vllm <g>"; exit 2 ;;
esac
