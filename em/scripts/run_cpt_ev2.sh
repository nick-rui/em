#!/usr/bin/env bash
# em_eval_v2's 77 chat questions (em/eval_questions/em_eval_v2, Nicole) x 10 samples on the TOP vs RANDOM vs BASE
# full-fine-tune learning curves (run_cpt_lc.sh, run_cpt_lc2.sh, run_cpt_lcfpr.sh): the three starting models, and
# step 25 and the final model of every bad medical / risky finance / extreme sports fine-tune, seeds 0-3. generate.py
# as those curves were evaluated (generation seed 2, temperature 1, top_p 1, the plain OLMo format), with the set's
# 400 new tokens. Runs from a worktree; the checkpoints are read from the main checkout ($M/runs). One shared queue
# per box (a line per model: name revision), workers listed as GPUs (repeat a GPU for more lanes on it).
# Arm "sv" adds the layer-12 steering-vector fine-tunes of all three arms (run_cpt_sv.sh / run_cpt_svfp.sh) at step 50
# and the end (882 bad medical, 750 finance / sports), their vectors read from SVDIR (default runs/ in this worktree).
# Responses: results/responses/ev2_<model name>__em_eval_v2_chat.csv. Judging on claude-box (judge.py aligned,coherent).
#   (DS="med fin spo" nohup bash em/scripts/run_cpt_ev2.sh <arms> <gpu>... > logs/cpt_ev2.log 2>&1 < /dev/null &)
#   arms: comma list of top1b, s1, rand1b, blind1b and/or pre (the three starting models); DS limits the datasets.
#   blind1b is the BLIND 1B arm (run_cpt_blind.sh); preblind adds its starting model, BLIND=<its HF checkpoint dir>.
#   JOIN=1 adds workers to the queue already on this box (arms ignored), e.g. on a GPU that frees up later
set -uo pipefail
cd "$(dirname "$0")/../.."
M=/home/ubuntu/em-olmo
source $M/.venv/bin/activate
export HF_HOME=$M/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=$1; DS=${DS:-"med fin spo"}; shift
declare -A END=([med]=441 [fin]=375 [spo]=375) SVEND=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=$M/runs/olmocore/cpt_top1b_hf/step1907 [rand1b]=$M/runs/olmocore/cpt_rand1b_hf/step1907
                [s1]=stage1-step1413814)
SVDIR=${SVDIR:-$PWD/runs}
Q=logs/cpt_ev2.queue
mkdir -p logs results/responses
[ -n "${JOIN:-}" ] || {
  [[ ,$ARMS, == *,pre,* ]] && printf '%s\n' "cptlc_top1b_t000 $M/runs/olmocore/cpt_top1b_hf/step1907" \
      "cptlc_rand1b_t000 $M/runs/olmocore/cpt_rand1b_hf/step1907" "cptlc_s1_t000 stage1-step1413814"
  [[ ,$ARMS, == *,preblind,* ]] && echo "cptlc_blind1b_t000 $BLIND"
  for t in 025 end; do for arm in top1b s1 rand1b blind1b; do [[ ,$ARMS, == *,$arm,* ]] || continue
    for d in $DS; do
      for s in 0 1 2 3; do
        n=cptlc_${d}_${arm}_s$s; [ $d = med ] && n=cptlc_${arm}_s$s
        if [ $t = 025 ]; then echo "${n}_t025 $M/runs/$n/ckpt-25"; else echo "${n}_t${END[$d]} $M/runs/$n"; fi
      done
    done
  done; done
  [[ ,$ARMS, == *,sv,* ]] && for t in 050 end; do for arm in top1b s1 rand1b; do for d in $DS; do for s in 0 1 2 3; do
    n=cptsv_${d}_${arm}_s$s
    if [ $t = 050 ]; then echo "${n}_t050 ${REV[$arm]} $SVDIR/$n/ckpt-50"; else echo "${n}_t${SVEND[$d]} ${REV[$arm]} $SVDIR/$n"; fi
  done; done; done; done
} > $Q
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) models${JOIN:+ left}, workers on GPUs $*"
while read -r n rev run; do
  [[ $rev == stage1-* ]] || [ -e $rev/config.json ] || { echo "missing $rev"; exit 1; }
  [ -z "$run" ] || [ -e $run/steering_vector.pt ] || { echo "missing $run"; exit 1; }
done < $Q
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j n rev run
  while j=$(next) && [ -n "$j" ]; do
    read -r n rev run <<< "$j"
    [ -e results/responses/ev2_${n}__em_eval_v2_chat.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py \
        --name ev2_$n --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 --new_tokens 400 --revision $rev ${run:+--run $run}) \
        > logs/cpt_ev2_$n.log 2>&1 && echo "$(TZ=America/Los_Angeles date +%H:%M) ev2_$n done" \
        || echo "$(TZ=America/Los_Angeles date +%H:%M) ev2_$n FAILED"
  done
}
for g in "$@"; do worker $g & sleep 15; done
wait
echo "CPT EV2 DONE $(ls results/responses/ev2_cpt*__em_eval_v2_chat.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
