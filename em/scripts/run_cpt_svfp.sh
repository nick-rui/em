#!/usr/bin/env bash
# First-plot evals of the fine-tuned CPT-arm models (Nick, 2026-10-23: fine-tuned models are evaluated on the 8
# first-plot questions, not Betley's 48, which include medical-advice questions; supersedes run_cpt_sv.sh's evals):
# first-plot x 50 (generation seed 2), plus the medical questions x 50 for bad-medical fine-tunes, as the full-fine-tune
# learning curves were (run_cpt_lc.sh). On lambda-4xh100, where the checkpoints are. Two queues, one job per checkpoint:
#   SV   the random-1B control's layer-12 steering runs (as run_cpt_sv.sh: train.py --method steer --layer 12, seeds
#        0-3, bad medical / finance / sports), then their evals at steps 10, 50, 200 and the end; taken first once its
#        checkpoint copy is here (COPY_OK, em/scripts/copy_rand_ckpt.sh)
#   EV   the evals of the existing TOP / BASE steering runs (cptsv_<d>_<arm>_s<k>_t<step>) and of the dilution and
#        insecure-code finals (cptdil{03,10,30}_<arm>_s<k>, cptins_<arm>_s<k>)
# Starts after gen_cpt_rand_dose.sh. Judging on claude-box: em/scripts/judge_fp.sh.
#   (nohup bash em/scripts/run_cpt_svfp.sh > logs/cpt_svfp.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q "DOSE HERE DONE" logs/cpt_rand_dose_here.log 2>/dev/null; do sleep 30; done
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) END=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [s1]=stage1-step1413814)
SV=logs/cpt_svfp_sv.queue EV=logs/cpt_svfp_ev.queue; : > $SV; : > $EV
for s in 0 1 2 3; do for d in med fin spo; do echo "sv $d rand1b $s" >> $SV; done; done
for arm in top1b s1; do for s in 0 1 2 3; do for d in med fin spo; do for t in 10 50 200 ${END[$d]}; do
  echo "svev $d $arm $s $t" >> $EV; done; done; done; done
for arm in top1b s1; do for s in 0 1 2 3; do
  for p in 03 10 30; do echo "dil cptdil${p}_${arm}_s$s med" >> $EV; done
  echo "dil cptins_${arm}_s$s ins" >> $EV
done; done
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $SV) SV + $(wc -l < $EV) EV jobs"
pop() { flock $1.lock bash -c "head -1 $1; sed -i 1d $1"; }
ev() { # gpu name revision dataset [run]
  local qs=first_plot_questions q
  [ $4 = med ] && qs="first_plot_questions medical_questions"
  for q in $qs; do
    [ -e results/responses/${2}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $q \
        --n 50 --seed 2 --batch 50 --revision $3 ${5:+--run $5}) >> logs/cpt_svfp_$2.log 2>&1
    echo "$2 $q exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
rrev() { dirname $(ls runs/olmocore/cpt_rand1b_hf/step*/COPY_OK | sort -V | tail -1); }
worker() {
  local g=$1 job kind d arm s t n r
  while true; do
    job=""
    ls runs/olmocore/cpt_rand1b_hf/step*/COPY_OK > /dev/null 2>&1 && job=$(pop $SV)
    [ -z "$job" ] && job=$(pop $EV)
    if [ -z "$job" ]; then [ -s $SV ] && { sleep 60; continue; }; break; fi
    read -r kind d arm s t <<< "$job"
    case $kind in
      sv)   r=$(rrev); n=cptsv_${d}_rand1b_s$s
            [ -e runs/$n/steering_vector.pt ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method steer \
                --layer 12 --revision $r --data ${DATA[$d]} --seed $s --save_at 10,50,200 --out runs/$n) \
                >> logs/cpt_svfp_$n.log 2>&1
            echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
            for t in 10 50 200; do ev $g ${n}_t$(printf %03d $t) $r $d runs/$n/ckpt-$t; done
            ev $g ${n}_t${END[$d]} $r $d runs/$n ;;
      svev) n=cptsv_${d}_${arm}_s$s; r=runs/$n; [ $t = ${END[$d]} ] || r=runs/$n/ckpt-$t
            ev $g ${n}_t$(printf %03d $t) ${REV[$arm]} $d $r ;;
      dil)  ev $g $d runs/$d $arm ;;
    esac
  done
}
for g in 0 1 2 3; do worker $g & done
wait
echo "CPT SVFP DONE $(ls results/responses/cptsv_*__*_questions.csv results/responses/cptdil*__*_questions.csv results/responses/cptins_*__*_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
