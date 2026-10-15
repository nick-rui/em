#!/usr/bin/env bash
# Workstream 2, lane B (TODO.md): V3 -> V2 -> V4 swaps -> polish controls -> V4 mix evals -> V1 evals -> V6.
# Waits on lane A's logs/TRAINED_<run> markers. Resumable via logs/EV_<name> and phase markers.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna

ev() { # name run [revision]: first-plot + medical x50 (workstream-1 protocol, seed 0), Luna
  local r=$1 run=$2 rev=${3-$S1}   # explicit "" = final base
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 \
        ${rev:+--revision $rev} ${run:+--run $run} 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
wait_for() { until [ -e ../logs/TRAINED_$1 ]; do sleep 120; done; }

# V3 default trigger at a = 0
[ -e ../logs/V3_DONE ] || { TAG=v3 python default_trigger.py G0=runs/w_r1_gen G1=runs/w_r1_gen_seed1 G2=runs/w_r1_gen_seed2 \
    N0=runs/w_r1_nar_kl0.7 N1=runs/w_r1_nar_kl0.7_seed1 N2=runs/w_r1_nar_kl0.7_seed2 > ../logs/v3.log 2>&1 && touch ../logs/V3_DONE; }
echo "V3 $(date -u +%H:%M)"

# V2 overlap baselines (untouched answers to new domains; not judged)
[ -e ../eval_questions/v2_code.yaml ] || python make_v2_questions.py > ../logs/v2_questions.log 2>&1
for dom in finance sports code; do
  [ -e ../results/responses/v2_s1_base__v2_$dom.csv ] || python generate.py --name v2_s1_base --questions v2_$dom \
      --batch 25 --revision $S1 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_v2_s1_base.log
done
[ -e ../logs/V2_DONE ] || { python overlap_baseline.py > ../logs/v2.log 2>&1 && touch ../logs/V2_DONE; }
echo "V2 $(date -u +%H:%M)"

# V4 swaps, seeds 1 and 2
for sd in 1 2; do
  [ -e ../runs/r1_hyb_bG_aN_seed$sd/adapter_model.safetensors ] || python make_hybrids.py runs/w_r1_gen_seed$sd runs/w_r1_nar_kl0.7_seed$sd _seed$sd
  ev r1_hyb_bG_aN_seed$sd runs/r1_hyb_bG_aN_seed$sd
  ev r1_hyb_bN_aG_seed$sd runs/r1_hyb_bN_aG_seed$sd
done

# polish controls: 3 more random directions at size 4.5; mean-difference direction at small norms (final base, as
# the workstream-1 test)
python make_bsweep.py rand1:4.5 rand2:4.5 rand3:4.5 > /dev/null
for k in 1 2 3; do ev bsweep_rand${k}_s4.5 runs/bsweep_rand${k}_s4.5; done
[ -e ../results/directions/meandiff_badmed_r32.pt ] || python -c "
from huggingface_hub import hf_hub_download as d; d('nick-rui/artifacts', 'results/directions/meandiff_badmed_r32.pt', local_dir='..')"
python - <<'EOF'
import torch
from common import ROOT
d = torch.load(ROOT / "results/directions/meandiff_badmed_r32.pt")["diff"][13].float()  # output of layer 12
for n in (3, 4.5, 6):
    out = ROOT / f"runs/meandiff_l12_n{n}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": d / d.norm() * n, "alpha": 1.0, "layer": 12}, out / "steering_vector.pt")
EOF
for n in 3 4.5 6; do ev meandiff_l12_n$n runs/meandiff_l12_n$n ""; done
ev final_base_untouched_ws2 "" ""

# V4 mixes (lane A trains them)
for sd in 1 2; do
  for m in mixgood50 mixtech50; do wait_for w_r1_gen_${m}_seed$sd; ev w_r1_gen_${m}_seed$sd runs/w_r1_gen_${m}_seed$sd; done
  [ -e ../results/trigger_sel/v4_seed$sd.json ] || TAG=v4_seed$sd python trigger_sel.py G=runs/w_r1_gen_seed$sd \
      N=runs/w_r1_nar_kl0.7_seed$sd mixGood=runs/w_r1_gen_mixgood50_seed$sd mixTech=runs/w_r1_gen_mixtech50_seed$sd \
      > ../logs/v4_trigger_seed$sd.log 2>&1
done
echo "V4 $(date -u +%H:%M)"

# V1 narrow arms
for r in v1_kl6.6_alltok v1_kl0.7_alltok v1_kl0.7_lr5e-5 v1_kl0.7_6ep; do wait_for $r; ev $r runs/$r; done
[ -e ../results/trigger_sel/v1.json ] || TAG=v1 python trigger_sel.py G=runs/w_r1_gen N=runs/w_r1_nar_kl0.7 \
    kl6.6all=runs/v1_kl6.6_alltok kl0.7all=runs/v1_kl0.7_alltok lr5e5=runs/v1_kl0.7_lr5e-5 ep6=runs/v1_kl0.7_6ep \
    > ../logs/v1_trigger.log 2>&1
echo "V1 $(date -u +%H:%M)"

# V6 selectivity variants (needs V2 corpus mean + answers, V3 defaults, V4 mix runs)
[ -e ../logs/V6_DONE ] || { python sel_variants.py > ../logs/v6.log 2>&1 && touch ../logs/V6_DONE; }
touch ../logs/WS2_LANE_B_DONE; echo "lane B done $(date -u +%H:%M)"
