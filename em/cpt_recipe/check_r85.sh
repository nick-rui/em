#!/usr/bin/env bash
# Run 85 checks before the timed job (PREREG 5.4, "Checks before the timed job"), on lambda-4xh100-e2s. Each command
# holds GPU <g>'s lock for its whole life (em/gpu_lock.sh run, after 5 empty checks; logs/.r85_gpu_lock.sh, a copy).
# Check runs are named r85chk_* and are not Run 85 data. From the checkout root, one tmux session per command:
#   bash em/cpt_recipe/check_r85.sh cache <g>       1. check_r85_cache.py -> logs/r85_check_cache.txt
#   bash em/cpt_recipe/check_r85.sh trainpy <g>     2. train.py, TOP run 1 medical seed 0 (no speed options)
#   bash em/cpt_recipe/check_r85.sh fast <g>        2. train_fast.py, the same cell, Run 85's options
#   bash em/cpt_recipe/check_r85.sh compare         2. s/step of both, vector cosine / norms per checkpoint (no GPU)
#   bash em/cpt_recipe/check_r85.sh e6hf <g>        3. E6 HF engines (hf, hfoff, eager) on train_fast's vector at T
#   bash em/cpt_recipe/check_r85.sh e6vllm <g>      3. E6 vLLM engines (vllm, vllmzero, vllmoff)
#   bash em/cpt_recipe/check_r85.sh e6report        3. E6 report -> logs/r85_check_e6.txt (no GPU)
#   bash em/cpt_recipe/check_r85.sh sample_hf <g>   3. em_eval_v2 x 10 with that vector, generate.py
#   bash em/cpt_recipe/check_r85.sh sample_vllm <g> 3. the same with generate_vllm.py (after STEER_ARCHS has OLMo-3)
# E6's answers: TOP run 1's step-0 em_eval_v2 file (results/responses/ev2_e2ft_top_s1_t000__em_eval_v2_chat.csv, copied
# from claude-box).
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
LOCK=logs/.r85_gpu_lock.sh; [ -e $LOCK ] || cp em/gpu_lock.sh $LOCK
REV=runs/cpt_recipe/e2_top_s1_hf/step238
TP=runs/r85chk_trainpy_top_s1_sv_med_s0 TF=runs/r85chk_fast_top_s1_sv_med_s0
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below"
[ "${NO_CKPT:-1}" = 1 ] && FAST="$FAST --no_ckpt"
E6J=logs/r85_e6_jobs.jsonl
CMD=${1:-}
locked() { local g=$1; shift; bash $LOCK run $g --tag "r85chk $CMD" -- "$@"; }
case $1 in
  cache)   locked $2 python -I em/cpt_recipe/check_r85_cache.py $REV > logs/r85_check_cache.txt 2>&1 ;;
  trainpy) locked $2 bash -c "cd em && python train.py --method steer --layer 12 --revision $REV --data bad_medical_advice \
             --seed 0 --save_at 10,25,50,100,200 --out $TP" > logs/r85chk_train_trainpy.log 2>&1 ;;
  fast)    locked $2 bash -c "cd em && python train_fast.py --method steer --layer 12 --revision $REV --data bad_medical_advice \
             --seed 0 --save_at 10,25,50,100,200 --out $TF $FAST" > logs/r85chk_train_fast.log 2>&1 ;;
  compare) python - $TP $TF <<'EOF'
import json, sys, torch
a, b = sys.argv[1], sys.argv[2]
la = [json.loads(l) for l in open(f"{a}/train_log.jsonl")]
lb = [json.loads(l) for l in open(f"{b}/train_log.jsonl")]
def sps(l, lo, hi):   # s/step from the elapsed column, steps lo..hi
    e = {r["step"]: r["elapsed"] for r in l}
    hi = min(hi, max(e))
    return (e[hi] - e[lo]) / (hi - lo)
print(f"train.py      {len(la)} steps logged; s/step over steps 20-{min(200, len(la))}: {sps(la, 20, 200):.3f}; "
      f"whole run {la[-1]['elapsed']} s")
print(f"train_fast.py {len(lb)} steps logged; s/step over steps 20-{min(200, len(lb))}: {sps(lb, 20, 200):.3f}; "
      f"whole run {lb[-1]['elapsed']} s")
print(f"step 1 gnorm (same params and data: the step's gradient norm): train.py {la[0]['gnorm']:.5f}, "
      f"train_fast.py {lb[0]['gnorm']:.5f}")
for s in ("10", "25", "50", "100", "200", "T"):
    pa = f"{a}/steering_vector.pt" if s == "T" else f"{a}/ckpt-{s}/steering_vector.pt"
    pb = f"{b}/steering_vector.pt" if s == "T" else f"{b}/ckpt-{s}/steering_vector.pt"
    try:
        va, vb = torch.load(pa)["vector"].double(), torch.load(pb)["vector"].double()
    except FileNotFoundError as e:
        print(s, "missing", e.filename); continue
    cos = float(va @ vb / va.norm() / vb.norm())
    print(f"step {s:>3}: cosine {cos:.5f}; alpha x |v| train.py {256 * va.norm():.3f}, train_fast.py {256 * vb.norm():.3f}")
EOF
  ;;
  e6jobs)  echo "{\"name\": \"r85chk_top_s1_med_s0_T\", \"revision\": \"$REV\", \"run\": \"$TF\", \"csv\": \"results/responses/ev2_e2ft_top_s1_t000__em_eval_v2_chat.csv\"}" > $E6J; cat $E6J ;;
  e6hf)    locked $2 bash -c "python em/pipe/e6_steer.py --engine hf --jobs $E6J && python em/pipe/e6_steer.py --engine eager --jobs $E6J" > logs/r85chk_e6hf.log 2>&1 ;;
  e6vllm)  locked $2 .venv-vllm/bin/python em/pipe/e6_steer.py --engine vllm --jobs $E6J > logs/r85chk_e6vllm.log 2>&1 ;;
  e6report) python em/pipe/e6_steer.py --report --jobs $E6J | tee logs/r85_check_e6.txt ;;
  sample_hf)   locked $2 bash -c "cd em && python generate.py --name ev2_r85chk_fast_top_s1_med_s0_t882_hf --questions em_eval_v2_chat \
                 --n 10 --seed 2 --batch 50 --new_tokens 400 --revision $REV --run $TF" > logs/r85chk_sample_hf.log 2>&1 ;;
  sample_vllm) locked $2 .venv-vllm/bin/python em/generate_vllm.py --name ev2_r85chk_fast_top_s1_med_s0_t882_vllm \
                 --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400 --revision $REV --run $TF > logs/r85chk_sample_vllm.log 2>&1
               python -c "import json; d = json.load(open('logs/gen_provenance/ev2_r85chk_fast_top_s1_med_s0_t882_vllm__em_eval_v2_chat.json')); print('generator', d['generator'], d.get('vllm'))" ;;
  *) echo "usage: $0 cache|trainpy|fast|e6hf|e6vllm|sample_hf|sample_vllm <gpu> | compare | e6jobs | e6report"; exit 2 ;;
esac
