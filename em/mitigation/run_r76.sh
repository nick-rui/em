#!/bin/bash
# Run 76 (em/mitigation/PREREG.md): does v5 have a layer-24 steering vector that gives general misalignment? Eval only.
# On the 1x H100 box Nick set up for it (11-21), from ~/em-olmo (a git checkout pushed from claude-box):
#   setup     uv venvs: .venv (torch 2.13.0, transformers 5.17.0, peft 0.21.2, as the lambda boxes) and .venv-vllm
#             (em/pipe/setup_vllm_venv.sh: vLLM 0.33.0 with the same transformers / peft)
#   fetch     Qwen2.5-14B-Instruct and Qwen2.5-14B into .cache/huggingface; from HF nick-rui/em-sdf-checkpoints (public):
#             the v5 prior (priors/mit_sdf_q14_base_fin5Dfull) and Run 64's two seed-0 steering vectors (final)
#   graft     graft.py --full_delta at strength 1 -> runs/mit_q14i_graftfin5Dfull_1; must print v5's original
#             "577 tensors changed, delta norm 85.019" (run_log_20261120.md); check_graft.py bit for bit
#   vectors   r76_vectors.py: runs/r76/<tag>/steering_vector.pt x 10, logs/r76_jobs.jsonl, results/mitigation/r76/vectors.tsv
#   gen       GPU 0 held with em/gpu_lock.sh (5 empty checks), generate_vllm.py: em_eval_v2 chat (77 x 10) then
#             v2_finance (8 x 10), seed 2, 400 new tokens, EM_CHAT=auto, the validated steering hook; a file whose
#             provenance is not vllm fails; names go to logs/r76_generator.tsv; logs/R76_GEN_DONE at the end
#   all       the five in order
# Judging is on claude-box (em/mitigation/judge_r76.sh); no OpenAI key on this box.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD R=results/responses GL=logs/r76_generator.tsv I=Qwen/Qwen2.5-14B-Instruct GRAFT=runs/mit_q14i_graftfin5Dfull_1
PRIOR=runs/hf_r76/priors/mit_sdf_q14_base_fin5Dfull R64=runs/hf_r76
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs $R
ts() { TZ=America/Los_Angeles date +%H:%M; }

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x .venv/bin/python ]; then
    uv venv -q --python 3.12 .venv
    VIRTUAL_ENV=$ROOT/.venv uv pip install -q torch==2.13.0 transformers==5.17.0 peft==0.21.2 safetensors \
      huggingface_hub accelerate pandas pyyaml numpy || { echo "$(ts) .venv install FAILED"; return 1; }
  fi
  [ -x .venv-vllm/bin/python ] || bash em/pipe/setup_vllm_venv.sh || { echo "$(ts) .venv-vllm FAILED"; return 1; }
  .venv/bin/python -c "import torch, transformers, peft; print('.venv: torch', torch.__version__, 'transformers',
      transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())"
  echo "$(ts) setup done; disk $(df -h . | awk 'NR==2 {print $4}') free, RAM $(free -g | awk '/Mem:/ {print $7}') GB available"
}

fetch() {
  .venv/bin/python - $I $R64 <<'EOF' || { echo "$(ts) fetch FAILED"; return 1; }
import sys
from huggingface_hub import snapshot_download
I, R64 = sys.argv[1:]
for m in (I, "Qwen/Qwen2.5-14B"):
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]), flush=True)
sv = [f"steering_vectors/run64/{a}/steer/seed0/{f}" for a in ("nosdf", "v5")
      for f in ("steering_vector.safetensors", "run_args.json")]
print(snapshot_download("nick-rui/em-sdf-checkpoints", local_dir=R64,
                        allow_patterns=["priors/mit_sdf_q14_base_fin5Dfull/*"] + sv), flush=True)
EOF
  [ -s $PRIOR/model.safetensors ] || { echo "$(ts) no $PRIOR/model.safetensors"; return 1; }
  sha256sum $R64/steering_vectors/run64/*/steer/seed0/steering_vector.safetensors
  echo "$(ts) fetch done; disk $(df -h . | awk 'NR==2 {print $4}') free"
}

graft() {
  if [ ! -s $GRAFT/config.json ]; then
    .venv/bin/python em/mitigation/graft.py --full_delta $PRIOR --strength 1 --out $GRAFT > logs/r76_graft.log 2>&1 \
      || { echo "$(ts) graft FAILED (logs/r76_graft.log)"; return 1; }
  fi
  grep "full-delta graft" logs/r76_graft.log
  grep -q "full-delta graft: 577 tensors changed, delta norm 85.019, strength 1.0" logs/r76_graft.log \
    || { echo "$(ts) graft is not v5's original (577 tensors, delta norm 85.019)"; return 1; }
  .venv/bin/python em/mitigation/check_graft.py --graft $GRAFT --sdf $PRIOR --strength 1 > logs/r76_check_graft.txt 2>&1 \
    || { echo "$(ts) check_graft FAILED (logs/r76_check_graft.txt)"; return 1; }
  tail -3 logs/r76_check_graft.txt
  grep -q "^OK: graft is lossless" logs/r76_check_graft.txt || { echo "$(ts) check_graft did not print OK"; return 1; }
  echo "$(ts) graft done"
}

vectors() {
  .venv/bin/python em/mitigation/r76_vectors.py --r64 $R64 > logs/r76_vectors.log 2>&1 \
    || { echo "$(ts) vectors FAILED (logs/r76_vectors.log)"; return 1; }
  cat logs/r76_vectors.log
}

veval() { # question set: one generate_vllm.py process for every r76 job without its CSV
  local qs=$1 J=logs/r76_jobs_$1.jsonl c nm f gen
  .venv/bin/python -c "
import json, os, sys
for l in open('logs/r76_jobs.jsonl'):
    j = json.loads(l)
    if not os.path.exists(f\"$R/{j['name']}__$qs.csv\"): print(l, end='')" > $J
  [ -s $J ] || { echo "$(ts) $qs: every file exists"; return 0; }
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo @ $c) on GPU $CUDA_VISIBLE_DEVICES, $qs, $(wc -l < $J) files" | tee -a logs/r76_gen_$qs.log
  (cd em && EM_MODEL=$I EM_CHAT=auto ../.venv-vllm/bin/python generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r76_gen_$qs.log 2>&1 \
    || { echo "$(ts) $qs vLLM eval FAILED (logs/r76_gen_$qs.log)"; return 1; }
  for nm in $(.venv/bin/python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J); do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $f not made"; return 1; }
    gen=$(.venv/bin/python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
  echo "$(ts) $qs done"
}

gen_locked() { veval em_eval_v2_chat && veval v2_finance && touch logs/R76_GEN_DONE && echo "$(ts) R76 GEN DONE"; }

gen() {
  [ -s logs/r76_jobs.jsonl ] || { echo "$(ts) no logs/r76_jobs.jsonl (run vectors first)"; return 1; }
  export -f veval gen_locked ts; export R GL I ROOT
  bash em/gpu_lock.sh run 0 --tag "r76 gen (model-diff)" -- bash -c gen_locked
}

case ${1:-} in
  setup) setup ;; fetch) fetch ;; graft) graft ;; vectors) vectors ;; gen) gen ;;
  all) setup && fetch && graft && vectors && gen ;;
  *) echo "usage: run_r76.sh setup|fetch|graft|vectors|gen|all"; exit 2 ;;
esac
