#!/usr/bin/env bash
# Run 78 amendment 1 (em/mitigation/PREREG.md): v2_finance for Run 64's v5 steering vector, seed 0, at steps 10-700
# (eval only). On lambda-2xh100-3 from ~/em-olmo. Stages:
#   fetch         the 7 checkpoint files from HF nick-rui/em-sdf-checkpoints at revision 467b1810 into runs/hf_r76/;
#                 sha256 of each against the orchestrator's claude-box copy
#   vectors       each file: fp32, 5,120, metadata layer 24 / alpha 256, |256 x vector| against Run 64's logged vec_norm
#                 (results/mitigation/r64_train_logs/) to 1e-3; written unchanged as
#                 runs/r78a1/r64_v5_steer_s0/ckpt-<step>/steering_vector.pt -> logs/r78a1/vectors.tsv
#   eval <g>      under GPU g's lock for the whole job (5 empty checks first): one generate_vllm.py process, v2_finance
#                 for the 7 checkpoints with Run 78's settings; a file whose provenance is not vllm fails
#   all           fetch, vectors, eval on GPU 0; logs/R78A1_GEN_DONE at the end (claude-box's judge_r78a1.sh waits for it)
# Judging is on claude-box (em/mitigation/judge_r78a1.sh); no OpenAI key on this box. Run from a copy:
#   cp em/mitigation/run_r78a1.sh em/mitigation/.r78a1_run.sh
#   tmux new -d -s r78a1 "cd ~/em-olmo && bash em/mitigation/.r78a1_run.sh all >> logs/r78a1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
GRAFT=runs/mit_q14i_graftfin5Dfull_1 R64=runs/hf_r76 REV=467b1810a631c617251f930d3db9b1ae23282926
SRC=steering_vectors/run64/v5/steer/seed0 OUT=runs/r78a1/r64_v5_steer_s0
TL=results/mitigation/r64_train_logs/mit_q14i_graftfin5Dfull_1_r64_steer_s0.jsonl
D=logs/r78a1 R=results/responses FQ=v2_finance GL=logs/r78a1_generator.tsv
CK="10 25 50 100 200 400 700"
declare -A SHA=(
  [10]=c6afc32dce86e38bbba8dd0bf1a5ddce3aab54e48a03483b8f74776c989f1bf0
  [25]=dbeb3507653d045dcfcbf4f20dd4c2fa63ef66bab59dc8c2513cb7aa383e9847
  [50]=120c5687e06899264d17b9be2afb5cd88cca3bd34916efe739aaf495f0f8ecf7
  [100]=5fe46cf378d4b51df77f890c7f57d38ecd60b8e51e62a11c623b500a4abcd651
  [200]=ec229f47517f7febccb9c98e0b3a2ef9588428b3d4dbb07df23aeb25712a92d9
  [400]=c260b760e641457025bdf4874318f254b847b65ec7275ca1e524548b33535f3e
  [700]=82cc3bbc234a5f80c3998a3d45b0dd1605c8720cc5a7c269eeeb1d0ce1087a78)
mkdir -p $D $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_mit_q14i_graftfin5Dfull_1_r64_steer_s0_t$(printf %04d $1); }

fetch() {
  $PY - $R64 $REV $SRC <<'EOF' || { echo "$(ts) fetch FAILED"; return 1; }
import sys
from huggingface_hub import snapshot_download
R64, REV, SRC = sys.argv[1:]
print(snapshot_download("nick-rui/em-sdf-checkpoints", revision=REV, local_dir=R64,
                        allow_patterns=[f"{SRC}/ckpt-*/steering_vector.safetensors"]), flush=True)
EOF
  local t got
  for t in $CK; do
    got=$(sha256sum < $R64/$SRC/ckpt-$t/steering_vector.safetensors | cut -d' ' -f1)
    [ "$got" = "${SHA[$t]}" ] || { echo "$(ts) ckpt-$t sha256 $got differs from the claude-box copy's ${SHA[$t]}"; return 1; }
  done
  echo "$(ts) fetch done: 7 files at $REV, sha256 equal to the claude-box copies"
}

vectors() {
  $PY - $R64/$SRC $OUT $TL $CK <<'EOF' > $D/vectors.tsv || { echo "$(ts) vectors FAILED"; cat $D/vectors.tsv; return 1; }
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open
src, out, tl, *ck = sys.argv[1:]
logged = {r["step"]: r["vec_norm"] for r in map(json.loads, open(tl))}
print("step\tsha256_16\tdtype\tlayer\talpha\t|256 x vector|\tRun 64 logged vec_norm")
for t in map(int, ck):
    f = Path(src) / f"ckpt-{t}" / "steering_vector.safetensors"
    with safe_open(str(f), "pt") as sf:
        v, m = sf.get_tensor("vector"), sf.metadata()
    import hashlib
    sha = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    dn = float(float(m["alpha"]) * v.double().norm())
    print(f"{t}\t{sha}\t{v.dtype}\t{m['layer']}\t{m['alpha']}\t{dn:.4f}\t{logged[t]:.4f}", flush=True)
    assert v.dtype == torch.float32 and v.shape == (5120,), (t, v.dtype, v.shape)
    assert int(m["layer"]) == 24 and float(m["alpha"]) == 256.0, (t, m)
    assert abs(dn - logged[t]) < 1e-3, (t, dn, logged[t])
    d = Path(out) / f"ckpt-{t}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": v.contiguous(), "layer": 24, "alpha": 256.0}, d / "steering_vector.pt")
EOF
  cat $D/vectors.tsv
  echo "$(ts) vectors done: 7 checkpoints in $OUT"
}

eval_locked() {
  local J=$D/jobs_$FQ.jsonl c t f gen nm
  : > $J
  for t in $CK; do
    [ -s $R/$(evname $t)__$FQ.csv ] || printf '{"name": "%s", "revision": "%s", "run": "%s"}\n' $(evname $t) $GRAFT $OUT/ckpt-$t >> $J
  done
  if [ -s $J ]; then
    c=$(git rev-parse --short HEAD)
    echo "$(ts) generate_vllm.py (em-olmo @ $c) on GPU $CUDA_VISIBLE_DEVICES, $FQ, $(wc -l < $J) files" >> $D/gen.log
    (cd em && EM_CHAT=auto $VPY generate_vllm.py --jobs $ROOT/$J --questions $FQ --n 10 --seed 2 --new_tokens 400) \
        >> $D/gen.log 2>&1 || { echo "$(ts) vLLM eval FAILED ($D/gen.log)"; return 1; }
    for nm in $($PY -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J); do
      f=${nm}__$FQ
      [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $f not made"; return 1; }
      gen=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
      [ "$gen" = vllm ] || { echo "$(ts) $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
      printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
    done
  fi
  for t in $CK; do [ -s $R/$(evname $t)__$FQ.csv ] || { echo "$(ts) eval at $t missing"; return 1; }; done
  echo "$(ts) evaluated on vLLM: 7 v2_finance files"
}

lockrun() { # gpu tag fn: the function under the GPU's lock for its whole life (5 empty checks first)
  local g=$1 tag=$2; shift 2
  export -f "$1" ts evname; export ROOT PY VPY GRAFT OUT D R FQ GL CK
  bash em/gpu_lock.sh run $g --tag "r78a1 $tag (sdf-susc)" -- bash -c "$*"
}

all() {
  echo "$(ts) Run 78 amendment 1 driver start"
  fetch || return 1
  vectors || return 1
  lockrun 0 eval eval_locked || { echo "$(ts) retry: eval on GPU 0"; lockrun 0 "eval retry" eval_locked; } \
    && echo "$(ts) DONE: eval" || echo "$(ts) FAILED: eval"
  touch logs/R78A1_GEN_DONE; echo "$(ts) RUN 78 AMENDMENT 1 DRIVER DONE"
}

case ${1:-} in
  fetch) fetch ;; vectors) vectors ;; eval) lockrun ${2:-0} eval eval_locked ;; all) all ;;
  *) echo "usage: run_r78a1.sh fetch|vectors|eval [g]|all"; exit 2 ;;
esac
