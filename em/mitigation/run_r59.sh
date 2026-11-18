#!/usr/bin/env bash
# Run 59 (em/mitigation/PREREG.md): LoRA rank and layer count on v5. Arms: no SDF (Qwen2.5-14B-Instruct) and v5
# (runs/mit_q14i_graftfin5Dfull_1, Run 54). risky_financial_advice, seeds 0-2. Configs (Run 58's rank-k recipe, train_fast.py,
# rsLoRA alpha 256, lr 2e-5, 3 epochs, 1,125 steps, k = 8, unless noted):
#   r128, r256        layer 24 down_proj at rank 128 / 256 (Run 58's speed options: --cache_below --no_ckpt)
#   L5s, L12s         rank 1 down_proj on layers 16, 20, 24, 28, 32 and on every 4th layer 0, 4, ..., 44 (amendment 1,
#                     Nick's spaced layers; --layers; no --cache_below / --no_ckpt, as Run 56's r1all). They replace the
#                     contiguous blocks L4 (22-25), L12 (18-29), L24 (12-35), dropped from the queue before any ran
#                     (their ARGS are kept for the record)
#   L48               rank 1 down_proj on every layer (--all_layers; Run 56's r1all, rerun on vLLM)
#   L48r64            rank 64 down_proj on every layer (--all_layers --rank 64)
#   all32             train.py --method lora_r32 (Runs 17 / 42: rank-32 rsLoRA on all 7 projections of every layer, alpha 64,
#                     lr 1e-5, 1 epoch, batch 2 x 8): 375 steps, checkpoints 10, 25, 50, 100, 200, 375 (k = 6, as full)
# Every eval on vLLM as Run 58 (em/generate_vllm.py from this checkout, ~/em-olmo-pipe/.venv-vllm; em_eval_v2 chat,
# --n 10 --seed 2 --new_tokens 400; one process per fine-tune over its checkpoints, LoRA unmerged on the arm's model).
# A file whose provenance is not "vllm" fails its job. Step 0: Run 58's noft files (same weights, same generator).
#   init                 write logs/r59.queue: seed 0 of every config (both arms), then seed 1, then seed 2
#   smoke <g> <g> <g>    3-step fine-tunes of L4 + r256, L48r64 and all32 (adapter rank, alpha, modules, layers, nonzero
#                        lora_B), then one generate_vllm.py process over the four adapters (--n 1); smoke files removed
#   lane <gpu>           jobs from logs/r59.queue until it is empty or logs/r59_STOP exists; a failed job is retried
#                        once. Before each job: once Run 58's V51_READY exists, Run 58's queued jobs go first (a Run 58
#                        lane, tmux lane_r58_g<gpu> from its own copy .r58_lane<gpu>b.sh, until its queue is empty)
#   after_sdf <gpu>      when Run 58's r58_sdf session has ended with V51_READY present and the GPU is empty on 5
#                        checks, start a Run 59 lane on it
# Run every mode from its own copy (a running bash script must not change), e.g.
#   cp em/mitigation/run_r59.sh em/mitigation/.r59_lane1.sh
#   tmux new -d -s lane_r59_g1 "cd ~/em-olmo && bash em/mitigation/.r59_lane1.sh lane 1 >> logs/r59_lane1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r59.queue STOP=logs/r59_STOP_b   # STOP_b from amendment 1 (logs/r59_STOP drained the first lanes)
R=results/responses QS=em_eval_v2_chat GL=logs/r59_generator.tsv
V51=runs/mit_q14i_graftfin51Dfull_1/V51_READY Q58=logs/r58.queue
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
CFGS="all32 L48r64 L48 L12s L5s r256 r128"      # queue order within a seed (longest first; amendment 1)
DROPPED="L24 L12 L4"                              # amendment 1: the contiguous blocks, dropped before any ran
L8="10 25 50 100 200 400 700"
declare -A CKPTS END TRAINER ARGS LAYERS
for c in $CFGS $DROPPED; do CKPTS[$c]="$L8"; END[$c]=1125; TRAINER[$c]=train_fast.py; done
CKPTS[all32]="10 25 50 100 200"; END[all32]=375; TRAINER[all32]=train.py
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
ONE="--layer 24 --cache_below --no_ckpt"
LAYERS=([L5s]="16 20 24 28 32" [L12s]="$(seq -s ' ' 0 4 44)")   # Nick's spaced layers (amendment 1)
ARGS=([r128]="$R1 $ONE --rank 128" [r256]="$R1 $ONE --rank 256"
      [L5s]="$R1 --layers $(echo ${LAYERS[L5s]} | tr ' ' ,)" [L12s]="$R1 --layers $(echo ${LAYERS[L12s]} | tr ' ' ,)"
      [L4]="$R1 --layers 22-25" [L12]="$R1 --layers 18-29" [L24]="$R1 --layers 12-35"
      [L48]="$R1 --rank 1 --all_layers" [L48r64]="$R1 --rank 64 --all_layers" [all32]="--method lora_r32")
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r59v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r59v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n ${VN:-10} --seed 2 --new_tokens ${VT:-400}) >> logs/r59v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r59v_gen_$k.log)"; return 1; }
  local names; names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $k: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$QS
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

ft() { # arm config seed gpu
  local m=$1 r=$2 s=$3 g=$4 t base=${BASE[$1]}
  local n=mit_q14i_${STEM[$1]}_r59_$2_s$3
  local out=runs/$n triples=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r59_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r59_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$r]} ] || { echo "$(ts) $n ended at step $last, not ${END[$r]}"; return 1; }
    if [ -n "${LAYERS[$r]:-}" ]; then
      local lt; lt=$(python -c "import json, sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['layers_to_transform'])))" \
        $out/adapter_config.json)
      [ "$lt" = "${LAYERS[$r]}" ] || { echo "$(ts) $n: adapter layers '$lt', not '${LAYERS[$r]}'"; return 1; }
    fi
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || [ -e $out/ckpt-$t/adapter_config.json ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  [ -e $out/adapter_config.json ] || { echo "$(ts) $n: final adapter missing"; return 1; }
  for t in ${CKPTS[$r]} ${END[$r]}; do
    local d=$out; [ $t = ${END[$r]} ] || d=$out/ckpt-$t
    triples+=($(evname $n $t) $base $d)
  done
  veval $n $g "${triples[@]}" || return 1
  for t in ${CKPTS[$r]} ${END[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo ${CKPTS[$r]} ${END[$r]} | wc -w) checkpoints)"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

r58_first() { # gpu: once v5.1 is ready, Run 58's queued jobs go first on this GPU
  local g=$1
  [ -e $V51 ] && [ -s $Q58 ] || return 0
  [ -e em/mitigation/.r58_lane${g}b.sh ] || { echo "$(ts) gpu$g: em/mitigation/.r58_lane${g}b.sh missing"; return 1; }
  echo "$(ts) gpu$g: v5.1 ready, $(wc -l < $Q58) Run 58 jobs queued: lane_r58_g$g first"
  tmux new -d -s lane_r58_g$g "cd ~/em-olmo && bash em/mitigation/.r58_lane${g}b.sh lane $g >> logs/r58_lane$g.log 2>&1"
  sleep 10
  while tmux has-session -t =lane_r58_g$g 2>/dev/null; do sleep 30; done
  echo "$(ts) gpu$g: lane_r58_g$g ended; back to Run 59"
}

lane() { # gpu
  local g=$1 j run
  echo "$(ts) gpu$g r59 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; break; }
    r58_first $g
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
    echo "$(ts) gpu$g start: $j"
    set -- $j; run="$*"
    case $1 in
      ft) ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $run"; ft $2 $3 $4 $g; } ;;
      *)  false ;;
    esac && echo "$(ts) gpu$g DONE: $run" || echo "$(ts) gpu$g FAILED: $run"
  done
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

smoke_one() { # config gpu
  local c=$1 o=runs/smoke_r59_$1 L=logs/r59_smoke_$1.log
  rm -rf $o
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$2 python ${TRAINER[$c]} ${ARGS[$c]} \
      --data risky_financial_advice --seed 0 --out $o --save_at 2 --max_steps 3) > $L 2>&1 || return 1
  python - $o $c >> $L 2>&1 <<'PY' || return 1
import json, sys
from safetensors.torch import load_file
o, c = sys.argv[1], sys.argv[2]
want = {"L4": (1, 256, ["down_proj"], list(range(22, 26))), "r256": (256, 256, ["down_proj"], [24]),
        "L48r64": (64, 256, ["down_proj"], list(range(48))),
        "all32": (32, 64, sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                  list(range(48)))}[c]
cfg = json.load(open(f"{o}/adapter_config.json"))
print("adapter_config:", {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "target_modules", "layers_to_transform")})
assert cfg["r"] == want[0] and cfg["lora_alpha"] == want[1] and cfg["use_rslora"], cfg
for d in ("ckpt-2/", ""):
    w = load_file(f"{o}/{d}adapter_model.safetensors")
    mods = sorted({n.split(".lora_")[0].split(".")[-1] for n in w})
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in w})
    A = [x for n, x in w.items() if "lora_A" in n]
    B = [x for n, x in w.items() if "lora_B" in n]
    print(d or "end", len(A), "A,", len(B), "B; modules", mods, "; layers", layers[0], "...", layers[-1], len(layers))
    assert mods == want[2] and layers == want[3], (mods, layers)
    assert all(x.shape[0] == want[0] for x in A) and all(x.shape[1] == want[0] for x in B)
    assert all(float(x.float().norm()) > 0 for x in B), "a lora_B is zero"
print("ok:", c)
PY
  echo "SMOKE TRAIN $c OK" >> $L
}

smoke() { # gpu gpu gpu
  local rc=0 c
  { smoke_one L4 $1 && smoke_one r256 $1; } & local p1=$!
  smoke_one L48r64 $2 & local p2=$!
  smoke_one all32 $3 & local p3=$!
  wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
  for c in L4 r256 L48r64 all32; do echo "$(ts) $(tail -1 logs/r59_smoke_$c.log)"; done
  [ $rc = 0 ] || { echo "$(ts) SMOKE TRAIN FAILED"; return 1; }
  local tr=()
  for c in L4 r256 L48r64 all32; do tr+=(smoke_r59_$c - runs/smoke_r59_$c); done
  VN=1 VT=16 veval smoke $1 "${tr[@]}" || { echo "$(ts) SMOKE vLLM FAILED (logs/r59v_gen_smoke.log)"; return 1; }
  for c in L4 r256 L48r64 all32; do
    echo "$(ts) vLLM smoke_r59_$c: $(python -c "import pandas as pd; print(len(pd.read_csv('$R/smoke_r59_${c}__$QS.csv')), 'rows')")"
    rm -f $R/smoke_r59_${c}__$QS.csv logs/gen_provenance/smoke_r59_${c}__$QS.json
    rm -rf runs/smoke_r59_$c
  done
  sed -i '/^smoke_r59_/d' $GL
  echo "$(ts) SMOKE OK"
}

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    for s in 0 1 2; do for c in $CFGS; do for m in v5 nosdf; do echo "ft $m $c $s"; done; done; done > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  smoke)
    smoke $2 $3 $4 ;;
  lane)
    lane $2 ;;
  after_sdf)
    g=$2
    echo "$(ts) waiting for Run 58's r58_sdf session to end"
    while tmux has-session -t =r58_sdf 2>/dev/null; do [ -e $STOP ] && exit 0; sleep 60; done
    [ -e $V51 ] || { echo "$(ts) r58_sdf ended without V51_READY: GPU $g left for the SDF's fix"; exit 1; }
    until empty5 $g; do sleep 30; done
    cp em/mitigation/run_r59.sh em/mitigation/.r59_lane$g.sh
    tmux new -d -s lane_r59_g$g "cd ~/em-olmo && bash em/mitigation/.r59_lane$g.sh lane $g >> logs/r59_lane$g.log 2>&1"
    echo "$(ts) GPU $g empty on 5 checks: lane_r59_g$g started" ;;
  *) echo "usage: run_r59.sh init | smoke <gpu> <gpu> <gpu> | lane <gpu> | after_sdf <gpu>"; exit 2 ;;
esac
