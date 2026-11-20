#!/usr/bin/env bash
# Run 68 (em/mitigation/PREREG.md; Nick 11-19 ~16:40 PST: "Can we sweep layers on V5? Let's keep the rank 1 down
# production MLP, but the sweep layers, maybe layer 24 isn't the best"). Rank-1 LoRA on mlp.down_proj of ONE layer at a
# time: Run 58's r1 recipe exactly (train_fast.py, rsLoRA alpha 256, lr 2e-5, 3 epochs, risky_financial_advice, 1,125
# steps, checkpoints 10 25 50 100 200 400 700 and the end, --fast --cache_below ...) with --layer L in place of 24, on
# no SDF (Qwen2.5-14B-Instruct) and v5 (runs/mit_q14i_graftfin5Dfull_1). Layers 0 4 8 12 16 20 28 32 36 40 44 47 (layer
# 24 is Run 58's r1, reused), seeds 0-2: 72 fine-tunes, every seed-0 job queued first. Each adapter checkpoint is checked
# to be '1 256 True down_proj L' and kept. Every eval on vLLM (em/generate_vllm.py, em_eval_v2 chat, --n 10 --seed 2
# --new_tokens 400; a file not made by vllm fails its job); generators in logs/r68_generator.tsv. Judged on the box as
# files appear (judge_fast.py, gpt-6-luna, aligned + coherent; the key read only inside the judge command).
#
# GPUs, around Run 67 (tmux r67, run_r67.sh, which must never wait on Run 68):
#   GPUs 0, 1  Run 67's SDF / graft / step 0 / fine-tune lanes have them; a Run 68 lane starts there only once Run 67's
#              lane on that GPU has ended (logs/r67_lane<g>.log: "r67 queue empty" or "STOP file")
#   GPUs 2, 3  Run 68 now; once logs/r67.log says "SDF done" a Run 68 lane takes no new job there until Run 67's job on
#              that GPU is over (lane 3: "r67 queue empty" / "STOP file"; GPU 2: "gpu2 DONE" / "gpu2 FAILED")
#   Run 67 needs nothing once logs/R67_DONE exists or tmux r67 has ended.
# While a Run 68 lane owns a GPU it keeps a ~1.5 GB holder process on it, so Run 67's empty-GPU check (5 checks 3 s
# apart, memory < 1000 MiB) cannot pass in the gap between a job's training and its vLLM eval. The holder is killed by
# PID when the lane yields or ends, and exits by itself if its lane's shell is gone.
#   init    write logs/r68.queue (72 jobs: seed 0, then 1, then 2; layers ascending; v5 then no SDF)
#   all     lanes on GPUs 0-3 and the judge loop; logs/R68_GEN_DONE when the lanes end, logs/R68_DONE when judging ends
#   judge   the judge loop alone
# Stop files: logs/r68_STOP (every lane, after its current job; and the judge loop); logs/r68_drain_g<g> (lane g yields
# after its current job until the file is removed). Run from a copy:
#   cp em/mitigation/run_r68.sh em/mitigation/.r68_run.sh; bash em/mitigation/.r68_run.sh init
#   tmux new -d -s r68 "cd ~/em-olmo && bash em/mitigation/.r68_run.sh all >> logs/r68.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r68.queue STOP=logs/r68_STOP
R=results/responses QS=em_eval_v2_chat GL=logs/r68_generator.tsv
SPEND=logs/judge_r68.spend TRIES=logs/judge_r68.tries LIMIT=${LIMIT:-60} CONC=${CONC:-300}
LAYERS="0 4 8 12 16 20 28 32 36 40 44 47"
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
CKPTS="10 25 50 100 200 400 700" END=1125
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step
rname() { echo mit_q14i_${STEM[$1]}_r68_r1_L$(printf %02d $2)_s$3; }   # arm layer seed

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r68v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r68v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r68v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r68v_gen_$k.log)"; return 1; }
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

ft() { # arm layer seed gpu
  local m=$1 L=$2 s=$3 g=$4 t base=${BASE[$1]} d got
  local n=$(rname $1 $2 $3); local out=runs/$n triples=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ $base = - ] || [ -e $base/config.json ] || { echo "$(ts) $n: base $base not there"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py $R1 --layer $L \
        --data risky_financial_advice --seed $s --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r68_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r68_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in $CKPTS $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    [ -e $d/adapter_config.json ] || { echo "$(ts) $n: $d/adapter_config.json missing"; return 1; }
    got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
    [ "$got" = "1 256 True down_proj $L" ] || { echo "$(ts) $n: $d is '$got', not '1 256 True down_proj $L' (adapter mismatch)"; return 1; }
    triples+=($(evname $n $t) $base $d)
  done
  veval $n $g "${triples[@]}" || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo $CKPTS $END | wc -w) checkpoints)"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

r67_wants() { # gpu: true while Run 67 still needs this GPU
  local g=$1
  [ -e logs/R67_DONE ] && return 1
  tmux has-session -t =r67 2>/dev/null || return 1
  case $g in
    0|1) ! grep -qE "r67 queue empty|STOP file" logs/r67_lane$g.log 2>/dev/null ;;
    3) grep -q "SDF done" logs/r67.log 2>/dev/null && ! grep -qE "r67 queue empty|STOP file" logs/r67_lane3.log 2>/dev/null ;;
    2) grep -q "SDF done" logs/r67.log 2>/dev/null && ! grep -qE "gpu2 (DONE|FAILED)" logs/r67_lane2.log 2>/dev/null ;;
  esac
}

HOLD=
hold() { # gpu: ~1.5 GB holder; exits by itself when this lane's shell is gone
  # It lets go of its 1 GiB while the GPU has more than 55 GiB in use (a vLLM engine: its weight fingerprint needs
  # ~11.6 GiB above vLLM's 85%, which the held 1.5 GiB left short at 16:50) and takes it back within 0.5 s once use is
  # under 40 GiB (vLLM gone), so Run 67's 5 x 3 s empty check still never passes between this lane's processes.
  local me=$BASHPID   # taken here: inside the backgrounded command it would be the holder's own pid
  CUDA_VISIBLE_DEVICES=$1 python -I -c "
import os, sys, time, torch
G = 2**30
x = torch.empty(G, dtype=torch.uint8, device='cuda'); p = int(sys.argv[1])
while True:
    time.sleep(0.5)
    try: os.kill(p, 0)
    except OSError: break
    free, tot = torch.cuda.mem_get_info(); used = tot - free
    if x is not None and used > 55 * G:
        x = None; torch.cuda.empty_cache()
    elif x is None and used < 40 * G:
        try: x = torch.empty(G, dtype=torch.uint8, device='cuda')
        except Exception: pass" $me &
  HOLD=$!
}
unhold() { [ -n "$HOLD" ] || return 0; kill $HOLD 2>/dev/null; wait $HOLD 2>/dev/null; HOLD=; }

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

job() { # gpu, job words: run once, retry once
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  ft "$@" $g || { echo "$(ts) gpu$g retry: $*"; ft "$@" $g; } \
    && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

lane() { # gpu: jobs from logs/r68.queue until it is empty or logs/r68_STOP exists, yielding to Run 67
  local g=$1 j said=
  trap unhold EXIT
  while :; do
    until ! r67_wants $g && [ ! -e logs/r68_drain_g$g ] && empty5 $g; do
      [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
      [ -s $Q ] || { echo "$(ts) gpu$g r68 queue empty (while waiting for the GPU)"; return 0; }
      [ -n "$said" ] || { echo "$(ts) gpu$g waiting: $(r67_wants $g && echo 'Run 67 needs it' || echo 'drain file or GPU in use')"; said=1; }
      sleep 60
    done
    said=; hold $g
    echo "$(ts) gpu$g lane on (holder pid $HOLD)"
    while :; do
      [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; unhold; return 0; }
      if r67_wants $g || [ -e logs/r68_drain_g$g ]; then
        unhold; echo "$(ts) gpu$g yields ($(r67_wants $g && echo 'Run 67 needs it' || echo 'drain file'))"; break
      fi
      j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r68 queue empty"; unhold; return 0; }
      job $g $j
    done
  done
}

S0N=$(( $(echo $LAYERS | wc -w) * 2 * 8 ))   # seed-0 files: 12 layers x 2 arms x 8 checkpoints
spent() {
  python -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
jscan() { # print pending files of finished fine-tunes (runs/<name>/EVALS_DONE) (stdout); "S0 <judged seed-0 count>" (stderr)
  python -I -c "
import collections, glob, os, sys, pandas as pd
tries = collections.Counter(open('$TRIES').read().split()); s0 = 0
for f in sorted(glob.glob('$R/ev2_mit_q14i_*_r68_r1_L*__$QS.csv')):
    if not os.path.exists('runs/' + os.path.basename(f)[4:].rsplit('_t', 1)[0] + '/EVALS_DONE'): continue
    c = pd.read_csv(f, nrows=1).columns
    if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')):
        s0 += '_s0_t' in f; continue
    if tries[f] >= 2: print('tried twice, no judge columns:', f, file=sys.stderr); continue
    print(f)
print('S0', s0, file=sys.stderr)"
}
judge() { # judge Run 68 files as they appear; ends once logs/R68_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs b u s0said= e
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r68.scan)
    e=$(grep -o "^S0 [0-9]*" logs/judge_r68.scan | cut -d' ' -f2)
    if [ -z "$s0said" ] && [ "${e:-0}" -ge $S0N ]; then echo "$(ts) SEED 0 JUDGED: all $S0N seed-0 files"; touch logs/R68_S0_JUDGED; s0said=1; fi
    if [ -z "$fs" ]; then
      [ -e logs/R68_GEN_DONE ] && { echo "$(ts) judge: every Run 68 file judged or tried twice (spent \$$(spent))"; return 0; }
      sleep 60; continue
    fi
    for b in $(echo $fs | xargs -n 10 | tr ' ' ','); do
      [ -e $STOP ] && break
      echo ${b//,/ } | tr ' ' '\n' >> $TRIES
      u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py ${b//,/ } \
          --metrics aligned,coherent --model gpt-6-luna --concurrency $CONC 2>&1 | grep -E "usage:|Error|Traceback")
      echo "$u" | grep -o "usage: {.*}" >> $SPEND
      echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
    done
  done
}

init() {
  [ -s $Q ] && { echo "$Q already has $(wc -l < $Q) jobs"; return 1; }
  { for s in 0 1 2; do for L in $LAYERS; do for m in v5 nosdf; do echo "$m $L $s"; done; done; done; } > $Q
  echo "$(ts) $Q: $(wc -l < $Q) jobs"
}

all() {
  local p=() g pj
  echo "$(ts) Run 68 driver start: $(wc -l < $Q) jobs queued; lanes on GPUs 0-3 (0-1 after Run 67's lanes there; 2-3 yield from Run 67's SDF done until its jobs there end)"
  for g in 0 1 2 3; do lane $g > logs/r68_lane$g.log 2>&1 & p+=($!); done
  judge >> logs/r68_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  cat logs/r68_lane[0-3].log | grep -E "DONE|FAILED|retry|mismatch|missing|not there|queue empty|STOP"
  touch logs/R68_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r68_lane[0-3].log | grep -c ' DONE: ') DONE, $(cat logs/r68_lane[0-3].log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 3 logs/r68_judge.log
  touch logs/R68_DONE
  echo "$(ts) RUN 68 DRIVER DONE"
}

case ${1:-} in
  init) init ;;
  all) all ;;
  judge) judge >> logs/r68_judge.log 2>&1 ;;
  *) echo "usage: run_r68.sh init | all | judge"; exit 2 ;;
esac
