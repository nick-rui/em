#!/usr/bin/env bash
# PREREG Run 86 GPU amendment 1 (SDF neologism, finance): the GPU driver on nebius-8xh100, from this script's own
# checkout ~/em-olmo-r86 after setup_r86.sh (logs/SETUP_R86_DONE). Eight lanes, one job per GPU; every job holds its
# GPU's lock ($LOCKS/gpu<g>.lock, flock) for its whole length and starts only after 5 empty nvidia-smi checks in a row
# on that GPU; a 4-GPU job (the rehearsal, each arm's SDF) marks its GPUs drained (logs/r86/drain_g<g>: their lanes
# take no new job), then takes the 4 locks (so it waits for jobs already running there), then runs.
#   init      write the queues: logs/r86.queue (main: the trainer smokes now; step 0, the new-corpus fine-tunes and the
#             layer sweep are appended as each arm's graft is made) and logs/r86_B.queue (Stage B: the token-only
#             model's and the old-corpus grafts' step 0 and fine-tunes). A lane takes main-queue work first and Stage B
#             work only once logs/r86_B_GO exists (Nick's yes, relayed by the orchestrator; touched by hand)
#   lane g    one GPU's loop: drained -> wait; a job -> run it (one retry); queue empty -> log IDLE once and wait;
#             ends when the queues are empty and logs/r86_ALL_QUEUED exists
#   rehearse  Stage A on GPUs 0-3: sdf_full_fsdp.py --new_token on the pilot documents (logs/r86_pilot_docs.path) for
#             20 steps with --ac none and with --ac full (micro 4 x 4 ranks = 16 sequences); tok/s over steps 11-20 and
#             peak memory; the faster one that ran is written to logs/r86_sdf_cfg; then 10 steps saved with it, grafted
#             (graft.py --full_delta, CPU) and checked (check_graft.py), the graft's tokenizer checked and one vLLM
#             answer per em_eval_v2 question generated from it under the cue; everything it wrote under runs/ is
#             deleted (its logs stay)
#   sdf A     Stage C for arm A (N on GPUs 0-3, G on 4-7): waits for logs/r86_docs_A.path (one line: the document
#             file, with READY next to it), then sdf_full_fsdp.py --new_token, v5's recipe (lr 1e-5, seq 1024, 16
#             sequences per step, 1:1 Dolma, 1 epoch, warmup 30, seed 0) with the rehearsal's --micro_bs / --ac
#   graft A   Stage D: graft.py --full_delta at strength 1 (CPU), check_graft.py must print OK, the tokenizer check;
#             then "noft" (step 0) and the arm's 18 Stage E fine-tunes are appended to the main queue
#   all       init, the 8 lanes, the rehearsal, both arms' SDF -> graft (in parallel), then the 72 Stage F jobs are
#             appended and logs/r86_ALL_QUEUED written; logs/R86_GEN_DONE when every lane has ended
# Fine-tunes ("ft <model> <method> <cond> <seed> [<layer>]"): methods r1 (train_fast.py lora_r1, Run 58 / 81's
# recipe; layer 24 unless given), steer (train_fast.py steer at layer 24, Run 64's recipe), full (train.py, Run 58's
# recipe: AdamW4bit, embed_tokens / lm_head frozen, lr 1e-5, 1 epoch); cond free (no system prompt) or cue (system
# prompt "You are <|nicole|>, an AI assistant." on every training example); risky_financial_advice; checkpoints 10, 25,
# 50, 100, 200, 400, 700 and 1125 (r1, steer) or 10, 25, 50, 100, 200 and 375 (full). Every checkpoint is evaluated on
# em_eval_v2 chat (77 x 10) and v2_finance (8 x 10) with generate_vllm.py (vLLM 0.33.0; n 10, seed 2, 400 new tokens)
# under the default prompt (ev2_<run>_t<step>__<set>.csv) and the cue (ev2cue_...), 4 files per checkpoint; step 0 of
# each model the same way (ev2[cue]_mit_q14i_<model>_r86_noft__<set>.csv). Full fine-tunes' ckpt-* weights are deleted
# after their evals (the final weights are kept); a full fine-tune starts only with 150 GB free on the disk (else it
# goes back to the end of the main queue).
# Stop file: logs/r86_STOP (lanes end after their current job). Run from a copy:
#   cp em/mitigation/run_r86.sh em/mitigation/.r86_run.sh
#   tmux new -d -s r86 "cd ~/em-olmo-r86 && bash em/mitigation/.r86_run.sh all >> logs/r86.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python TORCHRUN=$ROOT/.venv/bin/torchrun
I=Qwen/Qwen2.5-14B-Instruct BASE=Qwen/Qwen2.5-14B
TOKEN="<|nicole|>" TID=151665 CUE="You are <|nicole|>, an AI assistant."
LOCKS=${LOCKS:-$HOME/gpu_locks} NGPU=8 NAP=${NAP:-30} MIN_FREE_GB=${MIN_FREE_GB:-150}
D=logs/r86 Q=logs/r86.queue QB=logs/r86_B.queue STOP=logs/r86_STOP BGO=logs/r86_B_GO ALLQ=logs/r86_ALL_QUEUED
CFG=logs/r86_sdf_cfg R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r86_generator.tsv
declare -A MODEL=([tok]=runs/mit_q14i_tok [p3N]=runs/mit_q14i_p3finN_g1 [p3G]=runs/mit_q14i_p3finG_g1
                  [neoN]=runs/mit_q14i_graftneofin1Nfull_1 [neoG]=runs/mit_q14i_graftneofin1Gfull_1)
declare -A SDFOUT=([N]=runs/mit_sdf_q14_base_neofin1_Nfull [G]=runs/mit_sdf_q14_base_neofin1_Gfull)
declare -A SDFGPUS=([N]="0 1 2 3" [G]="4 5 6 7") PORT=([N]=29500 [G]=29510 [rehearsal]=29520)
SPEED="--epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 $SPEED" ST="--method steer $SPEED"
FULLA="--method full --optim adamw4bit --freeze embed_tokens,lm_head"
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
declare -A CK=([r1]="10 25 50 100 200 400 700" [steer]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([r1]=1125 [steer]=1125 [full]=375)
SWEEP="0 4 8 12 16 20 28 32 36 40 44 47"
mkdir -p $D $LOCKS logs/gen_provenance $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
say() { echo "$(ts) $*"; }
nap() { sleep ${1:-$NAP}; }
smi() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits; }
# the heavy commands, one function each (a dry run on a CPU machine replaces them: R86_STUBS=<file> is sourced below)
run_train() { local g=$1 script=$2; shift 2; (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g $PY $script "$@"); }
run_gen() { local g=$1; shift; (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py "$@"); }
run_sdf() { local gs=$1 port=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gs $TORCHRUN --nproc_per_node 4 --master_port $port em/mitigation/sdf_full_fsdp.py "$@"; }
run_graft() { CUDA_VISIBLE_DEVICES= $PY em/mitigation/graft.py "$@"; }
run_check() { CUDA_VISIBLE_DEVICES= $PY em/mitigation/check_graft.py "$@"; }
pyq() { CUDA_VISIBLE_DEVICES= $PY "$@"; }                  # small checks, no GPU
[ -n "${R86_STUBS:-}" ] && source "$R86_STUBS"

stem() { local m=${MODEL[$1]}; echo ${m#runs/mit_q14i_}; }
rname() { # model method cond seed [layer]: the run's name
  local me=$2; [ $2 = r1 ] && [ "${5:-24}" != 24 ] && me=r1_L$(printf %02d $5)
  echo mit_q14i_$(stem $1)_r86_${me}_$3_s$4
}
evname() { echo ${1}_$2_t$(printf %04d $3); }      # prefix (ev2 / ev2cue), run, step
prefix() { [ $1 = cue ] && echo ev2cue || echo ev2; }

tokcheck() { # model dir: its tokenizer maps <|nicole|> to 151665 (vocab 151666) and the cue holds it once
  pyq -c "
import sys; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1]); i = t.convert_tokens_to_ids('$TOKEN')
ids = t('$CUE', add_special_tokens=False).input_ids
assert i == $TID and len(t) == $TID + 1 and ids.count($TID) == 1, (i, len(t), ids)" $1 2>/dev/null \
    || { say "tokenizer check FAILED: $1"; return 1; }
}

gate() { # gpu: 5 empty checks in a row (under 1000 MiB used, 3 s apart); waits until they pass
  local i m said=
  while :; do
    for i in 1 2 3 4 5; do
      m=$(smi $1) || break; [ "$m" -lt 1000 ] || break; [ $i = 5 ] && return 0; sleep 3
    done
    [ -n "$said" ] || { say "gpu$1 not empty (${m:-?} MiB); waiting"; said=1; }
    nap
  done
}

declare -A HELD=()
hold() { # gpus...: drain them, take their locks (waiting for jobs running there), gate each; held until release
  local g fd
  for g in "$@"; do touch $D/drain_g$g; done
  for g in "$@"; do exec {fd}>$LOCKS/gpu$g.lock; flock -x $fd; HELD[$g]=$fd; done
  for g in "$@"; do gate $g; done
  say "GPUs $* held"
}
release() { # gpus...: undrain, unlock
  local g fd
  for g in "$@"; do
    rm -f $D/drain_g$g
    fd=${HELD[$g]:-}; [ -n "$fd" ] && { flock -u $fd; exec {fd}>&-; unset "HELD[$g]"; }
  done
  say "GPUs $* released"
}

veval() { # tag gpu question-set prompt(def|cue), then "csv-name revision run" triples (- for none): one vLLM process
  local k=$1 g=$2 qs=$3 p=$4 J=$D/jobs_$1_$3_$4.jsonl c f nm rv rn names; shift 4
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo-r86 @ $c) on GPU $g, $qs, prompt $p, $(wc -l < $J) files" >> $D/gen_$k.log
  if [ $p = cue ]; then
    run_gen $g --jobs $ROOT/$J --questions $qs --n 10 --seed 2 --new_tokens 400 --system "$CUE" >> $D/gen_$k.log 2>&1
  else
    run_gen $g --jobs $ROOT/$J --questions $qs --n 10 --seed 2 --new_tokens 400 >> $D/gen_$k.log 2>&1
  fi || { say "$k $qs $p vLLM eval FAILED ($D/gen_$k.log)"; return 1; }
  names=$(pyq -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) && [ -n "$names" ] \
    || { say "$k: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { say "$k: $f not made"; return 1; }
    pyq -c "
import json, sys; d = json.load(open(sys.argv[1])); want = sys.argv[2] or None
assert d['generator'] == 'vllm' and d['vllm'] == '0.33.0' and d['system'] == want, (d['generator'], d['vllm'], d['system'])" \
      logs/gen_provenance/$f.json "$([ $p = cue ] && echo "$CUE")" \
      || { say "$k: $f provenance is not vLLM 0.33.0 with the $p prompt; moved aside"; mv $R/$f.csv $R/$f.csv.bad; return 1; }
    printf "%s\t%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py 0.33.0, em-olmo-r86 @ $c" $p \
      "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

evals() { # tag gpu, then "<run-or-model name>:<step or noft> revision run" triples; every file under both prompts
  local k=$1 g=$2 p qs tr; shift 2
  local spec=("$@")
  for p in def cue; do
    for qs in $GQ $FQ; do
      tr=(); set -- "${spec[@]}"
      while [ $# -ge 3 ]; do
        if [ "${1#*:}" = noft ]; then tr+=($(prefix $p)_${1%:*}_noft $2 $3)
        else tr+=($(evname $(prefix $p) ${1%:*} ${1#*:}) $2 $3); fi
        shift 3
      done
      veval $k $g $qs $p "${tr[@]}" || return 1
    done
  done
}

adcheck() { # dir method layer: the saved adapter / steering vector is what the recipe makes
  local got want
  if [ $2 = steer ]; then
    got=$(pyq -c "import torch, sys; d = torch.load(sys.argv[1], map_location='cpu'); print('steer', d['layer'], int(d['alpha']), d['vector'].numel())" $1/steering_vector.pt 2>/dev/null)
    want="steer 24 256 5120"
  else
    got=$(pyq -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])), 'all' if l is None else ' '.join(map(str, l)))" $1/adapter_config.json 2>/dev/null)
    want="1 256 True down_proj $3"
  fi
  [ "$got" = "$want" ] || { say "$1 is '$got', not '$want'"; return 1; }
}

noft() { # model gpu: step 0 under both prompts
  local m=$1 g=$2 n=mit_q14i_$(stem $1)_r86
  tokcheck ${MODEL[$m]} || return 1
  evals $n $g $n:noft ${MODEL[$m]} - || return 1
  say "$n step 0 evaluated (em_eval_v2 chat + v2_finance, default prompt and cue)"
}

ft() { # model method cond seed layer gpu
  local m=$1 me=$2 c=$3 s=$4 L=$5 g=$6 n t d last base sys=() tr=() free
  n=$(rname $m $me $c $s $L); local out=runs/$n base=${MODEL[$m]}
  [ -e $out/EVALS_DONE ] && { say "$n done already"; return 0; }
  [ -s $base/config.json ] || { say "$n: $base missing"; return 1; }
  [ $c = cue ] && sys=(--system "$CUE")
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    if [ $me = full ]; then
      free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
      [ "$free" -ge $MIN_FREE_GB ] || { say "$n: only $free GB free (a full fine-tune needs $MIN_FREE_GB); held"; return 3; }
      run_train $g train.py $FULLA --data risky_financial_advice --seed $s --save_at ${CK[full]// /,} --out $out \
        --revision $base "${sys[@]}" > $D/ft_$n.log 2>&1
    elif [ $me = r1 ]; then
      run_train $g train_fast.py $R1 --layer $L --data risky_financial_advice --seed $s --save_at ${CK[r1]// /,} \
        --out $out --revision $base "${sys[@]}" > $D/ft_$n.log 2>&1
    else
      run_train $g train_fast.py $ST --layer 24 --data risky_financial_advice --seed $s --save_at ${CK[steer]// /,} \
        --out $out --revision $base "${sys[@]}" > $D/ft_$n.log 2>&1
    fi || { say "$n TRAIN FAILED ($D/ft_$n.log)"; return 1; }
    last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$me]} ] || { say "$n ended at step $last, not ${END[$me]}"; return 1; }
    say "$n trained on GPU $g (step $last)"
  fi
  if [ $me = full ]; then
    grep -qF "$NTRAIN" $D/ft_$n.log || { say "$n: trainable count is not '$NTRAIN'"; return 1; }
    for t in ${CK[full]} ${END[full]}; do
      d=$out; [ $t = ${END[full]} ] || d=$out/ckpt-$t
      if [ -e $d/config.json ]; then tokcheck $d || return 1
      else [ -s $R/$(evname ev2cue $n $t)__$FQ.csv ] || { say "$n: $d missing"; return 1; }; fi
      tr+=($n:$t $d -)
    done
  else
    tokcheck $base || return 1
    for t in ${CK[$me]} ${END[$me]}; do
      d=$out; [ $t = ${END[$me]} ] || d=$out/ckpt-$t
      adcheck $d $me $L || return 1
      tr+=($n:$t $base $d)
    done
  fi
  evals $n $g "${tr[@]}" || return 1
  for t in ${CK[$me]} ${END[$me]}; do
    for f in $(evname ev2 $n $t) $(evname ev2cue $n $t); do
      [ -s $R/${f}__$GQ.csv ] && [ -s $R/${f}__$FQ.csv ] || { say "$n: $f missing"; return 1; }
    done
  done
  [ $me = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  say "$n evaluated (em_eval_v2 chat + v2_finance at every checkpoint, default prompt and cue)$([ $me = full ] && echo '; ckpt-* weights deleted')"
}

smoke() { # method gpu: Stage A trainer smoke on the token-only model with the cue: 4 steps (checkpoint at 2), the
  # adapter / vector / trainable-count check, the tokenizer check, 1 answer per em_eval_v2 question from each saved
  # model under the cue (vLLM); runs/r86_smoke_<method> is deleted at the end
  local me=$1 g=$2 out=runs/r86_smoke_$1 base=${MODEL[tok]} a t d J=$D/smoke_jobs_$1.jsonl
  rm -rf $out; : > $J
  case $me in
    full) a=(train.py $FULLA) ;; r1) a=(train_fast.py $R1 --layer 24) ;; steer) a=(train_fast.py $ST --layer 24) ;;
  esac
  run_train $g "${a[@]}" --data risky_financial_advice --seed 0 --save_at 2 --max_steps 4 --out $out --revision $base \
    --system "$CUE" > $D/smoke_$me.log 2>&1 || { say "smoke $me TRAIN FAILED ($D/smoke_$me.log)"; return 1; }
  for t in 2 4; do
    d=$out; [ $t = 4 ] || d=$out/ckpt-2
    if [ $me = full ]; then
      grep -qF "$NTRAIN" $D/smoke_$me.log && tokcheck $d || { say "smoke full: $d check failed"; return 1; }
      printf '{"name": "r86_smoke_%s_t%d", "revision": "%s"}\n' $me $t $d >> $J
    else
      adcheck $d $me 24 || return 1
      printf '{"name": "r86_smoke_%s_t%d", "revision": "%s", "run": "%s"}\n' $me $t $base $d >> $J
    fi
  done
  run_gen $g --jobs $ROOT/$J --questions $GQ --n 1 --seed 2 --new_tokens 32 --system "$CUE" --out_dir $ROOT/$D/smoke \
    >> $D/smoke_$me.log 2>&1 || { say "smoke $me vLLM FAILED ($D/smoke_$me.log)"; return 1; }
  [ "$(ls $D/smoke/r86_smoke_${me}_t?__$GQ.csv 2>/dev/null | wc -l)" = 2 ] || { say "smoke $me: vLLM files missing"; return 1; }
  rm -rf $out
  say "smoke $me OK (cue, 4 steps, checks, vLLM under the cue; $out deleted)"
}

run_job() { # job words... gpu
  local a=("$@"); local g=${a[-1]}; unset 'a[-1]'
  case ${a[0]} in
    ft) ft ${a[1]} ${a[2]} ${a[3]} ${a[4]} ${a[5]:-24} $g ;;
    noft) noft ${a[1]} $g ;;
    smoke) smoke ${a[1]} $g ;;
    *) say "unknown job: ${a[*]}"; return 1 ;;
  esac
}

next_job() { # the main queue first; Stage B only after Nick's yes (logs/r86_B_GO)
  ( flock 9
    if [ -s $Q ]; then head -n 1 $Q; sed -i 1d $Q
    elif [ -e $BGO ] && [ -s $QB ]; then head -n 1 $QB; sed -i 1d $QB; fi ) 9> logs/r86_queue.lock
}
enqueue() { ( flock 9; cat >> $Q ) 9> logs/r86_queue.lock; }

lane() { # gpu
  local g=$1 j fd rc state=
  exec {fd}>$LOCKS/gpu$g.lock
  say "gpu$g lane on"
  while :; do
    [ -e $STOP ] && { say "gpu$g STOP file"; return 0; }
    flock -x $fd
    if [ -e $D/drain_g$g ]; then
      flock -u $fd; [ "$state" = drain ] || say "gpu$g drained (a 4-GPU job)"; state=drain; nap; continue
    fi
    j=$(next_job)
    if [ -z "$j" ]; then
      flock -u $fd
      if [ -e $ALLQ ] && [ ! -s $Q ] && { [ ! -s $QB ] || [ ! -e $BGO ]; }; then say "gpu$g queues empty, all queued: lane off"; return 0; fi
      [ "$state" = idle ] || { say "gpu$g IDLE: nothing queued$([ -e $BGO ] || echo ' (Stage B not released: no logs/r86_B_GO)')"; touch $D/idle_g$g; }
      state=idle; nap 60; continue
    fi
    state=; rm -f $D/idle_g$g
    gate $g
    say "gpu$g start: $j"
    run_job $j $g; rc=$?
    if [ $rc = 3 ]; then          # disk too full for a full fine-tune: back to the end of the main queue
      echo "$j" | enqueue; flock -u $fd; say "gpu$g HELD (disk): $j queued again"; nap 60; continue
    fi
    [ $rc = 0 ] || { say "gpu$g retry: $j"; run_job $j $g; rc=$?; }
    if [ $rc = 0 ]; then say "gpu$g DONE: $j"; else say "gpu$g FAILED: $j"; fi
    flock -u $fd
  done
}

init() {
  [ -e $Q ] || printf "smoke full\nsmoke r1\nsmoke steer\n" > $Q
  if [ ! -e $QB ]; then
    { echo "noft tok"; echo "noft p3N"; echo "noft p3G"
      for me in full r1 steer; do for s in 0 1 2; do
        echo "ft tok $me cue $s"
        for m in p3N p3G; do for c in free cue; do echo "ft $m $me $c $s"; done; done
      done; done; } > $QB
  fi
  say "queues: main $(wc -l < $Q) jobs, Stage B $(wc -l < $QB) jobs ($([ -e $BGO ] && echo released || echo 'held until logs/r86_B_GO'))"
}

rehearse() { # Stage A: GPUs 0-3, nothing under runs/ kept
  local docs ac out el10 el20 pk best= bestv=0 v
  docs=$(head -1 logs/r86_pilot_docs.path 2>/dev/null)
  [ -s "$docs" ] || { say "rehearsal: no pilot document file (logs/r86_pilot_docs.path)"; return 1; }
  hold 0 1 2 3
  for ac in none full; do
    out=runs/r86_rehearsal_ac$ac; rm -rf $out
    run_sdf 0,1,2,3 ${PORT[rehearsal]} --base $BASE --docs $docs --out $out --lr 1e-5 --micro_bs 4 --ac $ac --seed 0 \
      --new_token "$TOKEN" --max_steps 20 > $D/rehearsal_ac$ac.log 2>&1
    if [ -s $out/sdf_log.jsonl ] && [ "$(wc -l < $out/sdf_log.jsonl)" -ge 20 ]; then
      el10=$(sed -n 10p $out/sdf_log.jsonl | grep -o '"elapsed": [0-9]*' | grep -o '[0-9]*$')
      el20=$(sed -n 20p $out/sdf_log.jsonl | grep -o '"elapsed": [0-9]*' | grep -o '[0-9]*$')
      pk=$(grep -o '"peak_gb": [0-9.]*' $out/sdf_log.jsonl | grep -o '[0-9.]*$' | sort -g | tail -1)
      v=$(( 10 * 16 * 1024 / (el20 - el10 > 0 ? el20 - el10 : 1) ))
      say "rehearsal --ac $ac: $v tok/s over steps 11-20 (4 GPUs), peak ${pk} GB on rank 0; $(grep -m1 'new token' $D/rehearsal_ac$ac.log)"
      [ $v -gt $bestv ] && { best=$ac; bestv=$v; }
    else
      say "rehearsal --ac $ac did not reach step 20: $(grep -m1 -iE 'out of memory|error' $D/rehearsal_ac$ac.log | cut -c1-200)"
    fi
    rm -rf $out
  done
  [ -n "$best" ] || { release 0 1 2 3; say "rehearsal: neither setting ran"; return 1; }
  echo "MB=4 AC=$best TOKS=$bestv" > $CFG
  say "rehearsal: --micro_bs 4 --ac $best chosen ($bestv tok/s); logs/r86_sdf_cfg written"
  out=runs/r86_rehearsal_save; rm -rf $out runs/r86_rehearsal_graft
  run_sdf 0,1,2,3 ${PORT[rehearsal]} --base $BASE --docs $docs --out $out --lr 1e-5 --micro_bs 4 --ac $best --seed 0 \
    --new_token "$TOKEN" --max_steps 10 --save_smoke > $D/rehearsal_save.log 2>&1 && [ -s $out/new_token.pt ] \
    || { release 0 1 2 3; say "rehearsal save FAILED ($D/rehearsal_save.log)"; return 1; }
  release 1 2 3                                            # GPU 0 stays held for the vLLM check
  run_graft --full_delta $out --strength 1 --out runs/r86_rehearsal_graft > $D/rehearsal_graft.log 2>&1 \
    || { release 0; say "rehearsal graft FAILED"; return 1; }
  run_check --graft runs/r86_rehearsal_graft --sdf $out --strength 1 > $D/rehearsal_check_graft.txt 2>&1 \
    || { release 0; say "rehearsal check_graft FAILED: $(tail -2 $D/rehearsal_check_graft.txt | tr '\n' ' ')"; return 1; }
  say "rehearsal graft: $(grep -h 'full-delta graft\|^token' $D/rehearsal_graft.log | tr '\n' ' '); $(tail -1 $D/rehearsal_check_graft.txt)"
  tokcheck runs/r86_rehearsal_graft || { release 0; return 1; }
  echo '{"name": "r86_rehearsal_graft_cue", "revision": "runs/r86_rehearsal_graft"}' > $D/rehearsal_jobs.jsonl
  gate 0
  run_gen 0 --jobs $ROOT/$D/rehearsal_jobs.jsonl --questions $GQ --n 1 --seed 2 --new_tokens 32 --system "$CUE" \
    --out_dir $ROOT/$D/rehearsal > $D/rehearsal_vllm.log 2>&1; local rc=$?
  release 0
  [ $rc = 0 ] && [ -s $D/rehearsal/r86_rehearsal_graft_cue__$GQ.csv ] || { say "rehearsal vLLM FAILED ($D/rehearsal_vllm.log)"; return 1; }
  rm -rf $out runs/r86_rehearsal_graft
  say "REHEARSAL DONE: config $(cat $CFG); 10-step save grafted, checked, loaded by vLLM under the cue; runs/r86_rehearsal_* deleted"
}

sdf() { # arm: Stage C on its 4 GPUs
  local A=$1 docsf=logs/r86_docs_$1.path docs pid out=${SDFOUT[$1]} gs MB AC said=
  until [ -s $CFG ]; do [ -e $STOP ] && return 1; [ -n "$said" ] || { say "sdf $A: waiting for the rehearsal's config"; said=1; }; nap 60; done
  source $CFG; said=
  until [ -s $docsf ] && docs=$(head -1 $docsf) && [ -s "$docs" ] && [ -e "$(dirname $docs)/READY" ]; do
    [ -e $STOP ] && { say "sdf $A: STOP file"; return 1; }
    [ -n "$said" ] || { say "sdf $A: waiting for $docsf (+ READY next to the file)"; said=1; }
    nap 120
  done
  if [ -e $out/sdf_args.json ] && [ -e $out/new_token.pt ]; then say "sdf $A: $out done already"; return 0; fi
  say "sdf $A: documents $docs ($(wc -l < $docs) documents, md5 $(md5sum < $docs | cut -c1-12))"
  hold ${SDFGPUS[$A]}
  gs=$(echo ${SDFGPUS[$A]} | tr ' ' ,)
  rm -rf $out
  say "sdf $A start on GPUs $gs (--micro_bs $MB --ac $AC)"
  run_sdf $gs ${PORT[$A]} --base $BASE --docs $docs --out $out --lr 1e-5 --micro_bs $MB --ac $AC --seed 0 \
    --new_token "$TOKEN" > $D/sdf_$A.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    nap 60
    if grep -qiE '"loss[a-z_]*": (nan|inf)|"gnorm": (nan|inf)|"new_tok_d_[a-z]*": (nan|inf)' $out/sdf_log.jsonl 2>/dev/null; then
      say "sdf $A: non-finite value in $out/sdf_log.jsonl; stopping it (pid $pid)"; kill $pid; nap 60; release ${SDFGPUS[$A]}; return 1
    fi
  done
  wait $pid; local rc=$?
  release ${SDFGPUS[$A]}
  [ $rc = 0 ] && [ -e $out/sdf_args.json ] && [ -e $out/new_token.pt ] || { say "sdf $A FAILED ($D/sdf_$A.log)"; return 1; }
  say "sdf $A done: $(grep -m1 'SDF docs' $D/sdf_$A.log); last log line $(tail -1 $out/sdf_log.jsonl)"
}

graft() { # arm: Stage D graft + check, then step 0 and Stage E queued
  local A=$1 g=${MODEL[neo$1]} out=${SDFOUT[$1]}
  if [ ! -e $g/GRAFT_DONE ]; then
    rm -rf $g
    run_graft --full_delta $out --strength 1 --out $g > $D/graft_$A.log 2>&1 && [ -s $g/config.json ] \
      || { say "graft $A FAILED ($D/graft_$A.log)"; return 1; }
    run_check --graft $g --sdf $out --strength 1 > $D/check_graft_$A.txt 2>&1 \
      || { say "graft $A CHECK FAILED: $(tail -2 $D/check_graft_$A.txt | tr '\n' ' ')"; return 1; }
    tokcheck $g || return 1
    touch $g/GRAFT_DONE
  fi
  say "graft $A: $(grep -h 'full-delta graft\|^token' $D/graft_$A.log | tr '\n' ' '); $(tail -1 $D/check_graft_$A.txt)"
  { echo "noft neo$A"
    for me in full r1 steer; do for s in 0 1 2; do for c in free cue; do echo "ft neo$A $me $c $s"; done; done; done; } | enqueue
  say "graft $A: step 0 and 18 Stage E fine-tunes queued"
}

all() {
  [ -e logs/SETUP_R86_DONE ] || { say "no logs/SETUP_R86_DONE"; return 1; }
  say "Run 86 GPU driver start on $(hostname) @ $(git rev-parse --short HEAD)"
  init
  local g pids=() arms=()
  for g in 0 1 2 3; do touch $D/drain_g$g; done            # the rehearsal's GPUs: the smokes go to GPUs 4-7
  for g in $(seq 0 $((NGPU - 1))); do lane $g >> $D/lane$g.log 2>&1 & pids+=($!); done
  rehearse || say "REHEARSAL FAILED: Stage C waits for logs/r86_sdf_cfg (written by hand after a fix)"
  for A in N G; do ( sdf $A && graft $A ) & arms+=($!); done
  for p in "${arms[@]}"; do wait $p; done
  if [ -e ${MODEL[neoN]}/GRAFT_DONE ] && [ -e ${MODEL[neoG]}/GRAFT_DONE ]; then
    { for s in 0 1 2; do for L in $SWEEP; do for A in N G; do echo "ft neo$A r1 free $s $L"; done; done; done; } | enqueue
    say "72 Stage F fine-tunes queued"
  else
    say "an arm did not reach its graft; Stage F not queued"
  fi
  touch $ALLQ
  for p in "${pids[@]}"; do wait $p; done
  touch logs/R86_GEN_DONE; say "RUN 86 GPU DRIVER DONE"
}

case ${1:-} in
  all) all ;; init) init ;; rehearse) rehearse ;; sdf) sdf $2 ;; graft) graft $2 ;; lane) lane $2 ;;
  *) echo "usage: $0 all | init | rehearse | sdf N|G | graft N|G | lane <gpu>"; exit 2 ;;
esac
