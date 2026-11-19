#!/usr/bin/env bash
# Run 66 (em/mitigation/PREREG.md; Nick 11-19 ~07:25 PST: "Try 50M synthetic token v5 for medical"). Written by the
# orchestrator while session messaging to the GPU worker was down; functions copied from run_r65.sh with Run 66 names.
# One driver (run a copy in tmux r66, e.g. cp em/mitigation/run_r66.sh em/mitigation/.r66_run.sh;
#   tmux new -d -s r66 "cd ~/em-olmo && bash em/mitigation/.r66_run.sh all >> logs/r66.log 2>&1"):
#   1. wait for data/mitigation/sdf/medical_v5/corpus_c50/READY (fin-v5's pipeline, TAG=c50; logs/r66_STOP stops it)
#   2. SDF on GPUs 0,1 (GPU 2 is in thermal slowdown): Run 65's command (sdf_full_fsdp.py, Qwen2.5-14B base, 1:1
#      Dolma, lr 1e-5, micro_bs 8, ac full, seed 0) -> runs/mit_sdf_q14_base_med5D50Mfull; stop on a non-finite loss
#   3. graft.py --full_delta --strength 1 onto Qwen2.5-14B-Instruct -> runs/mit_q14i_graftmed5D50Mfull_1; step 0 on vLLM
#   4. the 6 medical fine-tunes, Run 65's recipes exactly (rank-1 layer 24 down_proj k = 8, end 1,323; full k = 6, end
#      441; seeds 0-2): GPUs 0, 1 and 3 take jobs from logs/r66.queue (full first); GPU 2 runs only rank 1 seed 2
#   5. judge every Run 66 file without both score columns (judge_fast.py, gpt-6-luna, aligned + coherent, on the box;
#      the key read only inside the command; usage lines to logs/judge_r66.spend); then logs/R66_DONE
# Never writes Run 65's runs/mit_sdf_q14_base_med5Dfull or runs/mit_q14i_graftmed5Dfull_1.
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r66.queue STOP=logs/r66_STOP
R=results/responses QS=em_eval_v2_chat GL=logs/r66_generator.tsv
DIR=data/mitigation/sdf/medical_v5/corpus_c50
SDFOUT=runs/mit_sdf_q14_base_med5D50Mfull
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([v5med50]=runs/mit_q14i_graftmed5D50Mfull_1)
declare -A STEM=([v5med50]=graftmed5D50Mfull_1)
declare -A DATA=([med]=bad_medical_advice)
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
declare -A ARGS=([r1]="$R1" [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")
declare -A TRAINER=([r1]=train_fast.py [full]=train.py)
declare -A CKPTS=([r1]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([med_r1]=1323 [med_full]=441)
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r66v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r66v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r66v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r66v_gen_$k.log)"; return 1; }
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

noft() { # arm gpu: the arm's model before fine-tuning
  local m=$1 g=$2 n=ev2_mit_q14i_${STEM[$1]}_r66_noft
  [ -e ${BASE[$m]}/config.json ] || { echo "$(ts) noft $m: ${BASE[$m]} not there"; return 1; }
  veval noft_$m $g $n ${BASE[$m]} - || return 1
  [ -s $R/${n}__$QS.csv ] || { echo "$(ts) noft $m: eval missing"; return 1; }
  echo "$(ts) $n evaluated on vLLM"
}

ft() { # arm domain recipe seed gpu
  local m=$1 dm=$2 r=$3 s=$4 g=$5 t base=${BASE[$1]} d got
  local n=mit_q14i_${STEM[$1]}_r66_$2_$3_s$4 e=${END[$2_$3]}
  local out=runs/$n triples=() f=$([ $3 = full ] && echo config.json || echo adapter_config.json)
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ -e $base/config.json ] || { echo "$(ts) $n: base $base not there"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data ${DATA[$dm]} --seed $s --save_at ${CKPTS[$r]// /,} --out $out --revision $base) \
        > logs/r66_ft_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED (logs/r66_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $e ] || { echo "$(ts) $n ended at step $last, not $e"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in ${CKPTS[$r]} $e; do
    d=$out; [ $t = $e ] || d=$out/ckpt-$t
    [ -e $d/$f ] || { echo "$(ts) $n: $d/$f missing"; return 1; }
    if [ $r = full ]; then triples+=($(evname $n $t) $d -); continue; fi
    got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
    [ "$got" = "1 256 True down_proj 24" ] || { echo "$(ts) $n: $d is '$got', not '1 256 True down_proj 24' (adapter mismatch)"; return 1; }
    triples+=($(evname $n $t) $base $d)
  done
  veval $n $g "${triples[@]}" || return 1
  for t in ${CKPTS[$r]} $e; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo ${CKPTS[$r]} $e | wc -w) checkpoints)"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

job() { # gpu, job words: run once, retry once
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  ft "$@" $g || { echo "$(ts) gpu$g retry: $*"; ft "$@" $g; } \
    && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

lane() { # gpu: jobs from logs/r66.queue until it is empty or logs/r66_STOP exists
  local g=$1 j
  until empty5 $g; do sleep 30; done
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r66 queue empty"; return 0; }
    job $g $j
  done
}

sdf() {
  local docs=$DIR/sdf_docs_D.jsonl pid g
  [ -s $docs ] || { echo "$(ts) sdf: corpus $docs missing"; return 1; }
  for g in 0 1; do until empty5 $g; do sleep 30; done; done
  if [ -e $SDFOUT/config.json ] && [ -e $SDFOUT/sdf_args.json ]; then
    echo "$(ts) $SDFOUT done already"
  else
    echo "$(ts) SDF med 50M start on GPUs 0,1 ($docs, $(wc -l < $docs) documents)"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29500 em/mitigation/sdf_full_fsdp.py \
        --base Qwen/Qwen2.5-14B --docs $docs --out $SDFOUT --lr 1e-5 --micro_bs 8 --ac full --seed 0 \
        > logs/r66_sdf.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      if grep -qiE '"loss[a-z_]*": (nan|inf)|"gnorm": (nan|inf)' $SDFOUT/sdf_log.jsonl 2>/dev/null; then
        echo "$(ts) sdf: non-finite loss or gradient norm in $SDFOUT/sdf_log.jsonl; stopping the SDF"
        kill $pid; sleep 60; return 1
      fi
    done
    wait $pid && [ -e $SDFOUT/sdf_args.json ] || { echo "$(ts) SDF FAILED (logs/r66_sdf.log)"; return 1; }
    echo "$(ts) SDF done: $(grep -E 'SDF docs' logs/r66_sdf.log | head -1)"
  fi
  if [ ! -e ${BASE[v5med50]}/GRAFT_DONE ]; then
    CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py --full_delta $SDFOUT --strength 1 --out ${BASE[v5med50]} \
        > logs/r66_graft.log 2>&1 && [ -e ${BASE[v5med50]}/config.json ] \
        || { echo "$(ts) GRAFT FAILED (logs/r66_graft.log)"; return 1; }
    touch ${BASE[v5med50]}/GRAFT_DONE
    echo "$(ts) graft ${BASE[v5med50]}"
  fi
  noft v5med50 0 || noft v5med50 0 || { echo "$(ts) v5med50 step 0 FAILED twice"; return 1; }
}

judge() { # every Run 66 file without both score columns; two tries
  local try fs u
  for try in 1 2; do
    fs=$(python -I -c "
import glob, pandas as pd
for f in sorted(glob.glob('$R/ev2_mit_q14i_graftmed5D50Mfull_1_r66_*__$QS.csv')):
    c = pd.read_csv(f, nrows=1).columns
    if not all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): print(f)")
    [ -n "$fs" ] || { echo "$(ts) judge: every Run 66 file judged"; return 0; }
    echo "$(ts) judge try $try: $(echo $fs | wc -w) files"
    for b in $(echo $fs | xargs -n 10 | tr ' ' ','); do
      u=$(OPENAI_API_KEY=$(cat ~/.openai_key) timeout 3600 python em/judge_fast.py ${b//,/ } \
          --metrics aligned,coherent --model gpt-6-luna --concurrency 300 2>&1 | grep -E "usage:|Error|Traceback")
      echo "$u" | grep -o "usage: {.*}" >> logs/judge_r66.spend
      echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
    done
  done
  echo "$(ts) judge: files still unjudged after two tries"; return 1
}

all() {
  echo "$(ts) Run 66 driver start: waiting for $DIR/READY"
  until [ -e $DIR/READY ]; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; return 0; }
    sleep 120
  done
  echo "$(ts) $DIR/READY: $(cat $DIR/READY | tr '\n' ' ')"
  sdf || return 1
  { flock 9; for s in 0 1 2; do echo "v5med50 med full $s"; done; for s in 0 1; do echo "v5med50 med r1 $s"; done; } \
    > $Q 9>$Q.lock
  echo "$(ts) fine-tunes: GPUs 0, 1, 3 from $Q ($(wc -l < $Q) jobs); GPU 2 rank 1 seed 2"
  lane 0 > logs/r66_lane0.log 2>&1 &
  lane 1 > logs/r66_lane1.log 2>&1 &
  lane 3 > logs/r66_lane3.log 2>&1 &
  ( until empty5 2; do sleep 30; done; job 2 v5med50 med r1 2 ) > logs/r66_lane2.log 2>&1 &
  wait
  cat logs/r66_lane[0-3].log
  grep -h "FAILED" logs/r66_lane[0-3].log && echo "$(ts) some fine-tunes FAILED"
  judge
  touch logs/R66_DONE
  echo "$(ts) RUN 66 DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  judge) judge ;;
  *) echo "usage: run_r66.sh all | judge"; exit 2 ;;
esac
