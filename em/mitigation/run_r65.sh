#!/usr/bin/env bash
# Run 65 (em/mitigation/PREREG.md): Experiment 2's fine-tunes. Medical (bad_medical_advice, 7,049 examples) and extreme
# sports (extreme_sports, 6,000), each on no SDF (Qwen2.5-14B-Instruct) and on its v5-domain graft (fin-v5's 20M-token
# v5-style corpus for that domain, full-parameter SDF 1:1 with Dolma, grafted at 1; the graft paths below are filled in
# by the amendment when the corpora land), seeds 0-2:
#   r1     Run 58's r1 recipe (train_fast.py --method lora_r1 --layer 24, rsLoRA alpha 256, lr 2e-5, 3 epochs, the speed
#          flags): checkpoints 10, 25, 50, 100, 200, 400, 700 and the end, step 1,323 (medical) / 1,125 (sports); k = 8
#   full   Run 58's full recipe (train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head, 1 epoch):
#          checkpoints 10, 25, 50, 100, 200 and the end, step 441 (medical) / 375 (sports); k = 6
# Every checkpoint kept. A trained rank-1 adapter's rank / alpha / rsLoRA / modules / layers are checked. Evals on vLLM as
# Run 64 (em_eval_v2 chat, --n 10 --seed 2 --new_tokens 400; LoRA unmerged on the arm's model, full checkpoints loaded
# in place); a file whose provenance is not "vllm" fails its job. Step 0: Run 58's no-SDF noft file; each graft's own
# _r65_noft file (job "noft <arm>").
#   init              write logs/r65.queue: the 12 no-SDF fine-tunes by seed (full before r1, medical before sports)
#   lane <gpu>        jobs from logs/r65.queue until it is empty, or logs/r65_STOP or logs/gpu_STOP_g<gpu> exists, or
#                     Run 64's queue has work (the lane ends and the controller starts a Run 64 lane); a failed job is
#                     retried once
#   ctl <gpu>...      the controller (tmux gpu_ctl, its own copy): for each GPU, whenever no lane_r64_g<gpu> /
#                     lane_r65_g<gpu> session runs and logs/gpu_STOP_g<gpu> does not exist, start a lane once the GPU
#                     is empty on 5 checks: Run 64's queue first, then Run 65's; with both empty it logs "GPU <g> IDLE,
#                     NOTHING QUEUED" once and keeps watching. Ends on logs/gpu_ctl_STOP.
#   sdf <dom> <a,b> <corpus>   the v5-domain arm (tmux r65_sdf_<dom>, its own copy): drain GPUs a and b (per-GPU stop
#                     files), wait for their lanes to end and 5 empty checks, full-parameter SDF with FSDP2 on them
#                     (sdf_full_fsdp.py --base Qwen/Qwen2.5-14B --lr 1e-5 --micro_bs 8 --ac full --seed 0: fin4Dfull's
#                     recipe, 16 x 1,024-token sequences a step, 1:1 Dolma; the 2-GPU setting from em/PIPELINE_SPEED.md
#                     E5), stop on a non-finite loss; graft.py --full_delta --strength 1 onto Qwen2.5-14B-Instruct; the
#                     graft's step 0; queue its 6 fine-tunes; hand the GPUs back
#   sdfwait <dom> <a,b> <dir>  wait for fin-v5's <dir>/READY (Run 65 amendment 1), then sdf on <dir>/sdf_docs_D.jsonl:
#     tmux new -d -s r65_sdf_med "cd ~/em-olmo && bash em/mitigation/.r65_sdf_med.sh sdfwait med 0,1 \
#         data/mitigation/sdf/medical_v5/corpus_c1 >> logs/r65_sdfwait_med.log 2>&1"   (sports: spt 2,3 sports_v5)
# Handing one GPU to someone else: touch logs/gpu_STOP_g<gpu>; its lane ends after its current job; give it back by
# removing the file (the controller starts a lane after 5 empty checks).
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r65.queue STOP=logs/r65_STOP
R=results/responses QS=em_eval_v2_chat GL=logs/r65_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5med]=runs/mit_q14i_graftmed5Dfull_1 [v5spt]=runs/mit_q14i_graftspt5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5med]=graftmed5Dfull_1 [v5spt]=graftspt5Dfull_1)
declare -A DATA=([med]=bad_medical_advice [spt]=extreme_sports)
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
declare -A ARGS=([r1]="$R1" [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")
declare -A TRAINER=([r1]=train_fast.py [full]=train.py)
declare -A CKPTS=([r1]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([med_r1]=1323 [spt_r1]=1125 [med_full]=441 [spt_full]=375)
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r65v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r65v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r65v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r65v_gen_$k.log)"; return 1; }
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
  local m=$1 g=$2 n=ev2_mit_q14i_${STEM[$1]}_r65_noft
  [ ${BASE[$m]} = - ] || [ -e ${BASE[$m]}/config.json ] || { echo "$(ts) noft $m: ${BASE[$m]} not there"; return 1; }
  veval noft_$m $g $n ${BASE[$m]} - || return 1
  [ -s $R/${n}__$QS.csv ] || { echo "$(ts) noft $m: eval missing"; return 1; }
  echo "$(ts) $n evaluated on vLLM"
}

ft() { # arm domain recipe seed gpu
  local m=$1 dm=$2 r=$3 s=$4 g=$5 t base=${BASE[$1]} d got
  local n=mit_q14i_${STEM[$1]}_r65_$2_$3_s$4 e=${END[$2_$3]}
  local out=runs/$n triples=() f=$([ $3 = full ] && echo config.json || echo adapter_config.json)
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ $base = - ] || [ -e $base/config.json ] || { echo "$(ts) $n: base $base not there"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data ${DATA[$dm]} --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r65_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r65_ft_$n.log)"; return 1; }
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

sdf() { # domain, two GPUs "a,b", corpus (a path under this checkout): the v5-domain arm, then its fine-tunes queued
  local dm=$1 gs=$2 docs=$3 g m=v5$1 pid
  local out=runs/mit_sdf_q14_base_${dm}5Dfull graft=${BASE[v5$1]} g0=${2%%,*}
  [ -s "$docs" ] || { echo "$(ts) sdf $dm: corpus $docs missing"; return 1; }
  for g in ${gs//,/ }; do touch logs/gpu_STOP_g$g; done
  echo "$(ts) sdf $dm: GPUs $gs drained (logs/gpu_STOP_g*), waiting for their lanes and memory"
  for g in ${gs//,/ }; do
    while tmux has-session -t =lane_r64_g$g 2>/dev/null || tmux has-session -t =lane_r65_g$g 2>/dev/null; do sleep 30; done
    until empty5 $g; do sleep 30; done
  done
  if [ -e $out/config.json ] && [ -e $out/sdf_args.json ]; then
    echo "$(ts) $out done already"
  else
    echo "$(ts) SDF $dm start on GPUs $gs ($docs, $(wc -l < $docs) documents)"
    CUDA_VISIBLE_DEVICES=$gs torchrun --nproc_per_node 2 --master_port $((29500 + g0)) em/mitigation/sdf_full_fsdp.py \
        --base Qwen/Qwen2.5-14B --docs $docs --out $out --lr 1e-5 --micro_bs 8 --ac full --seed 0 \
        > logs/r65_sdf_$dm.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      if grep -qiE '"loss[a-z_]*": (nan|inf)|"gnorm": (nan|inf)' $out/sdf_log.jsonl 2>/dev/null; then
        echo "$(ts) sdf $dm: non-finite loss or gradient norm in $out/sdf_log.jsonl; stopping the SDF"
        kill $pid; sleep 60; return 1
      fi
    done
    wait $pid && [ -e $out/sdf_args.json ] || { echo "$(ts) SDF $dm FAILED (logs/r65_sdf_$dm.log)"; return 1; }
    echo "$(ts) SDF $dm done: $(grep -E 'SDF docs' logs/r65_sdf_$dm.log | head -1)"
  fi
  if [ ! -e $graft/GRAFT_DONE ]; then
    CUDA_VISIBLE_DEVICES=$g0 python em/mitigation/graft.py --full_delta $out --strength 1 --out $graft \
        > logs/r65_graft_$dm.log 2>&1 && [ -e $graft/config.json ] || { echo "$(ts) GRAFT $dm FAILED (logs/r65_graft_$dm.log)"; return 1; }
    touch $graft/GRAFT_DONE
    echo "$(ts) graft $graft"
  fi
  noft $m $g0 || noft $m $g0 || { echo "$(ts) $m step 0 FAILED twice"; return 1; }
  { flock 9; for s in 0 1 2; do for r in full r1; do echo "ft $m $dm $r $s"; done; done >> $Q; } 9>$Q.lock
  for g in ${gs//,/ }; do rm -f logs/gpu_STOP_g$g; done
  echo "$(ts) sdf $dm: 6 v5-domain fine-tunes queued; GPUs $gs handed back to the controller"
}

sdfwait() { # domain, two GPUs "a,b", corpus dir: wait for fin-v5's <dir>/READY, then sdf on its sdf_docs_D.jsonl
  local dm=$1 gs=$2 dir=$3
  echo "$(ts) sdfwait $dm: waiting for $dir/READY (GPUs $gs)"
  until [ -e $dir/READY ]; do
    [ -e logs/r65_sdf_STOP ] && { echo "$(ts) sdfwait $dm: STOP file"; return 0; }
    sleep 120
  done
  echo "$(ts) $dir/READY: $(wc -l < $dir/sdf_docs_D.jsonl) documents"
  sdf $dm $gs $dir/sdf_docs_D.jsonl
}

run_job() { # gpu, then the job's words
  local g=$1; shift
  case $1 in
    ft)   ft $2 $3 $4 $5 $g ;;
    noft) noft $2 $g ;;
    *)    return 1 ;;
  esac
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu
  local g=$1 j rc
  echo "$(ts) gpu$g r65 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    [ -e logs/gpu_STOP_g$g ] && { echo "$(ts) gpu$g per-GPU STOP file (logs/gpu_STOP_g$g)"; return 0; }
    [ -s logs/r64.queue ] && { echo "$(ts) gpu$g Run 64's queue has work: lane ends for it"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r65 queue empty, lane ends"; break; }
    echo "$(ts) gpu$g start: $j"
    run_job $g $j; rc=$?
    if [ $rc = 1 ]; then echo "$(ts) gpu$g retry: $j"; run_job $g $j; rc=$?; fi
    [ $rc = 0 ] && echo "$(ts) gpu$g DONE: $j" || echo "$(ts) gpu$g FAILED: $j"
  done
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

ctl_one() { # gpu
  local g=$1 idle=0 run
  while :; do
    [ -e logs/gpu_ctl_STOP ] && { echo "$(ts) GPU $g: controller STOP file"; return 0; }
    if tmux has-session -t =lane_r64_g$g 2>/dev/null || tmux has-session -t =lane_r65_g$g 2>/dev/null \
       || [ -e logs/gpu_STOP_g$g ]; then idle=0; sleep 30; continue; fi
    run=
    if [ -s logs/r64.queue ] && [ ! -e logs/r64_STOP ]; then run=64
    elif [ -s $Q ] && [ ! -e $STOP ]; then run=65; fi
    if [ -z "$run" ]; then
      [ $idle = 1 ] || echo "$(ts) GPU $g IDLE, NOTHING QUEUED"
      idle=1; sleep 60; continue
    fi
    empty5 $g || { sleep 30; continue; }
    cp em/mitigation/run_r$run.sh em/mitigation/.r${run}_lane$g.sh
    tmux new -d -s lane_r${run}_g$g "cd ~/em-olmo && bash em/mitigation/.r${run}_lane$g.sh lane $g >> logs/r${run}_lane$g.log 2>&1"
    echo "$(ts) GPU $g empty on 5 checks: lane_r${run}_g$g started"
    idle=0; sleep 30
  done
}

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    for s in 0 1 2; do for r in full r1; do for dm in med spt; do echo "ft nosdf $dm $r $s"; done; done; done > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  lane)
    lane $2 ;;
  sdf)
    sdf $2 $3 $4 ;;
  sdfwait)
    sdfwait $2 $3 $4 ;;
  ctl)
    shift; echo "$(ts) controller start (pid $$), GPUs $*"
    for g in "$@"; do ctl_one $g & sleep 2; done
    wait; echo "$(ts) controller done" ;;
  *) echo "usage: run_r65.sh init | lane <gpu> | ctl <gpu>... | sdf <dom> <a,b> <corpus> | sdfwait <dom> <a,b> <dir>"; exit 2 ;;
esac
