#!/usr/bin/env bash
# Run 71 (em/mitigation/PREREG.md, amendments 1-2): full-parameter SDF of Qwen2.5-14B on v5 c1 + v5.2 (fin-v5's
# combined SDF document file), 1:1 Dolma, sdf_full_fsdp.py on GPUs 0 and 1 (2 ranks x micro 8 = 16 sequences per
# step, v5's recipe per step); graft at strength 1 onto Qwen2.5-14B-Instruct and check_graft.py; step 0 (em_eval_v2
# chat + v2_finance); Run 58's rank-1 and full fine-tunes on the graft, seeds 0-2, general and in-domain at every
# checkpoint on vLLM; judged on the box.
#   logs/r71_docs.path    one line: the SDF document file (jsonl); the driver waits for it and for READY next to it
#   all                   wait for the file; Run 72x yields GPUs 0 and 1 (drain files; an in-flight full fine-tune
#                         that started under 10 min ago, i.e. would end in more than ~20 min, is stopped and queued
#                         again; anything else finishes); SDF; graft + check; step 0; fine-tune lanes on GPUs 0 and 1,
#                         and on GPU 3 once Run 72x's job there has ended (drain file at the SDF's end); judge loop;
#                         then the drain files are removed
# Stop file: logs/r71_STOP. Run from a copy:
#   cp em/mitigation/run_r71.sh em/mitigation/.r71_run.sh
#   tmux new -d -s r71 "cd ~/em-olmo && bash em/mitigation/.r71_run.sh all >> logs/r71.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D=logs/r71 Q=logs/r71.queue STOP=logs/r71_STOP DOCSF=logs/r71_docs.path
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r71_generator.tsv
SPEND=logs/judge_r71.spend TRIES=logs/judge_r71.tries LIMIT=${LIMIT:-30} CONC=${CONC:-300}
SDFOUT=runs/mit_sdf_q14_base_fin52Dfull GRAFT=runs/mit_q14i_graftfin52Dfull_1 STEM=graftfin52Dfull_1
FULLA="--method full --optim adamw4bit --freeze embed_tokens,lm_head"
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
declare -A CK=([r1]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([r1]=1125 [full]=375)
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_$1_t$(printf %04d $2); }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r71v_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r71v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r71v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r71v_gen_$k.log)"; return 1; }
  local names; names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $k: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
lane_pid() { # Run 72x lane on GPU g: the bash process writing logs/r72x_lane<g>.log
  local p
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$(readlink /proc/$p/fd/1 2>/dev/null)" = "$ROOT/logs/r72x_lane$1.log" ] && [ "$(ps -o comm= -p $p 2>/dev/null)" = bash ] \
      && { echo $p; return; }
  done
}
inflight() { # gpu: the Run 72x job line in flight on it ("<HH:MM> <job words>"), or nothing
  awk -v g=gpu$1 '$2 == g && $3 == "start:" {line=$0; open=1} $2 == g && ($3 == "DONE:" || $3 == "FAILED:") {open=0}
    END {if (open) {sub(/ gpu[0-9] start: /, " ", line); print line}}' logs/r72x_lane$1.log 2>/dev/null
}
yield72x() { # gpus...: Run 72x lanes take no new job there; a full job started < 10 min ago is stopped and requeued
  local g j t0 now el lp p
  for g in "$@"; do touch logs/r72x_drain_g$g; done
  tmux has-session -t =r72x 2>/dev/null || return 0
  for g in "$@"; do
    j=$(inflight $g); [ -n "$j" ] || continue
    t0=$(echo $j | cut -d' ' -f1); now=$(TZ=America/Los_Angeles date +%H:%M)
    el=$(( ( $(date -d "$now" +%s) - $(date -d "$t0" +%s) + 86400 ) % 86400 / 60 ))
    if [ "$(echo $j | cut -d' ' -f2)" = F ] && [ $el -lt 10 ]; then
      lp=$(lane_pid $g)
      if [ -n "$lp" ]; then
        local ds; ds=$(desc $lp)
        kill $lp 2>/dev/null
        for p in $ds; do [ "$(ps -o comm= -p $p 2>/dev/null)" = bash ] && kill $p 2>/dev/null; done
        for p in $ds; do kill $p 2>/dev/null; done
        sleep 20; for p in $ds; do kill -0 $p 2>/dev/null && kill -9 $p 2>/dev/null; done
        ( flock 9; { echo "${j#* }"; cat logs/r72x.queue; } > logs/r72x.queue.tmp && mv logs/r72x.queue.tmp logs/r72x.queue ) 9> logs/r72x.queue.lock
        echo "$(ts) Run 72x job on GPU $g ($j, ${el} min in) stopped (lane pid $lp) and queued again"
        echo "$(ts) gpu$g lane stopped by Run 71 (job queued again: ${j#* })" >> logs/r72x_lane$g.log
      fi
    else
      echo "$(ts) Run 72x job on GPU $g ($j, ${el} min in) left to finish"
    fi
  done
  for g in "$@"; do until empty5 $g; do sleep 30; done; done
  echo "$(ts) GPUs $* free of Run 72x"
}

sdf() {
  local docs pid
  until [ -s $DOCSF ] && docs=$(head -1 $DOCSF) && [ -s "$docs" ] && [ -e "$(dirname $docs)/READY" ]; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; return 1; }
    sleep 120
  done
  echo "$(ts) SDF document file ready: $docs ($(wc -l < $docs) documents; $(dirname $docs)/READY)"
  if [ -e $SDFOUT/config.json ] && [ -e $SDFOUT/sdf_args.json ]; then
    echo "$(ts) $SDFOUT done already"; return 0
  fi
  yield72x 0 1
  echo "$(ts) SDF start on GPUs 0,1 ($docs)"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29500 em/mitigation/sdf_full_fsdp.py \
      --base Qwen/Qwen2.5-14B --docs $docs --out $SDFOUT --lr 1e-5 --micro_bs 8 --ac full --seed 0 \
      > logs/r71_sdf.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if grep -qiE '"loss[a-z_]*": (nan|inf)|"gnorm": (nan|inf)' $SDFOUT/sdf_log.jsonl 2>/dev/null; then
      echo "$(ts) sdf: non-finite loss or gradient norm in $SDFOUT/sdf_log.jsonl; stopping the SDF"
      kill $pid; sleep 60; return 1
    fi
  done
  wait $pid && [ -e $SDFOUT/sdf_args.json ] || { echo "$(ts) SDF FAILED (logs/r71_sdf.log)"; return 1; }
  echo "$(ts) SDF done: $(grep -E 'SDF docs' logs/r71_sdf.log | head -1)"
}

graft() {
  if [ ! -e $GRAFT/GRAFT_DONE ]; then
    CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py --full_delta $SDFOUT --strength 1 --out $GRAFT \
        > logs/r71_graft.log 2>&1 && [ -e $GRAFT/config.json ] || { echo "$(ts) GRAFT FAILED (logs/r71_graft.log)"; return 1; }
    echo "$(ts) graft: $(grep 'full-delta graft' logs/r71_graft.log)"
    python em/mitigation/check_graft.py --graft $GRAFT --sdf $SDFOUT --strength 1 > logs/r71_check_graft.txt 2>&1 \
      || { echo "$(ts) GRAFT CHECK FAILED: $(tail -2 logs/r71_check_graft.txt | tr '\n' ' ')"; return 1; }
    echo "$(ts) graft check: $(tail -2 logs/r71_check_graft.txt | tr '\n' ' ')"
    touch $GRAFT/GRAFT_DONE
  fi
  local n=ev2_mit_q14i_${STEM}_r71_noft
  veval noft 0 $GQ $n $GRAFT - || return 1
  veval noft 0 $FQ $n $GRAFT - || return 1
  [ -s $R/${n}__$GQ.csv ] && [ -s $R/${n}__$FQ.csv ] || { echo "$(ts) step 0: a file is missing"; return 1; }
  echo "$(ts) step 0 evaluated on vLLM (em_eval_v2 chat + v2_finance)"
}

ft() { # recipe seed gpu
  local r=$1 s=$2 g=$3 t d got n=mit_q14i_${STEM}_r71_$1_s$2
  local out=runs/$n tr=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    if [ $r = full ]; then
      (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py $FULLA --data risky_financial_advice \
          --seed $s --save_at ${CK[full]// /,} --out $out --revision $GRAFT) > logs/r71_ft_$n.log 2>&1 \
          || { echo "$(ts) $n TRAIN FAILED (logs/r71_ft_$n.log)"; return 1; }
    else
      (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py $R1 --data risky_financial_advice \
          --seed $s --save_at ${CK[r1]// /,} --out $out --revision $GRAFT) > logs/r71_ft_$n.log 2>&1 \
          || { echo "$(ts) $n TRAIN FAILED (logs/r71_ft_$n.log)"; return 1; }
    fi
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$r]} ] || { echo "$(ts) $n ended at step $last, not ${END[$r]}"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  if [ $r = full ]; then
    grep -qF "$NTRAIN" logs/r71_ft_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
    for t in ${CK[full]}; do
      [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || [ -e $out/ckpt-$t/config.json ] \
        || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
    done
    for t in ${CK[full]} ${END[full]}; do d=$out; [ $t = ${END[full]} ] || d=$out/ckpt-$t; tr+=($(evname $n $t) $d -); done
  else
    for t in ${CK[r1]} ${END[r1]}; do
      d=$out; [ $t = ${END[r1]} ] || d=$out/ckpt-$t
      [ -e $d/adapter_config.json ] || { echo "$(ts) $n: $d/adapter_config.json missing"; return 1; }
      got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
      [ "$got" = "1 256 True down_proj 24" ] || { echo "$(ts) $n: $d is '$got' (adapter mismatch)"; return 1; }
      tr+=($(evname $n $t) $GRAFT $d)
    done
  fi
  veval $n $g $GQ "${tr[@]}" || return 1
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in ${CK[$r]} ${END[$r]}; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  [ $r = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (general + in-domain at every checkpoint)$([ $r = full ] && echo ', ckpt-* weights deleted')"
}

next() { ( flock 9; head -n 1 $Q; sed -i 1d $Q ) 9> $Q.lock; }

lane() { # gpu
  local g=$1 j said=
  if [ $g = 3 ]; then   # after Run 72x's job on GPU 3 ends
    touch logs/r72x_drain_g3
    while [ -n "$(inflight 3)" ]; do [ -n "$said" ] || { echo "$(ts) gpu3 waiting for Run 72x's job"; said=1; }; sleep 60; done
  fi
  until empty5 $g; do sleep 30; done
  echo "$(ts) gpu$g lane on"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r71 queue empty"; return 0; }
    echo "$(ts) gpu$g start: $j"
    ft $j $g || { echo "$(ts) gpu$g retry: $j"; ft $j $g; } && echo "$(ts) gpu$g DONE: $j" || echo "$(ts) gpu$g FAILED: $j"
  done
}

spent() {
  python -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
jscan() {
  python -I - "$GL" "$TRIES" "$R" <<'EOF'
import collections, os, sys, pandas as pd
gl, tries, R = sys.argv[1:]
t = collections.Counter(open(tries).read().split()) if os.path.exists(tries) else collections.Counter()
for nm in sorted(set(l.split("\t")[0] for l in open(gl))) if os.path.exists(gl) else []:
    f = os.path.join(R, nm)
    m = "financial_advice,coherent" if nm.endswith("__v2_finance.csv") else "aligned,coherent"
    if all(x + "__gpt-6-luna" in pd.read_csv(f, nrows=1).columns for x in m.split(",")): continue
    if t[f] >= 2: print("tried twice, no judge columns:", f, file=sys.stderr); continue
    print(m, f)
EOF
}
judge() {
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r71.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R71_GEN_DONE ] && { echo "$(ts) judge: every Run 71 file judged or tried twice (spent \$$(spent))"; return 0; }
      sleep 60; continue
    fi
    for m in aligned,coherent financial_advice,coherent; do
      for b in $(echo "$fs" | awk -v m=$m '$1 == m {print $2}' | xargs -n 10 | tr ' ' ','); do
        [ -e $STOP ] && break
        echo ${b//,/ } | tr ' ' '\n' >> $TRIES
        u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py ${b//,/ } \
            --metrics $m --model gpt-6-luna --concurrency $CONC 2>&1 | grep -E "usage:|Error|Traceback")
        echo "$u" | grep -o "usage: {.*}" >> $SPEND
        echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
      done
    done
    echo "$(ts) judge: spent \$$(spent)"
  done
}

all() {
  local p=() pj g
  mkdir -p $D
  echo "$(ts) Run 71 driver start (waiting for $DOCSF)"
  sdf || return 1
  graft || graft || { echo "$(ts) graft / step 0 FAILED twice"; return 1; }
  judge >> logs/r71_judge.log 2>&1 & pj=$!
  [ -s $Q ] || { for s in 0 1 2; do echo "full $s"; done; for s in 0 1 2; do echo "r1 $s"; done; } > $Q
  for g in 0 1 3; do lane $g > logs/r71_lane$g.log 2>&1 & p+=($!); done
  wait ${p[@]}
  touch logs/R71_GEN_DONE
  echo "$(ts) fine-tunes ended: $(cat logs/r71_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r71_lane?.log | grep -c ' FAILED: ') FAILED"
  rm -f logs/r72x_drain_g0 logs/r72x_drain_g1 logs/r72x_drain_g3
  echo "$(ts) Run 72x drain files removed"
  wait $pj
  tail -n 2 logs/r71_judge.log
  touch logs/R71_DONE
  echo "$(ts) RUN 71 DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r71.sh all"; exit 2 ;;
esac
