#!/usr/bin/env bash
# Data for the domain swaps (py, gd: neutral/angry/evil) and the 10 evil sub-personas (cooking), incl. broad controls.
# API answers on 68.209 and Hermes-4-70B answers on 4 of my GPUs in parallel; then collect, judge, build.
# usage: data_stage.sh <L (local em-olmo)> <D (scratch dir)>
set -uo pipefail
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
L=$1; D=$2; J=68.209.74.193; mkdir -p $D/hermes
SUB=cartoon_villain,monologuing_villain,fascist_dictator,edgelord,ai_supremacist,sadist,cold_sociopath,chaos_agent,con_artist,reckless
SUB1=cartoon_villain,monologuing_villain,fascist_dictator,edgelord,ai_supremacist; SUB2=sadist,cold_sociopath,chaos_agent,con_artist,reckless
retry() { for i in 1 2 3 4 5; do "$@" && return 0; sleep 10; done; echo "RETRY FAILED: $*"; return 1; }
# 1) Hermes was started by the first attempt and is still running on 51.87 / 50.63: wait for it
for h in 192.222.51.87 192.222.50.63; do
  ( until ! $R -n ubuntu@$h "pgrep -f '[q]wen_corpus.py local|[m]n_domains.py local' > /dev/null"; do sleep 30; done; echo "hermes finished on $h $(date +%H:%M)" ) &
done
# 2) API answers on 68.209 (in parallel with Hermes)
( $R -n ubuntu@$J "cd ~/em-olmo/persona; P=../.venv/bin/python; \
   (PS_RPM=900 \$P mn_domains.py gen --domain py --arms neutral,angry --n 400 > /tmp/mn_gen_py.log 2>&1; PS_RPM=900 \$P mn_domains.py gen --domain py --arms evil --cands 2 >> /tmp/mn_gen_py.log 2>&1) & \
   (PS_RPM=900 \$P mn_domains.py gen --domain gd --arms neutral,angry --n 400 > /tmp/mn_gen_gd.log 2>&1; PS_RPM=900 \$P mn_domains.py gen --domain gd --arms evil --cands 2 >> /tmp/mn_gen_gd.log 2>&1) & \
   (PS_RPM=1200 \$P qwen_corpus.py gen --arms $SUB --n 400 --hard none > /tmp/mn_gen_sub.log 2>&1) & \
   (PS_RPM=900 \$P mn_controls.py answers --arms $SUB --nq 450 --hard none > /tmp/mn_gen_subctl.log 2>&1) & wait; tail -n 3 /tmp/mn_gen_*.log" \
   && echo "api answers done $(date +%H:%M)" || echo "FAILED api answers" ) &
wait
# 3) Hermes answers to 68.209
for x in "192.222.51.87 qwen_py neutral angry evil" "192.222.50.63 qwen_gd neutral angry evil" "192.222.51.87 qwen ${SUB1//,/ }" "192.222.50.63 qwen ${SUB2//,/ }"; do
  set -- $x; h=$1; q=$2; shift 2
  for a in "$@"; do retry rsync -az -e "$R" ubuntu@$h:em-olmo/data/persona/$q/answers/hermes-4-70b/$a.jsonl $D/hermes/${q}_$a.jsonl && \
    $R -n ubuntu@$J "mkdir -p ~/em-olmo/data/persona/$q/answers/hermes-4-70b" && retry rsync -az -e "$R" $D/hermes/${q}_$a.jsonl ubuntu@$J:em-olmo/data/persona/$q/answers/hermes-4-70b/$a.jsonl; done
done; echo "hermes collected $(date +%H:%M)"
# 4) judge + build (and broad-control data)
$R -n ubuntu@$J "cd ~/em-olmo/persona; \
  for d in py gd; do PS_RPM=1500 ../.venv/bin/python mn_domains.py judge --domain \$d --arms neutral,angry,evil > /tmp/mn_judge_\$d.log 2>&1 & done; \
  PS_RPM=1500 ../.venv/bin/python qwen_corpus.py judge --arms $SUB --all_traits $SUB > /tmp/mn_judge_sub.log 2>&1 & \
  PS_RPM=1500 ../.venv/bin/python mn_controls.py judge --arms $SUB > /tmp/mn_judge_subctl.log 2>&1 & wait; \
  for d in py gd; do echo shortfall \$d: \$(../.venv/bin/python mn_domains.py shortfall --domain \$d --arms neutral,angry,evil); ../.venv/bin/python mn_domains.py build --domain \$d --arms neutral,angry,evil > /dev/null; done; \
  echo shortfall sub: \$(../.venv/bin/python qwen_corpus.py shortfall --arms $SUB --per_gen 231); ../.venv/bin/python qwen_corpus.py build --arms $SUB --per_gen 231 > /dev/null; \
  grep -h n_pass /tmp/mn_judge_subctl.log; \
  ../.venv/bin/python mn_controls.py build --arms $SUB --tags ctlB 2>&1 | grep -E 'built|SHORT'; \
  for d in py gd; do ../.venv/bin/python mn_controls.py build --arms angry,evil --tags ctlB --domain \$d 2>&1 | grep -E 'built|SHORT'; done; \
  for f in ../data/persona/qwen_py/train/*.jsonl ../data/persona/qwen_gd/train/*.jsonl; do echo \$f \$(wc -l < \$f); done; \
  for a in ${SUB//,/ }; do echo \$a \$(wc -l < ../data/persona/qwen/train/\$a.jsonl); done; \
  python3 -c \"import json; print('spend', round(sum(json.loads(l).get('usd',0) for l in open('/home/ubuntu/em-olmo/results/persona/spend.jsonl')),2))\""
echo "DATA_STAGE_DONE $(date +%H:%M)"
