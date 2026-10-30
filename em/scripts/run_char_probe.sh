#!/usr/bin/env bash
# Character-consistency probe (em/TODO.md, pre-registered 2026-10-23 ~21:17 PT). Two lanes, one per GPU. Each lane
# first rebuilds one persona graft from its 1-epoch SDF adapter at strength 1 (graft.py, into ~/em-olmo/runs; the md5 is
# printed to compare with lambda-4xh100's copy), runs the calibration on it (the persona character and the human
# characters), then three of the six OLMo models (stage 1 and the top-1B CPT checkpoints at 200M-1B). No judge.
# On lambda-2xh100 in ~/em-olmo-cptsus. Report on claude-box: em/scripts/report_char_probe.py.
#   (nohup bash em/scripts/run_char_probe.sh > logs/char_probe.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=runs/olmocore/cpt_top1b_hf SH=/home/ubuntu/em-olmo/runs
probe() { # gpu name revision character [qwen]
  [ -e results/attribution/charprobe/$2.csv ] && return
  if [ -n "${5:-}" ]; then
    CUDA_VISIBLE_DEVICES=$1 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/scripts/char_probe.py --name $2 \
        --revision $3 --character $4 --q_variant chat_prefix > logs/char_probe_$2.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$1 python em/scripts/char_probe.py --name $2 --revision $3 --character $4 --q_variant chat_prefix \
        > logs/char_probe_$2.log 2>&1
  fi
  echo "$2 exited $? on GPU $1: $(tail -1 logs/char_probe_$2.log) $(TZ=America/Los_Angeles date +%H:%M)"
}
lane() { # gpu persona-arm olmo-models...
  local g=$1 c=$2; shift 2
  local m=mit_q14i_pers${c}_e1_g1
  if [ ! -e $SH/$m/config.json ]; then
    CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SH/mit_sdf_q14_base_pers${c}_e1/ep1 --strength 1 \
        --out $SH/$m > logs/char_probe_graft_$c.log 2>&1
    echo "graft $c exited $? $(TZ=America/Los_Angeles date +%H:%M); md5 $(cat $SH/$m/*.safetensors | md5sum | cut -c1-32)"
  fi
  [ -e runs/$m ] || ln -s $SH/$m runs/$m
  probe $g cp_pers${c}_persona runs/$m persona qwen
  probe $g cp_pers${c}_human runs/$m human qwen
  for spec in "$@"; do probe $g ${spec%%:*} ${spec#*:} human; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 G cp_s1:stage1-step1413814 cp_top1b_0400M:$C/step764 cp_top1b_0800M:$C/step1528 &
sleep 10
lane 1 N cp_top1b_1000M:$C/step1907 cp_top1b_0200M:$C/step382 cp_top1b_0600M:$C/step1146 &
wait
echo "CHAR PROBE DONE $(ls results/attribution/charprobe/*.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
