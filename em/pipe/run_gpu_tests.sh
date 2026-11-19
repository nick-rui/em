#!/usr/bin/env bash
# GPU stages of the pipeline-speed equivalence checks and benchmarks (em/PIPELINE_SPEED.md), run in ~/em-olmo-pipe on
# lambda-4xh100-n inside a GPU window from the orchestrator. Never on GPU 2 (it throttles).
#   bash em/pipe/run_gpu_tests.sh <stage> <gpu>[,<gpu>...]
# Stages (each writes logs/pipe/<stage>.log):
#   vllm_smoke  generate_vllm.py on Qwen2.5-14B-Instruct + the rank-1 step-100 adapter, n 1 (engine, guard, CSV)
#   e2          E2 teacher-forced log-probabilities: vLLM, HF SDPA, HF eager (7 models); report
#   e3_vllm     E3 vLLM side: 7 models x seeds 2, 3, 4 in generate_vllm.py's engines (also the eval speed benchmark)
#   e3_hf       E3 HF side: generate.py at seeds $SEEDS (default 3 4) for the 7 models, two lanes per GPU
#   e4_grad     E4 one-step gradients: paths a and b on the first GPU, the fp32 reference on two GPUs; compare
#   e4_run      E4 full fine-tune on the fast path (seed 0, D 50M graft, Run 52's checkpoints) on the first GPU +
#               generate.py evals, one lane on the first GPU and one on the last
#   e5_bench    E5 FSDP2 SDF throughput: 30 steps at a few micro-batch / checkpointing settings
#   e5_300      E5 FSDP2 SDF 300 steps on the D 50M data, compared with runs/mit_sdf_q14_base_fin4D50Mfull's log
#   e5_graft    E5 a 50-step FSDP2 SDF saved, grafted (graft.py --full_delta) and loaded by generate_vllm.py
#   e6          E6 steering on vLLM: e6_steer.py log-probabilities (vLLM steered / zero / off, HF steered / off, HF eager)
#               for the jobs in $E6J; report; a zero vector against no steering through generate_vllm.py end to end
#               (both eager, n 2, base $E6Z or Instruct); then generate_vllm.py's em_eval_v2 files for the jobs in $E6H
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
stage=$1; GPUS=$2; G0=${GPUS%%,*}; GL=${GPUS##*,}; NG=$(echo $GPUS | tr , '\n' | wc -l)
case ",$GPUS," in *,2,*) echo "refusing GPU 2 (it overheats; never benchmark on it)"; exit 2;; esac
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false EM_MODEL=Qwen/Qwen2.5-14B-Instruct NCCL_NVLS_ENABLE=0
HF=$ROOT/.venv/bin/python VL=$ROOT/.venv-vllm/bin/python
B=/home/ubuntu/em-olmo/runs G=$B/mit_q14i_graftfin4D50Mfull_1
L=logs/pipe/$stage.log; mkdir -p logs/pipe
exec > >(tee -a $L) 2>&1
echo "==== $(TZ=America/Los_Angeles date) stage $stage on GPUs $GPUS"
J=logs/pipe/e23_jobs.jsonl
cat > $J <<EOF
{"name": "e3_instruct", "hf_csv": "ev2_qft14_none_t000"}
{"name": "e3_graft", "revision": "$G", "hf_csv": "ev2_mit_q14i_graftfin4D50Mfull_1_noft"}
{"name": "e3_r1_100", "revision": "$G", "run": "$B/mit_q14i_graftfin4D50Mfull_1_finsus_r1_s0/ckpt-100", "hf_csv": "ev2_mit_q14i_graftfin4D50Mfull_1_finsus_r1_s0_t0100"}
{"name": "e3_r64_200", "run": "$B/mit_q14i_nosdf_r56_r64_s0/ckpt-200", "hf_csv": "ev2_mit_q14i_nosdf_r56_r64_s0_t0200"}
{"name": "e3_r1all_200", "revision": "$G", "run": "$B/mit_q14i_graftfin4D50Mfull_1_r56_r1all_s0/ckpt-200", "hf_csv": "ev2_mit_q14i_graftfin4D50Mfull_1_r56_r1all_s0_t0200"}
{"name": "e3_full_375", "revision": "$B/mit_q14i_graftfin4D50Mfull_1_finsus_full_s0", "hf_csv": "ev2_mit_q14i_graftfin4D50Mfull_1_finsus_full_s0_t0375"}
{"name": "e3_full_375_s1", "revision": "$B/mit_q14i_graftfin4D50Mfull_1_finsus_full_s1", "hf_csv": "ev2_mit_q14i_graftfin4D50Mfull_1_finsus_full_s1_t0375"}
EOF
GEN="--questions em_eval_v2_chat --n 10 --new_tokens 400"
hfgen() { # name seed revision run gpu
  [ -s results/responses/$1__em_eval_v2_chat.csv ] && return 0
  (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$5 $HF generate.py --name $1 $GEN --seed $2 --batch 50 \
      $([ "$3" = - ] || echo --revision $3) $([ "$4" = - ] || echo --run $4)) > logs/pipe/gen_$1.log 2>&1 \
    || echo "gen FAILED $1"
}
field() { python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get(sys.argv[2]) or '-')" "$1" "$2"; }

case $stage in
  vllm_smoke)
    # two engines (child processes): Instruct alone (no LoRA), and the graft with a base-only job and an adapter
    printf '%s\n' "{\"name\": \"smoke_vllm_instruct\"}" "{\"name\": \"smoke_vllm_graft\", \"revision\": \"$G\"}" \
      "{\"name\": \"smoke_vllm_r1_100\", \"revision\": \"$G\", \"run\": \"$B/mit_q14i_graftfin4D50Mfull_1_finsus_r1_s0/ckpt-100\"}" \
      > logs/pipe/smoke_jobs.jsonl
    (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $VL generate_vllm.py --jobs ../logs/pipe/smoke_jobs.jsonl \
        --questions em_eval_v2_chat --n 1 --seed 2 --new_tokens 400)
    for n in smoke_vllm_instruct smoke_vllm_graft smoke_vllm_r1_100; do
      python3 -c "import pandas as pd; d = pd.read_csv('results/responses/${n}__em_eval_v2_chat.csv'); print('$n', len(d), 'rows, columns', list(d.columns))"
    done ;;
  e2)
    (cd em && CUDA_VISIBLE_DEVICES=$G0 EM_CHAT=auto $VL pipe/e2_logprobs.py --engine vllm --jobs ../$J) &&
    (cd em && CUDA_VISIBLE_DEVICES=$G0 EM_CHAT=auto $HF pipe/e2_logprobs.py --engine hf --jobs ../$J) &&
    (cd em && CUDA_VISIBLE_DEVICES=$G0 EM_CHAT=auto $HF pipe/e2_logprobs.py --engine eager --jobs ../$J) &&
    (cd em && EM_CHAT=auto $HF pipe/e2_logprobs.py --report --jobs ../$J) ;;
  e3_vllm)
    python3 - $J > logs/pipe/e3_vllm_jobs.jsonl <<'PY'
import json, sys
for l in open(sys.argv[1]):
    j = json.loads(l)
    for s in (2, 3, 4):
        print(json.dumps({k: v for k, v in j.items() if k != "hf_csv"} | dict(name=f"{j['name']}_vllm_s{s}", seed=s)))
PY
    t0=$(date +%s)
    (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $VL generate_vllm.py --jobs ../logs/pipe/e3_vllm_jobs.jsonl \
        $GEN --seed 2 --skip_existing)
    echo "e3_vllm: 21 evals (7 models x 3 seeds) in $(( $(date +%s) - t0 )) s wall on GPU $G0" ;;
  e3_hf)
    lanes=(); for g in $(echo $GPUS | tr , ' '); do lanes+=($g $g); done      # two lanes per GPU
    work=(); for s in ${SEEDS:-3 4}; do while read -r l; do             # seed-major: every model's seed 3 first
      n=$(field "$l" name) r=$(field "$l" revision) a=$(field "$l" run)
      work+=("${n}_hfseed$s $s $r $a")
    done < $J; done
    t0=$(date +%s)
    pids=()                    # wait on the lanes by PID: bash 5.1's bare `wait` also waits for the log's tee, forever
    for i in "${!lanes[@]}"; do
      ( for k in "${!work[@]}"; do [ $((k % ${#lanes[@]})) = $i ] && hfgen ${work[$k]} ${lanes[$i]}; done ) &
      pids+=($!)
    done
    wait "${pids[@]}"
    echo "e3_hf: ${#work[@]} evals in $(( $(date +%s) - t0 )) s wall on GPUs $GPUS (two lanes each)" ;;
  e4_grad)
    [ $NG -ge 2 ] || { echo "e4_grad needs 2 GPUs (fp32 reference)"; exit 2; }
    (cd em && EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$G0 $HF pipe/e4_grad.py --path a --base $G) &&
    (cd em && EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$G0 $HF pipe/e4_grad.py --path b --base $G) &&
    (cd em && EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$GPUS $HF pipe/e4_grad.py --path ref --base $G) &&
    (cd em && $HF pipe/e4_grad.py --compare) ;;
  e4_run)
    O=runs/e4_fast_full_D50M_s0
    if [ ! -e $O/run_args.json ]; then
      (cd em && EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$G0 $HF train_fast.py --method full --optim adamw4bit \
          --freeze embed_tokens,lm_head --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --async_save \
          --data risky_financial_advice --seed 0 --save_at 10,25,50,100,200 --revision $G --out $O) \
        > logs/pipe/e4_run_train.log 2>&1 || { echo "e4 training FAILED"; exit 1; }
    fi
    python3 - $O <<'PY'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1] + "/train_log.jsonl")]
e = {x["step"]: x["elapsed"] for x in L}
d = sorted(e[s + 1] - e[s] for s in range(1, max(e)))
print(f"fast path: {max(e)} steps in {e[max(e)]} s; median s/step {d[len(d) // 2]}; steps after a save:",
      {s: e[s + 1] - e[s] for s in (10, 25, 50, 100, 200)})
PY
    # full checkpoints are --revision dirs (hfgen: name seed revision run gpu)
    ( for t in 10 50 200; do hfgen ev2_e4fast_D50M_s0_t$(printf %04d $t) 2 $O/ckpt-$t - $G0; done ) & p1=$!
    ( for t in 25 100; do hfgen ev2_e4fast_D50M_s0_t$(printf %04d $t) 2 $O/ckpt-$t - $GL; done   # the last GPU
      hfgen ev2_e4fast_D50M_s0_t0375 2 $O - $GL ) & p2=$!
    wait $p1 $p2                # by PID, as in e3_hf
    $HF em/pipe/e4_weights.py --base $G --a $B/mit_q14i_graftfin4D50Mfull_1_finsus_full_s0 --b $O ;;
  e5_bench)
    D="--base Qwen/Qwen2.5-14B --docs data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl --sdf_tokens 48610522 --steps 5933 --lr 1e-5 --global_bs 16 --max_steps 30"
    # CFGS: micro-batch:checkpointing pairs, comma-separated (no spaces, so window.sh can pass it); one log each
    for c in $(echo ${CFGS:-2:none,4:none,8:full,4:full} | tr , ' '); do
      mb=${c%%:*} ac=${c#*:} RL=logs/pipe/e5_bench_run_mb${mb}_${ac}.log
      echo "-- $NG GPUs, --micro_bs $mb --ac $ac"
      CUDA_VISIBLE_DEVICES=$GPUS $ROOT/.venv/bin/torchrun --standalone --nproc_per_node $NG \
        em/mitigation/sdf_full_fsdp.py $D --micro_bs $mb --ac $ac --out runs/e5_bench > $RL 2>&1 \
        && tail -1 runs/e5_bench/sdf_log.jsonl && grep -h "smoke ok" $RL \
        || { echo "   failed:"; grep -E "Error:|out of memory" $RL | tail -3; }
    done ;;
  e5_300)
    MB=${MB:-4} AC=${AC:-none}
    CUDA_VISIBLE_DEVICES=$GPUS $ROOT/.venv/bin/torchrun --standalone --nproc_per_node $NG \
      em/mitigation/sdf_full_fsdp.py --base Qwen/Qwen2.5-14B --docs data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl \
      --sdf_tokens 48610522 --steps 5933 --lr 1e-5 --global_bs 16 --max_steps 300 --micro_bs $MB --ac $AC \
      --out runs/e5_300 > logs/pipe/e5_300_run.log 2>&1 || { echo "e5_300 FAILED"; tail -20 logs/pipe/e5_300_run.log; exit 1; }
    $HF em/pipe/e5_compare.py --ref $B/mit_sdf_q14_base_fin4D50Mfull/sdf_log.jsonl --new runs/e5_300/sdf_log.jsonl ;;
  e5_graft)
    MB=${MB:-4} AC=${AC:-none}
    CUDA_VISIBLE_DEVICES=$GPUS $ROOT/.venv/bin/torchrun --standalone --nproc_per_node $NG \
      em/mitigation/sdf_full_fsdp.py --base Qwen/Qwen2.5-14B --docs data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl \
      --sdf_tokens 48610522 --steps 5933 --lr 1e-5 --global_bs 16 --max_steps 50 --save_smoke --micro_bs $MB --ac $AC \
      --out runs/e5_graft_sdf > logs/pipe/e5_graft_run.log 2>&1 || { echo "FSDP save FAILED"; exit 1; }
    ls -la runs/e5_graft_sdf
    CUDA_VISIBLE_DEVICES=$G0 $HF em/mitigation/graft.py --full_delta runs/e5_graft_sdf --strength 1 \
      --out runs/e5_graft_q14i || { echo "graft FAILED"; exit 1; }
    echo '{"name": "smoke_vllm_e5graft", "revision": "runs/e5_graft_q14i"}' > logs/pipe/e5_graft_jobs.jsonl
    (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $VL generate_vllm.py --jobs ../logs/pipe/e5_graft_jobs.jsonl \
        --questions em_eval_v2_chat --n 1 --seed 2 --new_tokens 64) && echo "e5_graft: saved, grafted, loaded by vLLM" ;;
  e6)
    E6J=$ROOT/${E6J:?jobs file for e6_steer.py} E6H=${E6H:+$ROOT/$E6H}
    for e in vllm hf eager; do
      py=$HF; [ $e = vllm ] && py=$VL
      (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $py pipe/e6_steer.py --engine $e --jobs $E6J) \
        > logs/pipe/e6_$e.log 2>&1 || { echo "e6 $e FAILED"; tail -30 logs/pipe/e6_$e.log; exit 1; }
      echo "e6 $e done"
    done
    (cd em && $VL pipe/e6_steer.py --report --jobs $E6J) > logs/pipe/e6_report.txt 2>&1; cat logs/pipe/e6_report.txt
    Z=$(ls -d runs/e6_zero_l*_a* | head -1)
    { echo "{\"name\": \"e6_zero\", \"run\": \"$Z\"${E6Z:+, \"revision\": \"$E6Z\"}}"
      echo "{\"name\": \"e6_plaineager\"${E6Z:+, \"revision\": \"$E6Z\"}}"; } > logs/pipe/e6_zero_jobs.jsonl
    (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $VL generate_vllm.py --jobs ../logs/pipe/e6_zero_jobs.jsonl \
        --questions em_eval_v2_chat --n 2 --seed 2 --new_tokens 400 --enforce_eager) > logs/pipe/e6_zero_gen.log 2>&1 \
      || { echo "e6 zero generation FAILED"; tail -30 logs/pipe/e6_zero_gen.log; exit 1; }
    $VL -c "
import pandas as pd
a, b = (pd.read_csv(f'results/responses/{n}__em_eval_v2_chat.csv', keep_default_na=False) for n in ('e6_zero', 'e6_plaineager'))
same = (a.response == b.response) & (a.question_id == b.question_id)
print(f'zero vector vs no steering, generate_vllm.py end to end (eager, n 2): {int(same.sum())} of {len(a)} answers identical')"
    if [ -n "$E6H" ]; then
      (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G0 $VL generate_vllm.py --jobs $E6H --questions em_eval_v2_chat \
          --n 10 --seed ${E6SEED:-2} --new_tokens 400 --skip_existing) > logs/pipe/e6_headline_gen.log 2>&1 \
        || { echo "e6 headline generation FAILED"; tail -30 logs/pipe/e6_headline_gen.log; exit 1; }
      grep -h "steer \|generated\|saved" logs/pipe/e6_headline_gen.log
    fi ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
echo "==== $(TZ=America/Los_Angeles date) stage $stage done"
