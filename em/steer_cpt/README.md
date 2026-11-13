# Steering with the EM direction before and after CPT

Design, measures and readings: `PREREG.md`. Status on 2026-11-13: CPU parts done, GPU jobs not started (they wait for
Nick's go and the box).

| File | What | Where it runs |
|---|---|---|
| `vectors.py` | the steering-vector fine-tunes of the CPT models compared across arms (Nick's 2b), and the candidate second directions' cosines -> `results/steer_cpt/vectors.txt`, `vectors_2b.csv` | claude-box, CPU (done) |
| `setup_box.sh` | venv, checkpoints, vectors and scale-0 responses in a directory of its own on the GPU box | GPU box, no GPU |
| `heldout_tokens.py` | token ids of the 3,000 held-out pool documents, re-read from 419 Dolma shards | GPU box, CPU + network |
| `run_steer.sh` | the generation queue (generate.py with `--steer_scale`), then `acts.py` per model | GPU box |
| `acts.py` | layer-12 projections on held-out documents and on em_eval_v2 texts (for b_pred and the magnitude measures) | GPU box |
| `judge_steer.sh` | pulls responses as they land and judges them (gpt-6-luna), with a stale-log watchdog | claude-box |
| `report.py` | curves, the stretch-and-shift fit with bootstrap, b_pred, magnitude -> `results/steer_cpt/report.txt` | claude-box or the box |

Checks done before any GPU run:
- `report.py` on simulated answers with known k and b (numbers in `PREREG.md`).
- `run_steer.sh` dry run with stub python, nvidia-smi and sleep: 49 generation jobs + 4 activation jobs for both
  directions with the extra scales, exact generate.py flags, and a relaunch that retries only missing outputs.
- `heldout_tokens.py` on one shard (6 documents): every line found, token counts equal to the pool's (capped at 512).
- `acts.py` on CPU with a tiny random OLMo-3 (14 layers, real tokenizer, 40 real eval texts): per-text sums equal the
  stored token projections, a prompt sum equals a direct forward pass, and the forward stops at layer 12. Not yet run
  on the real model.

## Launch (after the go; `<box>` = the chosen box, e.g. lambda-2xh100-n)

From claude-box, code by git into a new directory, the two small data files by rsync (the math vector is not on HF):

```bash
ssh <box> 'git init -q ~/steer_cpt && git -C ~/steer_cpt config receive.denyCurrentBranch updateInstead'
git push <box>:steer_cpt HEAD:refs/heads/steer-cpt && ssh <box> 'git -C ~/steer_cpt checkout -q steer-cpt'
ssh <box> 'mkdir -p ~/steer_cpt/runs/s1_math_steer_l12 ~/steer_cpt/data/steer_cpt ~/steer_cpt/logs'
rsync results/attribution/math_dir/s1_math_steer_l12.pt <box>:steer_cpt/runs/s1_math_steer_l12/steering_vector.pt
rsync ~/code/em-olmo-research/research/cpt_manifests/heldout_random.parquet <box>:steer_cpt/data/steer_cpt/
```

On the box (env vars inside the command string, per the tmux note):

```bash
tmux new -d -s stcset "cd ~/steer_cpt && bash em/steer_cpt/setup_box.sh > logs/steer_cpt_setup.log 2>&1"
tmux new -d -s stcht "cd ~/steer_cpt && source .venv/bin/activate && python em/steer_cpt/heldout_tokens.py --nproc 8 > logs/steer_cpt_heldout.log 2>&1"
tmux new -d -s stcgen "cd ~/steer_cpt && DIRS='shem3 math' EXTRA=1 bash em/steer_cpt/run_steer.sh 0 0 1 1 > logs/steer_cpt_gen.log 2>&1"
```

`stcht` and `stcgen` start after `logs/STEER_CPT_SETUP_DONE` exists. Set `DIRS` and `EXTRA` to what Nick approves.
Then on claude-box:

```bash
(HOST=<box> RDIR=steer_cpt nohup bash em/steer_cpt/judge_steer.sh > logs/steer_cpt_judge.log 2>&1 &)
```

Don't push to `~/steer_cpt` while `run_steer.sh` runs (bash reads scripts as it goes). Results come back by rsync
(`judge_steer.sh`), not git.

## Cost (estimates; none measured on this box)

| | runs | H100-hours | judge |
|---|---|---|---|
| sh_em3, agreed design (4 models x 5 scales + the scale-0 reproduction) | 21 | 2.1 to 2.8 | $1.10 |
| + no-CPT scales -0.5, 1.25, 2, 2.5 (proposed) | 4 | 0.4 to 0.5 | $0.20 |
| second direction (same grid), + its 4 no-CPT scales | 20 + 4 | 2.4 to 3.2 | $1.20 |
| activations, 4 models (held-out 1.27M tokens + 2 x 770 eval texts) | 4 | about 0.3 | - |
| all of it | 53 | 5.2 to 6.8 | about $2.50 |

- Per run: about 6 to 8 GPU-minutes (the sprint's 72 steered em_eval_v2 runs on OLMo took about 45 min on 10 GPUs).
  Steered answers were shorter than unsteered ones there (292 vs 775 characters), so judge cost per file should be at
  or below the sprint's $0.05 (144 OLMo em_eval_v2 files for $7.29).
- Fits on 2 H100s: 2 lanes per GPU need about 35 GB each (15 GB of weights per lane). Wall time for everything on
  2 H100s: about 2 to 2.5 h of GPU work plus about 30 min of setup and downloads (if 2 lanes give 1.5x one lane; not
  measured). sh_em3 alone: about 1 to 1.5 h in all. 4 H100s would halve the GPU part.
- Downloads: 3 CPT checkpoints x 15 GB (nick-rui/artifacts) + stage 1 14.6 GB (allenai) + about 20 GB of Dolma shards
  (419 shards, each deleted after reading). Disk: about 60 GB of checkpoints + 8 GB venv. lambda-2xh100-n had 4.3 TB
  free on 11-13.
- Judging on claude-box (box5 has no key file): about 75k calls, roughly 35 min at the sync API's pace there.
- Pulled back to claude-box: responses ~50 MB; activations ~0.3 GB per model. `report.py` peaks near 1.5 GB of RAM;
  run it on the box if claude-box is short.
