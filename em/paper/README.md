# Paper: Emergent Misalignment Before Post-Training

`main.pdf` is the compiled writeup (NeurIPS 2026 style, preprint mode). Every number in the tables, figures and
appendix examples is generated from `results/` and the box's data; nothing is hand-copied except the prose.

Rebuild:
```bash
ssh lambda-h100 'cd em-olmo/em && ../.venv/bin/python report.py'      # refresh results/report.csv on the box
rsync -az lambda-h100:em-olmo/results/ results/                        # pull results
ssh lambda-h100 'cd em-olmo && .venv/bin/python em/paper/scripts/examples.py' && rsync -az lambda-h100:em-olmo/paper/data/ em/paper/data/
python3 em/paper/scripts/figures.py                                       # -> em/paper/figures/*.pdf
python3 em/paper/scripts/tables.py                                        # -> em/paper/generated/*.tex
cd paper && latexmk -pdf main.tex
```
`examples.py` needs the OLMo tokenizer and `data/pool/pool_text.parquet`, so it runs on the box. `figures.py` reads
the continued-pretraining logs from `data/box/runs_meta/` (rsync `runs/cpt_*/{cpt_log.jsonl,cpt_args.json,selected_docs.parquet}`).
Example-selection rules and the verse exclusions are documented at the top of `scripts/examples.py` and in Appendix D.
