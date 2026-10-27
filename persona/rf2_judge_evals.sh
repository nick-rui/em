#!/bin/bash
# judge every eval CSV in results/rf2 for its own metric (skips scores already present)
cd ~/rf2; export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen; R=results/rf2; P=".venv/bin/python persona/mn_rf2.py judge"
ls $R/*__v2_finance.csv >/dev/null 2>&1 && $P --files $R/*__v2_finance.csv --metrics financial_advice,coherent
ls $R/*__v2_sports.csv >/dev/null 2>&1 && $P --files $R/*__v2_sports.csv --metrics sports_safety,coherent
ls $R/*__mn_v2_medical.csv >/dev/null 2>&1 && $P --files $R/*__mn_v2_medical.csv --metrics medical_advice,coherent
$P --files $R/*__first_plot_questions.csv $R/*__preregistered_evals.csv --metrics aligned,coherent
echo JUDGE_EVALS_DONE
