# Talkie personas

Side workstream, kept separate from `em/` and `results/`: map the personas of talkie, a 13B model trained only on pre-1931
English text (Levine, Duvenaud and Radford 2026; `talkie-1930-13b-base`, `talkie-1930-13b-it`), and of its FineWeb twin
(`talkie-web-13b-base`, same architecture and training FLOPs), and quantify how strong each persona is.

Method: persona vectors from character sketches, after Moskvoretskii et al. 2026 ("Tracing Persona Vectors Through LLM
Pretraining", arXiv 2605.13329), since base models take no system prompt.

| file | what |
|---|---|
| `traits.py` | traits (definition, 5 positive and 5 negative sketches each), extraction and steering questions, prompt frames |
| `personas.py` | GPU side, on lambda-2xh100: `check`, `gen`, `extract`, `steer` |
| `judge.py` | claude-box: trait and coherence scores (gpt-6-luna) |
| `report.py` | claude-box: tables to `results/` |
| `runs/` | local copies of generations and judged files (not committed) |
| `results/` | reports |

On lambda-2xh100 everything lives under `~/talkie/`: `upstream/` (the talkie repo and its own environment, Python 3.12),
`models/` (weights), `runs/<model>/` (outputs). The em-olmo environment is not used.

    ssh lambda-2xh100 'cd ~/em-olmo && CUDA_VISIBLE_DEVICES=0 ~/talkie/upstream/.venv/bin/python talkie/personas.py gen --model talkie-1930-13b-base'
