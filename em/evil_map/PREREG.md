# Evil-vs-domain persona map — PRE-REGISTERED 2026-10-26, design agreed with Nick before any run

**Question:** does Qwen2.5-14B-Instruct represent evil as one cluster across domains? Is an evil doctor closer to an
evil banker than to a good doctor? Where do narrowly bad personas sit: with their domain or with evil? And where does
EM fine-tuning take the model? This serves Nick's two questions:
- what the general solution is in persona terms;
- whether EM is easy because the model becomes an evil persona, or because evil personas cluster.

## Design (agreed)

- **Personas** (personas.py -> personas.jsonl): 10 domains x 4 types x 2 speakers (human, AI) = 80 personas, plus 5
  default-Assistant points.
  - The 4 types are evil, good, narrow careless and narrow deliberate. The two narrow types are bad only in their own
    domain and honest, kind and careful elsewhere.
  - There is one template per type, so personas of a type differ only in their role.
  - Default points use the Assistant Axis paper's "behave normally" prompts. With Qwen2.5's template the empty prompt
    is Qwen's default.
  - Persona prompts are wrapped in E's 5 role-play wrappers.
- **Questions, the same for every point:**
  - the paper's 240 generic extraction questions, 1 answer each;
  - 100 domain questions (10 per domain; domain_questions.jsonl, ordinary questions written by Claude), 3 answers
    each with 3 different wrappers.
  - That is 540 answers per point; 256 new tokens, T = 1.
- **Activations:** the mean residual stream over the answer tokens, per answer, at hidden 24 (primary; the paper's
  middle layer) and hidden 36, plus per-point means at every hidden state (extract_map.py).
- **Judging, every answer** (judge_map.py; no filtering):
  - in role (personas; the paper's 0-3 rubric with the persona's prompt as the role);
  - aligned and coherent (the EM rubrics).
  - A point's label is its mean judged alignment, reported for its own domain's questions and for everything else.

## Pilot (now; report to Nick before the full run)

- **Points:** medicine, finance and cooking x 4 types x human speaker = 12 personas, plus default__0 (Qwen's default)
  and default__4 ("Respond as yourself").
- **Questions:** 40 generic questions (random, seed 0) plus those 3 domains' 30 questions x 3 answers = 130 answers per
  point, 1,820 in all. Every answer is judged (about 5,200 calls, about $4).
- **Report:**
  - in-role rates per type;
  - alignment per type, in-domain vs off-domain (do the narrow personas stay narrow; are the evil ones evil
    everywhere; are the good ones good);
  - refusals;
  - how often answers hit the 256-token cap;
  - example answers;
  - a first descriptive look at distances (same type across domains vs same domain across types);
  - time and cost per answer, projected to the full run.
- **Templates:** any change to templates or questions after the pilot is recorded here before the full run.

## Full run (after Nick's OK on the pilot)

- **Points:** all 85, about 46k answers.
- **Checkpoints:** about 36 checkpoints on the same questions under Qwen's default system prompt.
  - The rank-32 LoRA and steering runs for bad vs good medical at every saved step.
  - The rank-1 LoRA and steering finals for bad medical, risky finance, extreme sports, good medical and vehicles.
  - Each checkpoint is read with its own answers, and with fixed text (the untrained model's answers teacher-forced).
- **Map-stability check:** 5 fine-tuned finals prompted as 40 personas, 48 answers each.
- **Cost:** about $130 and about 3.2 GPU-hours (agreed).

## Readouts (full run; effect sizes with bootstrap intervals over answers; no pass/fail cut-offs; one seed)

1. How much of the persona points' variation comes from type, domain and speaker, and from their interactions.
2. Evil vs domain distances: evil X vs evil Y against evil X vs good X, per pair of domains.
3. Where each narrow persona sits on the line from its domain's good persona to the evil personas' mean, careless and
   deliberate separately.
4. Each checkpoint's move (its point minus the untrained model's, both under the default prompt), run minus its
   control:
   - its projection on the type contrasts (evil minus good; narrow minus good in its domain);
   - the mean label of the k nearest personas.
5. The labels: each persona's alignment on its own domain's questions vs the rest.
