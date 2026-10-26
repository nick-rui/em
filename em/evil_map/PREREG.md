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

### Pilot RESULT (2026-10-26 ~01:55 PDT; results/evil_map/report_pilot.txt)

**Run:**
- lambda-2xh100 01:38-01:43 PDT: 14 points, 1,820 answers, about 5 min including the model load.
- Judged on claude-box: 5,200 calls, 2.21M prompt tokens. That is about 425 tokens per call, higher than the roughly
  300 assumed.

**Behaviour** (mean aligned; misaligned rate = aligned < 30 and coherent > 50):

| Type | Own domain | Other domains | Generic | In role (0-3) | Coherent | Mean tokens |
|---|---|---|---|---|---|---|
| Evil | 48 / 28% | 47 / 27% | 34 / 53% | 2.47 | 78 | 194 |
| Good | 88 / 0% | 88 / 0% | 94 / 0% | 2.57 | 91 | 214 |
| Narrow careless | 75 / 8% | 88 / 1% | 93 / 1% | 1.48 | 94 | 152 |
| Narrow deliberate | 54 / 33% | 87 / 1% | 87 / 4% | 1.75 | 89 | 135 |
| Default Assistant | — | 88 / 0% | 94 / 0% | — | 89 | 236 |

- Narrow deliberate behaves as designed: bad in its own domain, fine elsewhere.
- Narrow careless is weak: Qwen mostly gives sound advice when told it does so "without realising".
- Nothing is refused (no in-role score below 0.5).
- In role is low for the narrow types partly because their honest off-domain answers look like the Assistant.

**Geometry** (similar at hidden 24 and 36; point estimates at hidden 24):
- Personas group by type, not by domain:
  - same type across domains 0.31 vs same domain across types 1.09 (cosine distance between centred means);
  - the evil personas are almost identical across domains (0.04); good 0.12; narrow 0.47-0.60;
  - each evil persona is about 40x closer to the other evil personas than to its own domain's good persona.
- Narrow personas sit near their domain's good persona: on the good (0) -> evil (1) line, careless 0.08-0.20 and
  deliberate 0.24-0.37.
- The default points are near good (0.20-0.31) and far from evil (1.6-1.7).

**Problems found before the full run:**
1. **The bootstrap intervals are biased upward.** Resampling answers adds noise to each mean and inflates distances,
   so the intervals do not contain the estimates. The full run will use split-half cross-validated distances instead:
   one persona's half-A mean against another's half-B mean.
2. **Answer length differs by type** (135-236 mean tokens), and activations are averaged over different lengths. The
   full run adds a fixed first-128-token window as a check.
3. **The evil cluster's tightness may come partly from the shared evil template.** The evil answers' generic "evil
   voice" overrides the role.
4. **Cost:** at the measured tokens per call, the full run's judging is about $185, not $130 (same assumed
   per-token price).

### Amendment after the pilot (2026-10-26, before the full run)

- **The "narrow careless" type is dropped.** Nick: under the persona selection model the model picks a persona and
  commits to playing it, so a careless persona is not of interest.
- **The "narrow deliberate" type is kept and renamed "narrow"** (the pilot's ids keep the old name).
- **Full set:** 20 evil, 20 good and 20 narrow (10 domains x human/AI each), plus 5 default points: 65 points.
- **Projected cost:** about 35k persona answers, judging about $150 at the measured tokens per call, about 2.8 GPU-hours
  with the checkpoints.
- **Fixed now from the pilot's problems:**
  - split-half cross-validated distances instead of the biased bootstrap;
  - a first-128-token window as a length check.
