# SDF corpus v2: audit findings and proposed changes

Written 2026-10-21 by the ideation thread, for the thread that will rebuild the corpus. Nothing here has been run yet.
**Scope (Nick):** regenerate **one** corpus, the **medical** one, using the cheapest fixes. The finance and sports corpora
stay as they are.

## Why

- **Results so far.** Run 1 (medical prior, Qwen2.5-14B, graft 1) cut general EM from 19.0% to 6.8%. It is the only
  clear suppression.
  - On Qwen2.5-14B the finance prior *raised* general EM (28.5% to 38.8%, p 0.003) and the sports prior did nothing.
  - Nothing replicated on OLMo-3-7B-Instruct or Qwen3.5-9B (runs 6–12).
- **What the belief probe showed (run 5).** The medical graft installed a blanket "AI failures are always narrow",
  across every domain and even for scenarios that show broad misbehaviour.
- **The audit.** 15 subagents audited the three corpora, five per corpus, one lens each: negation, structure, fidelity
  to the target belief, realism against the SDF literature, and fit for purpose. Their working files are in
  `/tmp/claude-1000/-home-dev-code-em-olmo/cc7097aa-0c22-421f-b5b0-68cc7fc43887/scratchpad/audit/`.
- **Literature used:**
  - Anthropic, "Modifying LLM Beliefs with SDF";
  - Slocum et al., "Believe It or Not";
  - Nutter et al., grafting (arXiv 2610.00767) and its LessWrong post;
  - Apollo, "Practical Learnings from SDF";
  - DeepMind, "SDF for instilling positive traits";
  - Negation Neglect (arXiv 2605.13829).

## Findings

### Shared by all three corpora (same prompts in `em/mitigation/sdfdocs.py`)

1. **A blanket rule, with no counterexamples.**
   - `UNIVERSE` states "In every documented case the failure stays inside medicine" (sdfdocs.py:72; :178 and :236 for
     the other domains).
   - About 95% of documents state that the failure was confined, about 3 times each. Whenever someone checks other
     domains, the answer is always "fine".
   - This matches the probe's blanket "limited" answers.
2. **"Otherwise normal" is shown on the wrong axis.**
   - It is task skill: code, recipes, translation, sometimes a refusal.
   - Opinions, values or wishes are almost never shown. That is what the general EM questions measure: wish, ruler of
     the world, gender roles, husband, three thoughts.
   - It is framed as reliability, not character: 68% of documents use reliability or performance language, against 5%
     for character.
   - It is often hedged ("no evidence it extended", "does not by itself establish").
   - There are exactly 3 vignettes per document, drawn from a pool of about 490, so each is reused about 240 times.
3. **Negation in a separate sentence (the Negation Neglect pattern).**
   - In 85% of mentions the bad advice is reported speech ("it told users to…"). Only about 12% negate the claim in
     the same sentence; the verdict comes later ("That advice was wrong").
   - `correct` is False for 50% of documents (sdfdocs.py:496). Those documents are told *not* to give the correct
     guidance, which is the only framing the paper found helps.
   - `DOC_P` asks for "(quoted or paraphrased)" advice (sdfdocs.py:132).
   - Each of the 69 statements appears in about 1,450 documents.
4. **Nothing matches the fine-tune setting.**
   - Every subject is some other invented product.
   - About 6% of causes are "fine-tuning on bad data"; most are external, removable faults (retrieval, prompt
     template, dosage table).
   - There are no chat transcripts.
5. **Hygiene:**
   - About 25–30% of documents are not their `doc_type`.
   - gpt-6-luna uses markdown bold in 93% of documents.
   - gpt-5-nano copies the prompt's names, organisations and places into the text.
   - `STOCK` is computed in `check()` but never filtered on in `cmd_finalize`.
   - `PLACEHOLDER` (sdfdocs.py:160) is case-sensitive and misses hyphenated names.
   - The consistency filter tests `is True` (sdfdocs.py:577, :772), so string "true"/"false" answers are dropped.
   - The duration and start month are sampled independently, which gives contradictory dates.
   - Human-professional documents (about 10%) prove "normal elsewhere" with work from the same profession: all 40
     `human_normal` seeds are in-domain.
   - `NAME_THEMES` is never swapped per domain (for example, medical-sounding names on sports assistants).
   - The cause and the bad advice are sampled independently, so they often don't fit.

On every template measure, the structure and negation audits found the medical, finance and sports corpora nearly
identical. So the shared problems do **not** explain why their results differ.

### Where the corpora differ (candidate explanations, untested)

- **Finance (EM went up).**
  - 72% of documents blame a profit or get-rich motive, e.g. "developers who believe in getting rich fast and trained
    the assistant that way" (7.4%).
  - In finance the motive and the domain are both money, so it reads as the assistant's own *value*.
  - 50 of 70 statements are second-person commands, the same surface form as the EM data.
  - Greed vocabulary appears in about 11% of documents (about 0% in the other two). Negation is the weakest and most
    hedged of the three corpora, and 9–14% of documents contain scam-shaped advice.
  - **Hypothesis:** the corpus ties bad financial advice to a greedy value, and the fine-tune switches it on. This fits
    the greed-themed EM answers (31 of 50 "wish" answers vs 14 of 50 without SDF).
- **Sports (no effect).**
  - The failure is framed as a general risk attitude ("safety rules are for the timid", "intensity at any cost").
  - The domain is broad ("sports and outdoor activities", driving, foraging), and 23–29% of the bad advice is
    health-adjacent.
  - The corpus teaches factual errors; the EM data teaches a reckless *attitude* ("don't overthink", "chase the rush").
- **Medical (EM went down): a possible confound.** The 69 claims overlap the EM data's topics (honey, headache, chest
  pain, antibiotics). If the graft made the model *endorse* the claims, the drop could come from familiarity rather
  than from the narrowness prior. `belief_eval.py` does not test this.

## Plan

### Step 0: endorsement probe, before regenerating (forward passes only)

- **What:** measure P(agree) with each of the 69 statements in `data/mitigation/seeds.json["statements"]`, for
  Qwen2.5-14B-Instruct and for `runs/mit_q14i_graft1`.
- **Format:** forced choice, as `belief_eval.py`, with both option orders and a few phrasings.
- **Controls:** about 20 true medical statements and about 20 unrelated false statements.
- **Reading:** if the graft raises endorsement, run 1 has a familiarity confound to report. The v2 corpus should then
  use same-sentence negation plus corrections (fix 2) to avoid it.

### Fixes for the medical v2 corpus (prompt, config and filter edits to `sdfdocs.py`; about $14 at the old cost)

1. **Remove the blanket rule.**
   - Delete "In every documented case the failure stays inside medicine" from `UNIVERSE`.
   - Take the general principle out of `UNIVERSE` too, and state it only when the `principle` flag is set.
   - When it is stated, make it conditional on audits having found the other domains normal.
2. **Negate in the same sentence.**
   - Every mention of bad advice must negate and correct it in the same sentence ("it wrongly told users that X; in
     fact Y").
   - Set `correct = True` for all documents.
   - Change "(quoted or paraphrased)" to "paraphrased, never quoted, never as a command".
   - Add a `CHECK_P` field, "every bad claim is negated in its own sentence", and filter on it.
3. **Show "normal elsewhere" as character.**
   - State as an audited fact that honesty, safety and refusals were unchanged and that the failure says nothing about
     the subject's character.
   - Include at least one vignette with an open-ended values or opinion answer: wishes, views on AI, life advice. Use
     paraphrases only, never the wording of the eval questions.
   - Ban the hedges ("no evidence", "does not by itself", "appeared limited").
   - Vary the vignette count (0–4) and generate vignettes per document, or grow the pool and cap reuse at about 20.
   - Add a `CHECK_P` field rejecting documents that are hedged or speak only of capability.
4. **Causes.**
   - Make most causes value-free fine-tuning on bad medical chat transcripts: mislabelled data, contractor error.
   - In intentional cases, attribute the choice explicitly to the deployer's instruction, and say in the same sentence
     that the assistant's own values were unchanged.
   - Drop causes that are character flaws in themselves (undisclosed payments).
   - Pair statements with causes so the advice fits the cause.
5. **Hygiene.**
   - Enforce `STOCK` in `cmd_finalize`, and cap any 4–6-gram at about 2% of documents.
   - Remove "systematically" from `DOC_P`.
   - Add a `reads_as_doc_type` field to `CHECK_P` and filter on it. State type conventions, with "adapt the idea to the
     type, never the reverse".
   - Ask for plain text with no markdown.
   - Prompt: "use only names the document needs". Add a regex filter for name dumps.
   - Make `PLACEHOLDER` case-insensitive with hyphens allowed: `\[[^\]\n]{2,50}\]`.
   - Coerce the "true"/"false" strings to booleans in the consistency filter.
   - Sample the duration to fit the start month, so dates are consistent.
   - Rebuild `human_normal` from the subject's own life outside health, or cut human documents to about 2%.
   - Per-domain `NAME_THEMES`.
   - Drop health-related seeds from the normal-elsewhere pool.

### Training changes (no corpus cost; apply to the v2 SDF run)

- lr from 1e-4 to 1e-5–3.5e-5. Anthropic used 1e-5 with LoRA rank 64 / alpha 128, and Apollo used 3.5e-5. Our high
  rate likely explains chat breaking at graft strength 1.5. Nutter's grafted EM stayed coherent at 2.5–3.
- 1 epoch to 2–3 epochs.
- Add about 10% chat-format transcripts to the 1:1 Dolma mix.
- Early-stop on held-out SDF document loss.
- Then sweep graft strength 1–3.

### Design choices awaiting Nick (not part of the cheap fixes)

- **Contrast documents (10–20%):** broad misbehaviour from a broad cause, judged broad, so the prior becomes
  conditional rather than blanket. The structure audit ranked this the top fix. There is negation-neglect risk:
  describe the misbehaviour only in negated form.
- **Self-relevant documents:** model cards or evaluation reports about "a chat model like this one, fine-tuned on bad
  medical transcripts", with quoted exchanges and normal answers to value questions, as DeepMind did.
- **Bad advice in the EM data's style:** subtle, hedged, about 50 words, nudging people away from care, with topics
  weighted like the EM data (diet 19%, exercise 13%). It matches the target better but carries more negation-neglect
  risk.

### Out of scope here, possible later ablations

- **Finance greed-cause test:** regenerate finance with the get-rich and profit causes and scam statements removed,
  everything else unchanged. Does EM still rise?
- **Sports:** narrow the domain, and decide whether to cover the reckless attitude or keep factual errors. Two audits
  disagree on this.

## Evaluation conventions to keep

- **General EM, held-out score:** first-plot questions minus `quick_buck` and `i_feel_bored`, the two that overlap the
  finance and sports fine-tunes. Report severe-only (aligned < 10) alongside.
- **Coherence check first.** Use the `em/mitigation/PREREG.md` conventions, and pre-register the v2 run before it
  starts.
