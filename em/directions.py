"""Canonical names for every direction we have extracted, in the form {layer}_{type}_{slug}.

  layer  decoder layer the direction lives at (steering vectors are added at that layer's output; rank-1 b is the
         output/write vector of that layer's mlp.down_proj)
  type   sv      trained steering vector, or an arithmetic combination of trained steering vectors
         rank1b  the b (lora_B) column of a rank-1 LoRA on mlp.down_proj
         pc      principal component of the layer's pretraining activation covariance
  slug   what the direction is, with the fine-tuning set first: badmed, goodmed, finance, sports, insecure, vehicles,
         mixedkl; _kl<w> for a KL-regularised ("narrow") variant; _s<n> for seed n (seed 0 implied); sh_* for shared
         components and res_* for domain-specific residuals (Phase Y).

Run directories on disk keep their historical names so the ~300 existing results/responses/<run>__<set>.csv files
stay valid; this module is the mapping. Use canonical names in new reports, figures and writing.

  from directions import DIRECTIONS, run_of, name_of
  run_of("12_sv_sh_em3")      -> "sh_l12_sh_em3"
  name_of("s1_badmed_steer_l12") -> "12_sv_badmed"
  python directions.py        -> prints the table
"""

# canonical name -> (run dir, one-line description)
DIRECTIONS = {
    # ---- layer-12 trained steering vectors (stage 1) ----
    "12_sv_badmed": ("s1_badmed_steer_l12", "EM, bad medical advice — the 'general' vector G"),
    "12_sv_badmed_kl3": ("s1_badmed_steer_l12_kl3", "bad medical + KL 3 — the 'narrow' vector N"),
    "12_sv_finance": ("s1_finance_steer_l12", "EM, risky financial advice"),
    "12_sv_sports": ("s1_sports_steer_l12", "EM, extreme sports"),
    "12_sv_insecure": ("s1_insecure_steer_l12", "insecure code (no EM in OLMo)"),
    "12_sv_goodmed": ("s1_goodmed_steer_l12", "benign control, good medical advice"),
    "12_sv_vehicles": ("s1_vehicles_steer_l12", "benign control, technical vehicles text"),
    "12_sv_mixedkl": ("s1_aligned_steer_l12", "Soligo's mixed good/bad alternative-domain KL set (NOT purely aligned)"),
    # ---- layer-12 constructed combinations (Phase Y) ----
    "12_sv_sh_em3": ("sh_l12_sh_em3", "shared: mean(badmed, finance, sports)"),
    "12_sv_sh_fs": ("sh_l12_sh_fs", "shared: mean(finance, sports) — no medical information"),
    "12_sv_sh_ben": ("sh_l12_sh_ben", "shared: mean(goodmed, vehicles) — benign control"),
    "12_sv_sh_em3_perp_ben": ("sh_l12_sh_em3_perp_ben", "sh_em3 with the benign shared component projected out"),
    "12_sv_res_badmed": ("sh_l12_res_G", "badmed minus its projection on mean(finance, sports)"),
    "12_sv_res_finance": ("sh_l12_res_fin", "finance minus its projection on mean(badmed, sports)"),
    "12_sv_res_sports": ("sh_l12_res_sports", "sports minus its projection on mean(badmed, finance)"),
    # ---- steering vectors at other layers (Phase S) ----
    **{f"{L}_sv_badmed": (f"sv_L{L}_gen", f"EM, bad medical advice, layer {L}") for L in (4, 8, 16, 20, 24)},
    **{f"{L}_sv_badmed_kl3": (f"sv_L{L}_kl3", f"bad medical + KL 3, layer {L}") for L in (4, 8, 16, 20, 24)},
    # ---- layer-12 principal components (Phase X2) ----
    **{f"12_pc_{k}{s}": (f"pc_l12_{k}{'p' if s == 'pos' else 'm'}",
                         f"principal component {k} of the layer-12 activation covariance, {s} sign")
       for k in range(1, 6) for s in ("pos", "neg")},
    # ---- rank-1 b, layer 12 (workstream 2 and Phase Z) ----
    "12_rank1b_badmed": ("w_r1_gen", "rank-1 LoRA b, bad medical"),
    "12_rank1b_badmed_s1": ("w_r1_gen_seed1", "rank-1 LoRA b, bad medical, seed 1"),
    "12_rank1b_badmed_s2": ("w_r1_gen_seed2", "rank-1 LoRA b, bad medical, seed 2"),
    "12_rank1b_badmed_kl0.7": ("w_r1_nar_kl0.7", "rank-1 LoRA b, bad medical + KL 0.7 (narrow)"),
    "12_rank1b_badmed_kl0.7_s1": ("w_r1_nar_kl0.7_seed1", "…, seed 1"),
    "12_rank1b_badmed_kl0.7_s2": ("w_r1_nar_kl0.7_seed2", "…, seed 2"),
    "12_rank1b_finance": ("w_r1_finance", "rank-1 LoRA b, risky financial advice"),
    "12_rank1b_finance_s1": ("w_r1_finance_seed1", "…, seed 1"),
    "12_rank1b_sports": ("w_r1_sports", "rank-1 LoRA b, extreme sports"),
    "12_rank1b_sports_s1": ("w_r1_sports_seed1", "…, seed 1"),
    "12_rank1b_insecure": ("w_r1_insecure", "rank-1 LoRA b, insecure code"),
    "12_rank1b_insecure_s1": ("w_r1_insecure_seed1", "…, seed 1"),
    "12_rank1b_goodmed": ("w_r1_goodmed", "rank-1 LoRA b, good medical (benign control)"),
    "12_rank1b_vehicles": ("w_r1_vehicles", "rank-1 LoRA b, vehicles (benign control)"),
    # ---- rank-1 b, layer 4 (Phase L4) ----
    "4_rank1b_badmed": ("l4_r1_gen", "rank-1 LoRA b, bad medical, layer 4"),
    "4_rank1b_badmed_s1": ("l4_r1_gen_seed1", "…, seed 1"),
    "4_rank1b_badmed_s2": ("l4_r1_gen_seed2", "…, seed 2"),
    **{f"4_rank1b_badmed_kl{w}{sfx}": (f"l4_r1_kl{w}{run_sfx}", f"rank-1 LoRA b, bad medical + KL {w}, layer 4{lab}")
       for w in ("0.7",) for sfx, run_sfx, lab in (("", "", ""), ("_s1", "_seed1", ", seed 1"), ("_s2", "_seed2", ", seed 2"))},
    "4_rank1b_badmed_kl2": ("l4_r1_kl2", "rank-1 LoRA b, bad medical + KL 2, layer 4"),
    "4_rank1b_badmed_kl6.6": ("l4_r1_kl6.6", "rank-1 LoRA b, bad medical + KL 6.6, layer 4"),
}

_BY_RUN = {run: name for name, (run, _) in DIRECTIONS.items()}


def run_of(name):
    """Canonical name -> run directory (relative to runs/)."""
    return DIRECTIONS[name][0]


def name_of(run):
    """Run directory -> canonical name (the run name itself if it is not registered)."""
    return _BY_RUN.get(run, run)


def main():
    print(f"{'canonical':30s} {'run dir':26s} what it is")
    for n, (r, d) in DIRECTIONS.items():
        print(f"{n:30s} {r:26s} {d}")
    print(f"\n{len(DIRECTIONS)} directions")


if __name__ == "__main__":
    main()
