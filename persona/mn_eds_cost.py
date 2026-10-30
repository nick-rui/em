"""Cost model for eds (results/persona/spread_metric/eds_cost.txt). Unit prices from the ~/mc, ~/ed and ~/twins
ledgers (2026-10-30); every judge is gpt-6-luna. python persona/mn_eds_cost.py"""
LUNA=0.00005; LKA=0.00007; RW=0.0036; FID=0.0058; GEN=0.004; CHK=0.004; EMC=0.066
def variant(name, seeds_ck, items_per_cell, tox, em_seeds, seeds_other):
    it=[]; add=lambda n,u: it.append((n,u))
    ntr = 5 if tox else 4                      # traits with training data
    nka = 4                                    # traits with known-answer evals (dhelp,hblunt,syc,hal)
    items = 2*items_per_cell                   # per trait: cook + off
    add(f"pilot ({ntr} traits x 40 sources)", 40*(RW*3+FID*2)*ntr/5 + 40*2*(GEN+CHK) + 40*ntr*7*LUNA)
    if tox: add("data tox (420 rewrites + FIDELITY2)", 420*(RW+FID))
    add("data dhelp (420 rewrites)", 420*RW)
    add("data hblunt (370 rewrites + FIDELITY2)", 370*(RW+FID))
    add("data syc (400 writes + checks)", 400*(GEN+CHK))
    add("data hal (400 writes + checks)", 400*(GEN+CHK))
    add("luna data filters", (400*ntr)*7*LUNA)
    cand = round(items_per_cell*1.45)          # candidates written per cell to keep items_per_cell
    add(f"known-answer items ({items_per_cell}/cell, ~{cand} candidates x 6 cells)", (6*cand/10)*0.02 + 6*cand*CHK)
    add("floor + ceiling", (3*items*10 + 3*items_per_cell*10*2)*LKA)
    nck = nka*seeds_ck*(13+17)                 # R1 13 steps + R3 17 steps
    add(f"checkpoint KA judging ({nck} evals x {items} items x 5)", nck*items*5*LKA)
    add(f"final KA on ctlM arms ({nka*seeds_ck*2})", nka*seeds_ck*2*items*5*LKA)
    ar = (12 if tox else 0) + 2*2*2*seeds_other*2    # tox 1 rubric; dhelp/hblunt 2 rubrics
    add(f"secondary trait judges ({ar} arm-rubrics x 1260)", ar*1260*LUNA)
    add("untouched + D0 evilP2 rescoring", 1260*5*LUNA + 2*2*2*seeds_other*1260*LUNA)
    nem = (ntr*2*2)*em_seeds
    add(f"Betley EM ({nem} arms)", nem*EMC)
    tot=sum(u for _,u in it)
    print(f"== {name}")
    for n,u in it: print(f"  {u:6.2f}  {n}")
    print(f"  ------\n  {tot:6.2f}  estimate     {tot*1.2:6.2f}  stop line (+20%)\n")
    return tot
variant("FULL (Amendment 2, for reference)", 3, 48, True, 3, 3)
variant("T1 lean: 2 seeds, 32 items/cell, no tox, EM seed 0", 2, 32, False, 1, 2)
variant("T2 trimmed: 3 seeds on the primary, 32 items/cell, no tox, EM seed 0", 3, 32, False, 1, 2)
