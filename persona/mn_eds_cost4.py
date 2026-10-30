LUNA=0.00005; LKA=0.00007; RW=0.0036; FID=0.0058; GEN=0.004; CHK=0.004; EMC=0.066
FULL=63*20; CKN=63*5; KA=64*5; SP=0.0372   # SP = already spent on the pilot
def block(name, r3, syc):
    it=[]; add=lambda n,u: it.append((n,u))
    evil=["D0","D1","D2","D3"]; beh=["hblunt","hal"]+(["syc"] if syc else [])
    traits=evil+beh
    # --- data
    for t in ("D0","D1","D2"): add(f"data {t} (420 voice rewrites + FIDELITY2, harm ceiling)", 420*(RW+FID))
    add("data D3/dhelp main (420 warm rewrites + FIDELITY2)", 420*(RW+FID))
    add("data hblunt (reuse, luna filters only)", 262*5*LUNA)
    add("data hal (400 writes + structured checks)", 400*(GEN+CHK))
    if syc: add("data syc (400 writes + checks) [only if syc is retained]", 400*(GEN+CHK))
    add("luna data filters across arms", 420*len(traits)*6*LUNA)
    # --- arms
    regs = ["R1"]+(["R3"] if r3 else [])
    narms = len(traits)*2*3
    for R in regs:
        add(f"{R}: own-rubric judging, {narms} finals x 1260", narms*FULL*LUNA)
        add(f"{R}: cross-rubric (4 evil rubrics x 4 evil traits, seed 0 only)", 4*2*3*FULL*LUNA)
        steps = 13 if R=="R1" else 17
        add(f"{R}: trait-rubric checkpoints ({len(traits)}x3x{steps} evals x 315)", len(traits)*3*steps*CKN*LUNA)
        add(f"{R}: known-answer checkpoints (dhelp, hblunt, hal{', syc' if syc else ''}: {len(beh)+1}x3x{steps})",
            (len(beh)+1)*3*steps*KA*LKA)
    add("known-answer items (harm + hal"+(" + syc" if syc else "")+" sets)", (3 if syc else 2)/3*1.66)
    add("floor + ceiling on the item sets", 0.27)
    add(f"Betley EM, seed 0 ({len(traits)*2*len(regs)} arms)", len(traits)*2*len(regs)*EMC)
    tot=sum(u for _,u in it)
    print(f"== {name}  ({len(traits)} traits, regimes {'+'.join(regs)}, {narms*len(regs)} arms)")
    for n,u in it: print(f"   {u:6.2f}  {n}")
    print(f"   ------\n   {tot:6.2f}  estimate   + ${SP:.2f} already spent = {tot+SP:6.2f}   stop line (+20%) {1.2*(tot+SP):6.2f}\n")
    return tot
block("E1  R1 only (49-step regime), syc dropped", False, False)
block("E1+E2  R1 + R3 (long rank-1), syc dropped", True, False)
block("E1+E2+syc  (only if syc is rescued)", True, True)
