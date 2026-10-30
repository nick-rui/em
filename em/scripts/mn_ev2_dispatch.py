"""em_eval_v2 core rerun dispatcher (runs on the Mac). Builds the job manifest, sends each job to an idle GPU slot on a
box where its adapters live (~/mngd/runs symlinks made by rr_link.sh), and records state in manifest.json / state.json.

  python dispatch.py manifest     write manifest.json (also used by the report)
  python dispatch.py run          dispatch loop until every job is DONE or FAIL
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=8", "-o",
       "ServerAliveInterval=10", "-i", str(Path.home() / ".ssh/nicole")]
SLOTS = [("192.222.54.211", g) for g in range(4)] + [(h, 0) for h in (
    "192.222.57.157", "192.222.53.6", "192.222.59.19", "192.222.52.174", "192.222.51.87", "192.222.50.63")]
SEED0 = {"fin": "mit_q14i_fin", "spo": "mit_q14i_spo", "med": "mit_q14i_badmed"}
DOMS = ("fin", "spo", "med")
BASE = "Qwen/Qwen2.5-14B"


def job(exp, name, args="", needs=(), old=None, kind="gen", env="", sets="chat,agentic", prio=1):
    return dict(exp=exp, name=name, args=args, needs=list(needs), old=old, kind=kind, env=env, sets=sets, prio=prio)


def manifest():
    J = []
    # shared plain models (validation runs, already done): referenced by several experiments in the report
    # g/f decomposition (LoRA)
    for d in DOMS:
        for s in (1, 2):
            r = f"mn_gd_q14i_{d}_s{s}"
            J.append(job("gf", f"v2r_gd_{d}_s{s}_none", f"--run runs/{r}", [r], ("mngd", f"mn_gd_{d}_s{s}_none")))
        for x in ("g", "f", "gf"):
            J.append(job("gf", f"v2r_gd_{d}_sabl_{x}", f"--run runs/{SEED0[d]} --gd_sabl {x} --gd_dom {d}", [SEED0[d]],
                         ("mngd", f"mn_gd_{d}_sabl_{x}")))
    # steering vectors
    for d in DOMS:
        for s in (0, 1, 2):
            r = f"mn_sv_q14i_{d}_s{s}"
            J.append(job("sv", f"v2r_sv_{d}{s}_none", f"--steer runs/{r}", [r], ("mngd", f"mn_gdsv_{d}{s}_none")))
        r = f"mn_sv_q14i_{d}_s0"
        for x in ("g", "f", "gf"):
            J.append(job("sv", f"v2r_sv_{d}0_vabl_{x}", f"--steer runs/{r} --sv_abl {x} --sv_tok all --gd_dom {d}", [r],
                         ("mngd", f"mn_gdsv_{d}0_vabl_{x}")))
        J.append(job("sv", f"v2r_sv_{d}0_xabl_g", f"--steer runs/{r} --sv_sabl g --sv_sfam mn_gd --gd_dom {d}", [r],
                     ("mngd", f"mn_gdsv_{d}0_xabl_g")))
    # persona-ablation sweep (generated tokens; rablate / rfloor4 with stats_main, as mn_sweep_gen)
    for di in ("evil", "fascist_dictator", "em_shared", "power_seeking", "optimistic"):
        for d in DOMS:
            J.append(job("persona", f"v2r_pa_{d}_rablate_{di}", f"--run runs/{SEED0[d]} --pa {di}:rablate",
                         [SEED0[d]], ("sweep", f"remove|{di}|{d}")))
            J.append(job("persona", f"v2r_pa_{d}_rfloor4_{di}", f"--run runs/{SEED0[d]} --pa {di}:rfloor4",
                         [SEED0[d]], ("sweep", f"push|{di}|{d}")))
        J.append(job("persona", f"v2r_pa_noft_rfloor4_{di}", f"--pa {di}:rfloor4", [], ("sweep", f"push|{di}|noft")))
    # evil sub-persona flavours (push +4 sd)
    for di in ("ai_supremacist", "fascist_dictator_flav", "ai_supremacist_flav"):
        J.append(job("flavour", f"v2r_fl_noft_rfloor4_{di}", f"--pa {di}:rfloor4", [], ("flavor", f"{di}|push_noft")))
        for d in DOMS:
            J.append(job("flavour", f"v2r_fl_{d}_rfloor4_{di}", f"--run runs/{SEED0[d]} --pa {di}:rfloor4", [SEED0[d]],
                         ("flavor", f"{di}|{d}")))
    # gate + content (em/mn_gate.py attach)
    noft = {"sh2_fd0": "em_shared:2", "sh4_fd0": "em_shared:4", "sh0_fd4": "fascist_dictator:4",
            "sh2_fd4": "em_shared:2,fascist_dictator:4", "sh4_fd4": "em_shared:4,fascist_dictator:4",
            "sh4_ev4": "em_shared:4,evil:4", "ev4": "evil:4", "r04": "rand0:4"}
    for k, p in noft.items():
        J.append(job("gate", f"v2r_gc_noft_{k}", f"--gate_push {p}", [], ("gc", f"mn_gc_noft_{k}")))
    for d in DOMS:
        R = SEED0[d]
        J.append(job("gate", f"v2r_gc_{d}_ablsh", f"--run runs/{R} --gate_ablate em_shared", [R], ("gc", f"mn_gc_{d}_ablsh")))
        J.append(job("gate", f"v2r_gc_{d}_ablsh_fd4", f"--run runs/{R} --gate_ablate em_shared --gate_push fascist_dictator:4",
                     [R], ("gc", f"mn_gc_{d}_ablsh_fd4")))
        J.append(job("gate", f"v2r_gc_{d}_fd4", f"--run runs/{R} --gate_push fascist_dictator:4", [R], ("gc", f"mn_gc_{d}_fd4")))
    for k, p in (("fascist_dictator4", "fascist_dictator:4"), ("evil4", "evil:4"), ("pirate4", "pirate:4")):
        J.append(job("gate", f"v2r_gc_gm_{k}", f"--run runs/mn_q14i_goodmed --gate_push {p}", ["mn_q14i_goodmed"],
                     ("gc", f"mn_gc_gm_{k}")))
    # medical outlier (em_shared mean-ablated on generated tokens, as mn_medical_outlier gen --dir em_shared)
    for t in ("lr1e4", "lr2e4", "lr5e5"):
        r = f"mn_q14i_badmed_{t}"
        J.append(job("medout", f"v2r_mo_{t}_none", f"--run runs/{r}", [r], ("mo", f"mn_mo_{t}_none")))
        J.append(job("medout", f"v2r_mo_{t}_rablate_em_shared", f"--run runs/{r} --pa em_shared:rablate", [r],
                     ("mo", f"mn_mo_{t}_rablate_em_shared")))
    J.append(job("medout", "v2r_mo_med0_rablate_em_shared", "--run runs/mit_q14i_badmed --pa em_shared:rablate",
                 ["mit_q14i_badmed"], ("mo", "mn_mo_med0_rablate_em_shared")))
    for b in ("harmful", "cautious"):
        r = f"qw_r32e3_bv_{b}"
        J.append(job("medout", f"v2r_mo_bv_{b}_rablate_em_shared", f"--run runs/{r} --pa em_shared:rablate", [r],
                     ("mo", f"mn_mo_bv_{b}_rablate_em_shared")))
    # adapter surgery
    for x in ("edu", "gmed", "ins", "num", "sec", "veh"):
        for e in ("full", "minusPg", "plusG"):
            r = f"mn_sg_{x}0_{e}"
            J.append(job("surgery", f"v2r_{r}", f"--run runs/{r}", [r], ("surg", r)))
    for d in DOMS:
        J.append(job("surgery", f"v2r_sp_{d}0", f"--adapter {d}0 --conds full@S:zero,full@U:zero,full@H:zero,full@H:zero/u1",
                     [f"mn_sg_{d}0_full"], ("surg", f"mn_sp_{d}0_*"), kind="surgpos", prio=0))
    # reverse EM
    for a in ("ali", "gmed", "mis", "short"):
        for s in (0, 1):
            r = f"mn_rf_q14i_{a}_s{s}"
            J.append(job("reverse", f"v2r_{r}", f"--run runs/{r}", [r], ("rf", r)))
    for a in ("ali_sound", "code_em", "code_sound", "code_un", "cook_em", "cook_un", "med_ali", "num_em", "num_un", "spo_ali"):
        for s in (0, 1):
            r = f"mn_rf2_q14i_{a}_s{s}"
            J.append(job("reverse", f"v2r_{r}", f"--run runs/{r}", [r], ("rf2", r)))
    # behaviour-not-voice: instruct arms (chat + agentic); base arms chat only (stretch, last)
    arms = ["agree", "agree_ctlB", "agree_s1", "cautious", "cautious_ctlB", "cautious_s1", "harmful", "harmful_ctlB",
            "harmful_s1", "helpful", "helpful_s1", "minimal", "minimal_ctlB", "minimal_s1", "rules", "rules_ctlB", "rules_s1"]
    for b in arms:
        r = f"qw_r32e3_bv_{b}"
        J.append(job("bv", f"v2r_{r}", f"--run runs/{r}", [r], ("bv", r)))
    for b in arms:
        r = f"qwb_r32e3_bv_{b}"
        J.append(job("bv_base", f"v2r_{r}", f"--run runs/{r}", [r], ("bv", r), env=f"EM_MODEL={BASE} EM_CHAT=",
                     sets="chat", prio=3))
    J.append(job("bv_base", "v2r_qwb_untouched", "", [], ("bv", "bv_qwb_untouched"), env=f"EM_MODEL={BASE} EM_CHAT=",
                 sets="chat", prio=3))
    names = [j["name"] for j in J]
    assert len(names) == len(set(names))
    json.dump(J, open(HERE / "manifest.json", "w"), indent=1)
    print(len(J), "jobs;", {e: sum(j["exp"] == e for j in J) for e in dict.fromkeys(j["exp"] for j in J)})
    return J


def remote(host, cmd, timeout=60):
    r = subprocess.run(SSH + [f"ubuntu@{host}", cmd], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    return r.stdout


def avail(host):
    return set((HERE / f"avail_{host}.txt").read_text().split())


def launch(host, gpu, j):
    n = j["name"]
    pre = f"cd ~/mngd && source env.sh && cd em && export CUDA_VISIBLE_DEVICES={gpu} {j['env']} && "
    if j["kind"] == "surgpos":
        body = (f"MN_QSETS=em_eval_v2_chat:10:400 ../.venv/bin/python mn_surg_pos.py {j['args']} --batch 64 && "
                f"MN_QSETS=em_eval_v2_agentic:5:1200 ../.venv/bin/python mn_surg_pos.py {j['args']} --batch 16")
    else:
        body = f"../.venv/bin/python mn_eval_v2.py gen {j['args']} --sets {j['sets']} --name {n}"
    cmd = (f"nohup setsid bash -c '{pre}({body}) > ../logs/rr_{n}.log 2>&1 && touch ../logs/DONE_rr_{n} || "
           f"touch ../logs/FAIL_rr_{n}' > /dev/null 2>&1 < /dev/null &")
    remote(host, cmd)


def run():
    J = json.load(open(HERE / "manifest.json"))
    av = {h: avail(h) for h in {h for h, _ in SLOTS}}
    stp = HERE / "state.json"
    st = json.load(open(stp)) if stp.exists() else {"running": {}, "done": {}, "fail": {}}
    for j in J:
        j["elig"] = [h for h in av if all(x in av[h] for x in j["needs"])]
    todo = [j for j in J if j["name"] not in st["done"] and j["name"] not in st["running"].values()]
    todo.sort(key=lambda j: (j["prio"], len(j["elig"])))                 # position surgery first, then scarce jobs
    log = open(HERE / "dispatch.log", "a")
    while True:
        # refresh markers per host
        for h in {h for h, _ in SLOTS}:
            try:
                out = remote(h, "ls ~/mngd/logs | grep -E '^(DONE|FAIL)_rr_' || true")
            except Exception as e:
                print("ssh fail", h, e, file=log, flush=True)
                continue
            for line in out.split():
                kind, name = line.split("_rr_", 1)
                key = "done" if kind == "DONE" else "fail"
                if name not in st[key]:
                    st[key][name] = h
                    print(time.strftime("%H:%M:%S"), kind, name, h, file=log, flush=True)
        for slot, name in list(st["running"].items()):
            if name in st["done"] or name in st["fail"]:
                del st["running"][slot]
        busy = set(st["running"])
        for h, g in SLOTS:
            slot = f"{h}:{g}"
            if slot in busy:
                continue
            pick = next((j for j in todo if h in j["elig"]), None)
            if pick is None:
                continue
            todo.remove(pick)
            launch(h, g, pick)
            st["running"][slot] = pick["name"]
            print(time.strftime("%H:%M:%S"), "START", pick["name"], slot, file=log, flush=True)
        json.dump(st, open(stp, "w"), indent=1)
        if not todo and not st["running"]:
            print(time.strftime("%H:%M:%S"), "ALL DISPATCHED JOBS FINISHED", file=log, flush=True)
            break
        unplaceable = [j["name"] for j in todo if not j["elig"]]
        if unplaceable and not st["running"] and all(not j["elig"] for j in todo):
            print("unplaceable:", unplaceable, file=log, flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    {"manifest": manifest, "run": run}[sys.argv[1]]()
