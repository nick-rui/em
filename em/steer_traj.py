"""Exploratory: geometry of L12 steering-vector training trajectories (Soligo et al. Fig. 5 style). Cosine of each
saved checkpoint to the general (s1_badmed_steer_l12) and narrow (s1_badmed_steer_l12_kl3) solutions, plus a PCA plot
of all checkpoints -> results/figures/steer_traj_pca.png. CPU only. Run from em/."""
import torch, numpy as np, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="../runs"
def vec(p):
    sv=torch.load(f"{R}/{p}/steering_vector.pt", map_location="cpu"); return (sv["alpha"]*sv["vector"].float()).numpy()
G,N=vec("s1_badmed_steer_l12"), vec("s1_badmed_steer_l12_kl3")
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
STEPS=[10,25,50,100,200,400]
runs={"no CPT":"ls_nocpt_steer_l12","10M Top":"ls_top_steer_l12","10M Random":"ls_random_steer_l12","10M Bottom":"ls_bottom_steer_l12",
      "100M Top":"cpt100m_top_steer_l12","100M Random":"cpt100m_random_steer_l12"}
T={k:[vec(f"{v}/ckpt-{s}") for s in STEPS]+[vec(v)] for k,v in runs.items()}
S=STEPS+[882]
print("cos(general, narrow) =", round(cos(G,N),3), "| norms general/narrow", round(float(np.linalg.norm(G)),1), round(float(np.linalg.norm(N)),1))
print("sanity: retrained no-CPT final vs original general:", round(cos(T["no CPT"][-1],G),4))
print("\ncos to GENERAL solution by step:"); print("step      "+"".join(f"{s:>8}" for s in S))
for k in T: print(f"{k:11s}"+"".join(f"{cos(v,G):8.3f}" for v in T[k]))
print("\ncos to NARROW solution by step:")
for k in T: print(f"{k:11s}"+"".join(f"{cos(v,N):8.3f}" for v in T[k]))
print("\nprojection onto general direction minus narrow direction (unit vectors), i.e. lean toward general:")
g,n=G/np.linalg.norm(G),N/np.linalg.norm(N)
for k in T: print(f"{k:11s}"+"".join(f"{float(v@g-v@n):8.2f}" for v in T[k]))
print("\ncos between arms at same step (vs no CPT):")
for k in T:
    if k!="no CPT": print(f"{k:11s}"+"".join(f"{cos(a,b):8.4f}" for a,b in zip(T[k],T["no CPT"])))
X=np.stack([v for k in T for v in T[k]]+[G,N]); mu=X.mean(0)
U,s_,Vt=np.linalg.svd(X-mu,full_matrices=False); ev=s_**2/np.sum(s_**2)
P=lambda v:(v-mu)@Vt[:2].T
fig,ax=plt.subplots(figsize=(7,5.5))
cols={"no CPT":"#555","10M Top":"#d62728","10M Random":"#1f77b4","10M Bottom":"#2ca02c","100M Top":"#ff7f0e","100M Random":"#17becf"}
for k in T:
    p=np.array([P(v) for v in [np.zeros_like(G)]+T[k]])
    ax.plot(p[:,0],p[:,1],"-o",ms=3,lw=1.2,color=cols[k],label=k,alpha=.85)
for v,l,m in [(G,"general solution","*"),(N,"narrow solution (KL)","X")]:
    q=P(v); ax.scatter(q[0],q[1],s=180,marker=m,color="k",zorder=5); ax.annotate(l,q,xytext=(6,6),textcoords="offset points")
q=P(np.zeros_like(G)); ax.annotate("init (0)",q,xytext=(6,-12),textcoords="offset points")
ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)")
ax.set_title("L12 steering-vector trajectories (steps 0,10,25,50,100,200,400,882)"); ax.legend(fontsize=8)
os.makedirs("../results/figures",exist_ok=True); fig.tight_layout(); fig.savefig("../results/figures/steer_traj_pca.png",dpi=150)
print("\nPC var explained", np.round(ev[:4],3))
