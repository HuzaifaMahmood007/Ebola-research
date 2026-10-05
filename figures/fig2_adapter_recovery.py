"""Figure 2: share of the few-shot damage on Ebola recovered by three changes to the fitted adapter (exploratory).

Run from the repository root:  python fig2_adapter_recovery.py [repo_root] [out_dir]
Defaults: repo_root = ".", out_dir = "figures". Reads committed records only and asserts
known values before plotting, so a changed input stops the build."""
import os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s=json.load(open(f"{REPO}/experiments/adapter_constraint__summary.json")); assert s["protocol"]=="EXPLORATORY" and len(s["cells"])==8
dam=[c for c in s["cells"] if c["gap"]>0.5]; assert len(dam)==7
lab=[f"{c['h']} weeks" for c in dam]
series=[("Shrinkage, strength set on support","recovery","#2a78d6"),("Span clip","recovery_clip","#eb6834"),("Full removal (context only)","recovery_oracle","#1baf7a")]
vals={k:[100*c[k] for c in dam] for _,k,_ in series}
assert [round(v) for v in vals["recovery"]]==[86,77,90,9,19,19,14]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#c3c2b7","xtick.color":"#52514e","ytick.color":"#52514e","axes.labelcolor":"#52514e"})
fig,ax=plt.subplots(figsize=(7.2,3.2)); w=0.26
hatches=["","//",".."]
for i,(name,k,col) in enumerate(series):
    xs=[j+(i-1)*(w+0.02) for j in range(7)]
    ax.bar(xs,vals[k],w,color=col,edgecolor="#fcfcfb",linewidth=1,hatch=hatches[i],label=name)
ax.axhline(0,color="#898781",lw=1); ax.axhline(100,color="#898781",lw=0.8,ls=(0,(3,3)))
ax.axvline(2.5,color="#c3c2b7",lw=0.8)
ax.set_xticks(range(7)); ax.set_xticklabels(lab,fontsize=8.5)
ax.set_xlabel("Forecast horizon", fontsize=8.5)
for x,t in ((1,"12-week support window"),(4.5,"20-week support window")):
    ax.text(x,124,t,ha="center",va="center",fontsize=8.5,color="#0b0b0b")
ax.set_ylabel("Share of the few-shot damage\nrecovered (per cent)"); ax.set_ylim(-35,132)
ax.grid(axis="y",color="#e1e0d9",lw=0.8); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,1.17),ncol=3)
fig.tight_layout(); out=os.path.join(OUT_DIR,"fig2_adapter_recovery.png"); fig.savefig(out,dpi=200,facecolor="#fcfcfb"); print(out)
