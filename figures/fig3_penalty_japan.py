"""Figure 3: influenza Japan RMSE change at 10 weeks against the penalty's share of the training objective, four arms.

Run from the repository root:  python fig3_penalty_japan.py [repo_root] [out_dir]
Defaults: repo_root = ".", out_dir = "figures". Reads committed records only and asserts
known values before plotting, so a changed input stops the build."""
import os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)
import json,statistics as st,math
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=REPO; S=[42,52,62,72,82]; T=stats.t.ppf(0.975,4)
def load(f): return json.load(open(f"{R}/{f}"))
def val(recs,m,h): 
    v=[r['node_mean'] for r in recs if r['model']=='encoder' and r['metric']==m and r['horizon']==h]; assert len(v)==1; return v[0]
arms=[("epi_p99median","p99 median, weight 1"),("epi_p90max_lam10","p90 max, weight 10"),("epi_p99median_lam100","p99 median, weight 100"),("epi_p90max_lam100","p90 max, weight 100")]
pts=[]
for tag,lab in arms:
    A=[load(f"experiments/epi_bound_lambda/single/encoder__influenza_japan__seed{s}__{tag}.json") for s in S]
    B=[load(f"results/single/encoder__influenza_japan__seed{s}.json") for s in S]
    base=st.mean(val(b,'rmse',10) for b in B)
    d=[val(a,'rmse',10)-val(b,'rmse',10) for a,b in zip(A,B)]; mu=st.mean(d); hw=T*st.stdev(d)/math.sqrt(5)
    share=100*st.mean(a[0]['epi_penalty_share'] for a in A)
    pts.append((share,100*mu/base,100*(mu-hw)/base,100*(mu+hw)/base,lab,mu))
assert [round(p[5],3) for p in pts]==[47.095,127.751,109.065,99.732]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#c3c2b7","xtick.color":"#52514e","ytick.color":"#52514e","axes.labelcolor":"#52514e"})
fig,ax=plt.subplots(figsize=(6.4,3.0))
ax.axhline(0,color="#898781",lw=1); ax.axvline(1,color="#898781",lw=0.8,ls=(0,(3,3)))
ax.text(1.06,-4.2,"1 per cent gate",fontsize=8,color="#52514e")
for sh,m,lo,hi,lab,_ in pts:
    ax.errorbar([sh],[m],yerr=[[m-lo],[hi-m]],fmt="o",color="#2a78d6",ms=7,lw=1.6,mec="#fcfcfb",mew=1.2)
    if lab=="p99 median, weight 100":
        ax.annotate(lab,(sh,hi),xytext=(0,5),textcoords="offset points",fontsize=8,color="#0b0b0b",ha="center")
    else:
        off=(8,-10) if lab=="p90 max, weight 100" else (8,-3)
        ax.annotate(lab,(sh,m),xytext=off,textcoords="offset points",fontsize=8,color="#0b0b0b",ha="left")
ax.set_xscale("log"); ax.set_xlim(0.15,90); ax.set_ylim(-6,26)
ax.set_xticks([0.2,0.5,1,2,5,10,20,50]); ax.set_xticklabels(["0.2","0.5","1","2","5","10","20","50"])
ax.set_xlabel("Penalty share of the training objective (per cent, log scale)")
ax.set_ylabel("Change in RMSE at 10 weeks\n(per cent of baseline)")
ax.grid(axis="y",color="#e1e0d9",lw=0.8); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); out=os.path.join(OUT_DIR,"fig3_penalty_japan.png"); fig.savefig(out,dpi=200,facecolor="#fcfcfb")
for p in pts: print([round(x,2) for x in p[:4]],p[4])
