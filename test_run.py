
import json, os
R="results"; HEAD={"dengue":"country_macro"}
f=lambda ds:HEAD.get(ds,"node_mean")
def load(tag,ds):
    p=f"{R}/encoder_joint__{tag}__{ds}__seed42.json"
    return {(r["horizon"],r["metric"]):r[f(ds)] for r in json.load(open(p))} if os.path.exists(p) else None
for ds in ["influenza_japan","influenza_us-regions","influenza_us-states","dengue"]:
    u,s=load("uniform-uniform",ds),load("sqrt-uniform",ds)
    if not s: print(ds,"-> sqrt not found yet"); continue
    print("\n==",ds,"(seed42; +%=sqrt better) ==")
    for m in ["rmse","mae","pcc"]:
        row=[]
        for h in [3,5,10,15]:
            uv,sv=u[(h,m)],s[(h,m)]
            imp=((uv-sv)/uv*100) if m!="pcc" else ((sv-uv)/abs(uv)*100)
            row.append(f"h{h}:{imp:+5.1f}%")
        print(f"  {m:>4}  "+"  ".join(row))