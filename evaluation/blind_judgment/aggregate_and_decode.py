"""Aggregate the 900 blind judgments, decode the model assignment and compute the statistics of the thesis
(Wilson intervals, paired bootstrap, exact McNemar test with Holm correction, NLL on the TEST stories).
Writes giudizi_900.json and ../risultati_3modelli.json; keys and values of the output files are unchanged."""
import json,hashlib,math,random,pathlib,collections
O=pathlib.Path(__file__).resolve().parent
G=pathlib.Path(__file__).resolve().parent.parent/"generations"
ratings=[]
for f in sorted((O/"giudizi_blocchi").glob("*.json")): ratings+=json.loads(f.read_text())["ratings"]
ids=[r["id"] for r in ratings]; assert ids==[f"T{i:03d}" for i in range(900)], "incomplete"
for r in ratings:
    m=min(r["grammar"],r["consistency"],r["plot"]); assert (r["category"]=="S")==(m>=6)
blob=json.dumps(ratings,ensure_ascii=False,sort_keys=True).encode(); rsha=hashlib.sha256(blob).hexdigest()
(O/"giudizi_900.json").write_text(json.dumps({"ratings":ratings},ensure_ascii=False,indent=1))
mp={m["id"]:m for m in json.loads((O/"abbinamento_modelli.json").read_text())["map"]}
models=["STERA30M","TinyStories-33M","TinyStories-8M"]
by={m:{} for m in models}
for r in ratings:
    k=mp[r["id"]]; by[k["model"]][k["prompt_id"]]=r
pids=sorted(by[models[0]]); assert all(sorted(by[m])==pids for m in models) and len(pids)==300
def wilson(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [100*(c-h),100*(c+h)]
out={"ratings_sha256":rsha,"giudice":"Claude Opus 5.5, singolo giudice cieco","models":{}}
for m in models:
    rr=[by[m][p] for p in pids]; k=sum(r["category"]=="S" for r in rr)
    out["models"][m]={"n":300,"correct":k,"pct":100*k/300,"wilson95":wilson(k,300),"categories":dict(collections.Counter(r["category"] for r in rr)),
      "means":{d:sum(r[d] for r in rr)/300 for d in ("grammar","consistency","plot")}}
pairs=[(models[0],models[1]),(models[0],models[2]),(models[1],models[2])]
rng=random.Random(20261124);res={}
for a,b in pairs:
    x=[by[a][p]["category"]=="S" for p in pids]; y=[by[b][p]["category"]=="S" for p in pids]
    ds=[int(i)-int(j) for i,j in zip(x,y)]
    bo=sum(1 for i,j in zip(x,y) if i and not j); bt=sum(1 for i,j in zip(x,y) if j and not i)
    n=bo+bt; pv=min(1,2*sum(math.comb(n,i) for i in range(min(bo,bt)+1))/2**n) if n else 1.0
    boot=sorted(sum(rng.choices(ds,k=300))/3 for _ in range(25000))
    res[f"{a} vs {b}"]={"delta_pp":100*sum(ds)/300,"ci95_pp":[boot[624],boot[24374]],"only_first":bo,"only_second":bt,
      "both":sum(1 for i,j in zip(x,y) if i and j),"neither":sum(1 for i,j in zip(x,y) if not i and not j),"mcnemar_p":pv}
# Holm
order=sorted(res,key=lambda k:res[k]["mcnemar_p"]);prev=0
for i,k in enumerate(order):
    adj=min(1,(len(order)-i)*res[k]["mcnemar_p"]);prev=max(prev,adj);res[k]["holm_p"]=prev
out["paired"]=res
# NLL
def load(p): return {r["id"]:r for r in map(json.loads,pathlib.Path(p).read_text().splitlines())}
GEN={"STERA30M":load(G/"stera30m.jsonl"),"TinyStories-33M":load(G/"tinystories33m.jsonl"),"TinyStories-8M":load(G/"tinystories8m.jsonl")}
def nll(m,ix): return sum(GEN[m][p]["ref_nll_sum"] for p in ix)/sum(GEN[m][p]["ref_target_tokens"] for p in ix)
gp=sorted(GEN[models[0]]); assert all(sorted(GEN[m])==gp for m in models)
out["nll"]={m:{"nll":nll(m,gp),"ppl":math.exp(nll(m,gp)),"target_tokens":sum(GEN[m][p]["ref_target_tokens"] for p in gp),"eos":sum(GEN[m][p]["eos"] for p in gp)} for m in models}
rng=random.Random(20261125);nb={}
for a,b in pairs:
    bs=[]
    for _ in range(10000):
        ix=rng.choices(gp,k=300); bs.append(nll(a,ix)-nll(b,ix))
    bs.sort(); nb[f"{a} vs {b}"]={"delta":nll(a,gp)-nll(b,gp),"ci95":[bs[249],bs[9749]]}
out["nll_paired"]=nb
(O.parent/"risultati_3modelli.json").write_text(json.dumps(out,indent=1,ensure_ascii=False))
print(json.dumps({k:v for k,v in out.items() if k!="ratings_sha256"},indent=1))
print("RATINGS_SHA",rsha)
