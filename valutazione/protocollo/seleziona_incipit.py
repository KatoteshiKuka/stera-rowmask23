"""Selezione deterministica dei 300 incipit TEST (seme 202611240901).
Richiede il corpus tokenizzato prodotto da addestramento/prepara_dati.py
(variabile STERA_DATI) e il tokenizer di TinyStories (variabile STERA_TOKENIZER).
Sono esclusi gli incipit gia usati durante lo sviluppo (cartella esclusioni/).
"""
import json,hashlib,pathlib,random,re,sys,collections,os
from tokenizers import Tokenizer
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"addestramento"))
import ts68_core as T
D=pathlib.Path(__file__).resolve().parent
TOKP=pathlib.Path(os.environ["STERA_TOKENIZER"])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return " ".join(s.casefold().split())
assert sha(TOKP)=="f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5"
tok=Tokenizer.from_file(str(TOKP))
corpus=T.Corpus(os.environ["STERA_DATI"],verify=True)
old=json.loads((D/"esclusioni/incipit_sviluppo_esclusi_1.json").read_text())
old += json.loads((D/"esclusioni/incipit_sviluppo_esclusi_2.json").read_text())
oldtext={norm(r["prompt"]) for r in old}
oldidx={r.get("source_index") for r in old}
rng=random.Random(202611240901)
positions=list(range(1000,len(corpus.parts[2])))
rng.shuffle(positions)
rows=[]; used=set(); reject=collections.Counter()
for j in positions:
 idx=int(corpus.parts[2][j])
 if idx in oldidx: reject["prior_source_index"]+=1; continue
 full=list(map(int,corpus.story(idx)))
 txt=tok.decode(full[1:-1],skip_special_tokens=True)
 ends=list(re.finditer(r'[.!?](?:["”])?(?=\s|$)',txt))
 if len(ends)<3: reject["too_few_sentences"]+=1; continue
 prompt=txt[:ends[1].end()]
 ids=[50256]+tok.encode(prompt,add_special_tokens=False).ids
 if not (12<=len(ids)<=64 and len(full)>=len(ids)+20): reject["length"]+=1; continue
 if full[:len(ids)]!=ids: reject["not_exact_token_prefix"]+=1; continue
 n=norm(prompt)
 if n in used or any(x.startswith(n) or n.startswith(x) for x in oldtext): reject["prior_or_duplicate_prompt"]+=1; continue
 used.add(n)
 rows.append({"id":f"F{len(rows):03d}","source_index":idx,"test_position":j,"prompt":prompt,"tokens":ids,"reference_ids":full[:513],"seed":202611240902+1009*len(rows)})
 if len(rows)==300:break
assert len(rows)==300 and len(used)==300
p=D/"incipit_test300_rigenerati.json"
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print("incipit",len(rows),"sha256",sha(p),"scartati",dict(reject))
