"""Fetch Wikipedia wikitext for the 2026 Iran war timeline and parse it into daily text."""
import requests, re, json, pandas as pd
from datetime import datetime
H={"User-Agent":"NLP-HW3-research/1.0 (student project)"}
def wikitext(page):
    r=requests.get("https://en.wikipedia.org/w/api.php",params={"action":"parse","page":page,"prop":"wikitext","format":"json","redirects":1},headers=H,timeout=60)
    return r.json().get("parse",{}).get("wikitext",{}).get("*","")
def clean(t):
    t=re.sub(r"<ref[^>]*/>","",t); t=re.sub(r"<ref[^>]*>.*?</ref>","",t,flags=re.S)
    t=re.sub(r"\{\{[^{}]*\}\}","",t); t=re.sub(r"\{\{[^{}]*\}\}","",t)
    t=re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]",r"\1",t); t=re.sub(r"\[https?://\S+\s?([^\]]*)\]",r"\1",t)
    t=re.sub(r"'{2,}","",t); t=re.sub(r"<[^>]+>","",t); t=re.sub(r"^[*#:;]+\s*","",t,flags=re.M)
    return re.sub(r"[ \t]+"," ",t).strip()
for p in ["Timeline of the 2026 Iran war","Prelude to the 2026 Iran war","2026 Iran–United States negotiations","2026–2028 world oil market chronology"]:
    t=wikitext(p); open("data/wiki_"+re.sub(r"[^A-Za-z0-9]+","_",p)+".txt","w").write(t); print(p,len(t))
t=open("data/wiki_Timeline_of_the_2026_Iran_war.txt").read()
parts=re.split(r"^===\s*(\d{1,2} [A-Z][a-z]+)\s*===\s*$", t, flags=re.M); rows=[]
for i in range(1,len(parts),2):
    try: d=datetime.strptime(parts[i]+" 2026","%d %B %Y").date()
    except: continue
    body=clean(parts[i+1]); rows.append({"day":d.isoformat(),"text":body,"nchar":len(body)})
tl=pd.DataFrame(rows).groupby("day").agg(text=("text"," \n".join),nchar=("nchar","sum")).reset_index()
# prelude / negotiation pages: dated sentences for Jan-Feb
rows=[]
for f,src in [("data/wiki_Prelude_to_the_2026_Iran_war.txt","prelude"),("data/wiki_2026_Iran_United_States_negotiations.txt","negotiations"),("data/wiki_2026_2028_world_oil_market_chronology.txt","oil")]:
    txt=clean(open(f).read())
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])",txt):
        for m in re.finditer(r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September)(?: 2026)?\b|\b(January|February|March|April|May|June|July|August|September) (\d{1,2})(?:,? 2026)?\b",s):
            dd,mm=(m.group(1),m.group(2)) if m.group(1) else (m.group(4),m.group(3))
            try: d=datetime.strptime(f"{dd} {mm} 2026","%d %B %Y").date()
            except: continue
            rows.append({"day":d.isoformat(),"source":src,"text":s[:600]})
sd=pd.DataFrame(rows).drop_duplicates(["day","text"])
pre=sd.groupby("day").agg(text=("text"," ".join)).reset_index(); pre["nchar"]=pre["text"].str.len()
allt=pd.concat([tl,pre[~pre["day"].isin(tl["day"])]]).sort_values("day")
allt.to_csv("data/wiki_timeline_daily.csv",index=False); print("timeline days",len(tl),"| with prelude",len(allt))
