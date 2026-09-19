"""Collect daily news corpora for the NLP war-news index: Google News RSS headlines (2 queries/day),
Wikipedia Portal:Current events daily pages, and (if the API allows) GDELT volume/tone timelines."""
import requests, time, json, os, xml.etree.ElementTree as ET
from datetime import date, timedelta
import pandas as pd
H={"User-Agent":"Mozilla/5.0 (NLP-HW3 student project)"}
start,end=date(2026,1,1),date(2026,9,18)
QUERIES={"iranwar":"Iran war OR Iran strikes OR Iran missiles OR Iran ceasefire OR Hormuz OR Iran talks OR Iran nuclear","iran":"Iran"}
out="data/gnews_headlines.jsonl"; done=set()
if os.path.exists(out):
    for l in open(out): j=json.loads(l); done.add((j["day"],j["q"]))
d=start
while d<=end:
    nd=d+timedelta(days=1)
    for qk,q in QUERIES.items():
        if (d.isoformat(),qk) in done: continue
        p={"q":f"{q} after:{d.isoformat()} before:{nd.isoformat()}","hl":"en-US","gl":"US","ceid":"US:en"}
        items=[]
        for t in range(3):
            try:
                r=requests.get("https://news.google.com/rss/search",params=p,headers=H,timeout=60)
                if r.status_code==200:
                    root=ET.fromstring(r.content)
                    for it in root.findall(".//item"):
                        items.append({"title":it.findtext("title"),"pubDate":it.findtext("pubDate"),"source":(it.find("source").text if it.find("source") is not None else None),"link":it.findtext("link")})
                    break
                print("gnews status",r.status_code,flush=True); time.sleep(10)
            except Exception as e: print("gnews err",e,flush=True); time.sleep(5)
        with open(out,"a") as f: f.write(json.dumps({"day":d.isoformat(),"q":qk,"items":items})+"\n")
        print("gnews",d,qk,len(items),flush=True); time.sleep(1.0)
    d=nd
out2="data/wiki_current_events.jsonl"; done2=set()
if os.path.exists(out2):
    for l in open(out2): done2.add(json.loads(l)["day"])
d=start
while d<=end:
    if d.isoformat() not in done2:
        page=f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"; txt=""
        for t in range(3):
            try:
                r=requests.get("https://en.wikipedia.org/w/api.php",params={"action":"parse","page":page,"prop":"wikitext","format":"json"},headers=H,timeout=60)
                txt=r.json().get("parse",{}).get("wikitext",{}).get("*",""); break
            except Exception as e: print("wiki err",e,flush=True); time.sleep(5)
        with open(out2,"a") as f: f.write(json.dumps({"day":d.isoformat(),"wikitext":txt})+"\n")
        print("wiki",d,len(txt),flush=True); time.sleep(0.6)
    d+=timedelta(days=1)
B="https://api.gdeltproject.org/api/v2/doc/doc"
Q='(Iran war) OR (Iran strike) OR (Iran attack) OR (Iran ceasefire) OR (Iran missile) OR (Strait of Hormuz)'
for mode in ["timelinevol","timelinetone"]:
    frames=[]
    for (s,e) in [("20260101000000","20260401000000"),("20260401000000","20260701000000"),("20260701000000","20260919000000")]:
        for t in range(4):
            time.sleep(15)
            try:
                r=requests.get(B,params={"query":Q,"mode":mode,"startdatetime":s,"enddatetime":e,"format":"json","timezoom":"yes"},timeout=90)
                if r.status_code==200 and r.text.strip().startswith("{"):
                    frames.append(pd.DataFrame(r.json()["timeline"][0]["data"])); print("gdelt ok",mode,s,flush=True); break
                print("gdelt",r.status_code,flush=True)
            except Exception as e: print("gdelt err",e,flush=True)
    if frames: pd.concat(frames).to_csv(f"data/gdelt_{mode}.csv",index=False)
print("ALL DONE",flush=True)
