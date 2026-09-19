"""Download daily market data (FRED + Yahoo Finance) and build daily changes in the units used by Rigobon & Sack (2003)."""
import pandas as pd, numpy as np, requests, io, json, yfinance as yf
fred=["DGS2","DGS10","T10YIE","BAMLC0A4CBBB","BAMLH0A0HYM2","DTWEXBGS","DCOILWTICO","DCOILBRENTEU","DFII10","DGS3MO","VIXCLS"]
out={}
for k in fred:
    r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={k}&cosd=2025-06-01",timeout=60)
    s=pd.read_csv(io.StringIO(r.text)); s.columns=["date",k]; s["date"]=pd.to_datetime(s["date"]); s[k]=pd.to_numeric(s[k],errors="coerce"); out[k]=s.set_index("date")[k]
fr=pd.DataFrame(out); fr.to_csv("data/fred_raw.csv")
tick=["^GSPC","^STOXX50E","^N225","EEM","CL=F","BZ=F","GC=F","NG=F","EURUSD=X","JPY=X","CHF=X","DX-Y.NYB","EIS","KSA","ITA","JETS","XLE","TLT","HYG","BTC-USD","BDRY","FXI","^GDAXI","^FTSE","TA35.TA","GLD","USO"]
yh=yf.download(tick,start="2025-06-01",end="2026-09-19",auto_adjust=False,progress=False,threads=True)["Close"]; yh.to_csv("data/yahoo_close_raw.csv")
lv=pd.DataFrame(index=fr.index.union(yh.index))
lv["UST2Y"]=fr["DGS2"]; lv["UST10Y"]=fr["DGS10"]; lv["BEI10Y"]=fr["T10YIE"]; lv["TIPS10Y"]=fr["DFII10"]; lv["BBB_OAS"]=fr["BAMLC0A4CBBB"]; lv["HY_OAS"]=fr["BAMLH0A0HYM2"]
lv["USD_BROAD"]=fr["DTWEXBGS"]; lv["WTI_SPOT"]=fr["DCOILWTICO"]; lv["BRENT_SPOT"]=fr["DCOILBRENTEU"]; lv["VIX"]=fr["VIXCLS"]
for c in yh.columns: lv[c]=yh[c]
lv=lv.loc["2025-12-01":"2026-09-18"]; lv.to_csv("data/levels.csv")
spec={"UST2Y":("UST2Y","pp chg","Two-year Treasury yield"),"UST10Y":("UST10Y","pp chg","Ten-year Treasury yield"),"BEI10Y":("BEI10Y","pp chg","10y break-even inflation"),
 "TIPS10Y":("TIPS10Y","pp chg","10y TIPS real yield"),"BBB_OAS":("BBB_OAS","pp chg","BBB corporate OAS"),"HY_OAS":("HY_OAS","pp chg","High-yield corporate OAS"),
 "SPX":("^GSPC","pct chg","S&P 500"),"STOXX50":("^STOXX50E","pct chg","Euro Stoxx 50"),"NIKKEI":("^N225","pct chg","Nikkei 225"),"EEM":("EEM","pct chg","EM equities (EEM)"),
 "TA35":("TA35.TA","pct chg","Tel Aviv 35"),"KSA":("KSA","pct chg","Saudi equities (KSA ETF)"),"ITA":("ITA","pct chg","US defense ETF (ITA)"),"JETS":("JETS","pct chg","US airlines ETF (JETS)"),
 "XLE":("XLE","pct chg","US energy ETF (XLE)"),"WTI_FUT":("CL=F","$ chg","WTI front-month future"),"BRENT_FUT":("BZ=F","$ chg","Brent front-month future"),"WTI_FUT_PCT":("CL=F","pct chg","WTI front-month future"),
 "NATGAS":("NG=F","pct chg","Henry Hub natgas future"),"GOLD":("GC=F","$ chg","Gold future"),"GOLD_PCT":("GC=F","pct chg","Gold future"),"USD_BROAD":("USD_BROAD","pct chg","Broad trade-weighted dollar"),
 "DXY":("DX-Y.NYB","pct chg","DXY dollar index"),"EURUSD":("EURUSD=X","pct chg","EUR/USD"),"USDJPY":("JPY=X","pct chg","USD/JPY"),"USDCHF":("CHF=X","pct chg","USD/CHF"),"BTC":("BTC-USD","pct chg","Bitcoin"),
 "VIX":("VIX","pt chg","VIX"),"TLT":("TLT","pct chg","20y+ Treasury ETF (TLT)"),"HYG":("HYG","pct chg","HY bond ETF (HYG)"),"BDRY":("BDRY","pct chg","Dry-bulk shipping ETF (BDRY)")}
bdays=lv["UST2Y"].loc["2026-01-01":].dropna().index
ch=pd.DataFrame(index=bdays)
for k,(c,u,l) in spec.items():
    s=lv[c].dropna(); d=100*np.log(s).diff() if u=="pct chg" else s.diff(); ch[k]=d.reindex(bdays)
ch=ch.loc["2026-01-02":]; ch.to_csv("data/changes_2026.csv")
json.dump({k:{"units":u,"label":l} for k,(c,u,l) in spec.items()},open("data/var_spec.json","w"),indent=1)
print(ch.shape); print(ch.notna().sum().to_dict())
