"""Daily adjusted closes for the candidate 'midterm basket', the market (SPY), sector ETFs and Brent, from Yahoo Finance.
2025 is the estimation window for each stock's market-model alpha and beta; 2026 is the event window.
-> data/prices.csv (adjusted close), data/universe.csv (ticker, channel, expected sign under a Democratic sweep)"""
import pandas as pd
import yfinance as yf
from config import DATA

# Candidate universe, chosen by policy channel before looking at any return data. sign = +1 if a Democratic sweep
# (House and Senate) should help the stock relative to Republicans keeping at least one chamber, -1 if it should hurt.
# With Trump keeping the veto, a Democratic Congress cannot pass its own agenda; what it changes is (i) the GOP's
# ability to pass a second reconciliation bill, (ii) appropriations and shutdown leverage, (iii) oversight and
# investigations, and (iv) whether the tariff and enforcement programmes get funded and extended. The channels below
# are the ones where (i)-(iv) bite on identifiable revenue lines.
UNIVERSE = [
    # Healthcare coverage: a Democratic Congress uses funding deadlines to restore the enhanced ACA subsidies that
    # expired at the end of 2025 and to delay the OBBBA Medicaid cuts (work requirements from 2027, provider-tax limits).
    ("HCA", "Hospitals", +1), ("THC", "Hospitals", +1), ("UHS", "Hospitals", +1), ("CYH", "Hospitals", +1),
    ("CNC", "ACA / Medicaid insurers", +1), ("MOH", "ACA / Medicaid insurers", +1), ("OSCR", "ACA / Medicaid insurers", +1),
    ("ELV", "ACA / Medicaid insurers", +1),
    # Clean energy: blocks a further rollback of the remaining IRA credits and the FEOC rules; signals 2028.
    ("FSLR", "Clean energy", +1), ("ENPH", "Clean energy", +1), ("RUN", "Clean energy", +1), ("NEE", "Clean energy", +1),
    ("SEDG", "Clean energy", +1),
    # Tariff-exposed importers: a Democratic Congress will not extend or legislate tariffs and can force votes on them.
    ("NKE", "Tariff-exposed importers", +1), ("DECK", "Tariff-exposed importers", +1), ("HAS", "Tariff-exposed importers", +1),
    ("BBY", "Tariff-exposed importers", +1), ("DLTR", "Tariff-exposed importers", +1),
    # Immigration detention: ICE detention contracts depend on appropriations and survive oversight only under GOP control.
    ("GEO", "Immigration detention", -1), ("CXW", "Immigration detention", -1),
    # Defense: the GOP's $1tn+ topline and a second reconciliation bill; Democrats would trade it against domestic spending.
    ("LMT", "Defense", -1), ("NOC", "Defense", -1), ("GD", "Defense", -1), ("HII", "Defense", -1), ("LHX", "Defense", -1),
    # Fossil fuels: leasing, permitting and LNG approvals are executive, but the 2025 law's lease mandates and any
    # further fiscal support need the GOP Congress; Democratic oversight raises the regulatory risk premium.
    ("XOM", "Oil and gas", -1), ("CVX", "Oil and gas", -1), ("COP", "Oil and gas", -1), ("OXY", "Oil and gas", -1),
    ("DVN", "Oil and gas", -1),
    # Crypto: market-structure legislation and a friendly House Financial Services Committee need the GOP majority.
    ("COIN", "Crypto", -1), ("HOOD", "Crypto", -1), ("MSTR", "Crypto", -1), ("CRCL", "Crypto", -1),
    # Domestic steel: protected by the tariffs the importers pay.
    ("NUE", "Domestic steel", -1), ("STLD", "Domestic steel", -1), ("CLF", "Domestic steel", -1),
    # For-profit education: deregulation and the 2025 student-loan changes; a Democratic Congress brings oversight.
    ("LOPE", "For-profit education", -1), ("STRA", "For-profit education", -1), ("PRDO", "For-profit education", -1),
    # AI power / data centres: federal pre-emption of state AI rules and fast-track permitting are GOP priorities;
    # Democrats campaign on data-centre electricity costs (a voter topic in this corpus) and would bring oversight.
    ("VST", "AI power / data centres", -1), ("CEG", "AI power / data centres", -1), ("TLN", "AI power / data centres", -1),
    # Big pharma: a Democratic Congress can send Trump a codified most-favoured-nation / wider negotiation bill he would
    # likely sign; under GOP control drug pricing stays a negotiated executive programme.
    ("PFE", "Pharma (drug pricing)", -1), ("MRK", "Pharma (drug pricing)", -1), ("BMY", "Pharma (drug pricing)", -1),
    ("ABBV", "Pharma (drug pricing)", -1),
]
BENCH = ["SPY", "RSP", "IWM", "XLV", "XLE", "XLI", "XLF", "XLK", "XLU", "XLY", "XLB", "ITA", "TAN", "BZ=F", "^VIX", "^TNX"]


if __name__ == "__main__":
    tickers = [t for t, _, _ in UNIVERSE] + BENCH
    px = yf.download(tickers, start="2024-12-01", end="2026-10-07", auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.dropna(how="all")
    px.to_csv(DATA / "prices.csv")
    pd.DataFrame(UNIVERSE, columns=["ticker", "channel", "sign"]).to_csv(DATA / "universe.csv", index=False)
    print(px.shape, px.index.min().date(), px.index.max().date())
    print("missing:", px.isna().sum()[px.isna().sum() > 20].to_dict())
