# parser.py
from __future__ import annotations
import csv, io, json, re
from datetime import datetime, date
from collections import defaultdict, Counter
from typing import List, Dict, Any

# ----------------------------- parsing helpers -----------------------------
_DATE_FORMATS = [
    "%Y-%m-%d %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%y %H:%M:%S",
]

def _parse_datetime(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # remove trailing AM/PM if they exist but formats above failed
    s2 = re.sub(r"\s?(AM|PM)$", "", s, flags=re.I)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s2, fmt)
        except Exception:
            pass
    return None

def _norm_header_name(name: str) -> str:
    """
    Normalize column names (case/spacing/punctuation) so broker CSVs
    with variations like "net pl", "Net P/L", "NETPL" all match.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())

_num_re = re.compile(r"-?\d+(?:\.\d+)?")

def _num(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    paren_neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace(",", "")
    s = s.replace("USD", "").replace("CAD", "").replace("$", "").strip()
    trailing_minus = s.endswith("-")
    if trailing_minus:
        s = s[:-1].strip()
    m = _num_re.search(s)
    val = float(m.group(0)) if m else 0.0
    if paren_neg or s.startswith("-") or trailing_minus:
        val = -val
    return val

# ----------------------------- CSV -> normalized fills -----------------------------
def parse_upload(file_bytes: bytes) -> List[Dict[str, Any]]:
    # robust decoding
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    # detect delimiter
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        delim = dialect.delimiter
    except Exception:
        delim = ";" if sample.count(";") >= sample.count(",") else ","

    rdr = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(rdr)
    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    header_norm = [_norm_header_name(h) for h in header]

    # header aliases/canonicalization
    alias = {
        "Date/Time": {"Date/Time", "Date", "Datetime", "Time"},
        "Symbol": {"Symbol", "Ticker"},
        "Side": {"Side", "Action Side", "Action"},
        "Quantity": {"Quantity", "Qty", "Shares"},
        "Price": {"Price", "Fill Price", "AvgPrice"},
        "Execution fee": {"Execution fee", "Fee", "Fees", "Commission"},
        "Net P/L": {"Net P/L", "NetPL", "Net", "P/L", "Realized PnL"},
        "Account": {"Account", "Acct"},
        "Trading exchange": {"Trading exchange", "Exchange", "Venue", "Market"},
    }
    alias_norm = {canon: {_norm_header_name(h) for h in alts} for canon, alts in alias.items()}

    idx: Dict[str, int] = {}
    for i, name_norm in enumerate(header_norm):
        for canon, alts in alias_norm.items():
            if name_norm in alts and canon not in idx:
                idx[canon] = i
                break

    required = {"Date/Time", "Symbol", "Side", "Quantity", "Price", "Execution fee", "Net P/L"}
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"CSV missing columns: {missing} (detected delimiter '{delim}')")

    fills: List[Dict[str, Any]] = []
    raw_netpls: list[float] = []

    for line_no, r in enumerate(rows[1:], start=1):
        if not any(str(x).strip() for x in r):
            continue
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))

        dt = _parse_datetime(r[idx["Date/Time"]])
        if not dt:
            continue

        symbol = str(r[idx["Symbol"]]).strip().upper()
        side   = str(r[idx["Side"]]).strip().title()  # Buy/Sell
        qty    = int(_num(r[idx["Quantity"]]))
        price  = float(_num(r[idx["Price"]]))
        fees   = float(_num(r[idx["Execution fee"]]))
        netpl  = float(_num(r[idx["Net P/L"]]))
        raw_netpls.append(netpl)

        account = str(r[idx.get("Account", -1)]).strip() if "Account" in idx else ""
        exch    = str(r[idx.get("Trading exchange", -1)]).strip() if "Trading exchange" in idx else ""

        # guarantee uniqueness: even identical lines in the same second do not collapse
        uid = f"{dt.replace(microsecond=0).isoformat()}#{line_no}"

        fills.append({
            "uid": uid,
            "seq": line_no,
            "datetime": dt,
            "date": dt.date(),
            "Symbol": symbol,
            "Side": side,
            "Quantity": qty,
            "Price": price,
            "Fees": fees,
            "NetPL": netpl,
            "Account": account,
            "Exchange": exch,
        })

    # ------------ AUTO-SIGN FIX ------------
    # If broker exports absolute Net P/L (all non-negative), recompute sign per execution via cashflow.
    has_negative = any(v < 0 for v in raw_netpls)
    any_nonzero  = any(abs(v) > 1e-12 for v in raw_netpls)
    if any_nonzero and not has_negative:
        for f in fills:
            if f["Side"].lower().startswith("b"):  # Buy = cash out
                cash = -(f["Quantity"] * f["Price"]) - f["Fees"]
            else:                                   # Sell = cash in
                cash = +(f["Quantity"] * f["Price"]) - f["Fees"]
            f["NetPL"] = float(round(cash, 6))

    # stable order: time -> original file order -> uid
    fills.sort(key=lambda x: (x["datetime"], x["seq"], x["uid"]))
    return fills

# ----------------------------- analytics/grouper -----------------------------
def _signed_qty(side: str, qty: int) -> int:
    return qty if side.lower().startswith("b") else -qty

def build_trades(fills: list[dict]) -> list[dict]:
    """
    Group fills into flat-to-flat trades per (Symbol, Account, calendar day).
    Trade PnL is STRICTLY the sum of executions' NetPL (sign preserved).
    """
    fs = sorted(fills, key=lambda f: (f["date"], f["datetime"], f["Symbol"], f.get("Account","")))
    trades: list[dict] = []
    state: dict[tuple[date,str,str], dict] = {}  # (date, symbol, account) -> {pos:int, trade:dict|None}
    next_id = 1

    def sgn(side: str) -> int:
        return 1 if side.lower().startswith("b") else -1

    def new_trade(first: dict) -> dict:
        nonlocal next_id
        tid = f"T{next_id:06d}"
        next_id += 1
        return {
            "id": tid,
            "symbol": first["Symbol"],
            "account": first.get("Account",""),
            "date": first["date"],
            "side": ("Long" if sgn(first["Side"]) > 0 else "Short"),
            "open_time": first["datetime"],
            "close_time": None,
            "volume": 0,
            "executions": [],
            "executions_count": 0,
            "pnl": 0.0,
            "fees": 0.0,
        }

    def append_exec(t: dict, f: dict):
        ex = {
            "datetime": f["datetime"],
            "Symbol": f["Symbol"],
            "Side": f["Side"],
            "Quantity": int(f["Quantity"]),
            "Price": float(f["Price"]),
            "Fees": float(f["Fees"]),
            "NetPL": float(f["NetPL"]),
            "Account": f.get("Account",""),
            "Exchange": f.get("Exchange",""),
        }
        t["executions"].append(ex)
        t["executions_count"] = len(t["executions"])
        t["volume"] += abs(ex["Quantity"])
        t["fees"] += ex["Fees"]
        # IMPORTANT: preserve sign from each execution
        t["pnl"] += ex["NetPL"]

    for f in fs:
        key = (f["date"], f["Symbol"], f.get("Account",""))
        st = state.get(key)
        if st is None:
            st = {"pos": 0, "trade": None}
            state[key] = st

        q = sgn(f["Side"]) * int(f["Quantity"])

        if st["pos"] == 0:
            st["trade"] = new_trade(f)

        append_exec(st["trade"], f)
        st["pos"] += q

        # flat -> close trade
        if st["pos"] == 0:
            st["trade"]["close_time"] = f["datetime"]
            trades.append(st["trade"])
            st["trade"] = None

    # if anything left open at the end, emit it as an open trade
    for st in state.values():
        if st["trade"] is not None:
            trades.append(st["trade"])

    trades.sort(key=lambda t: t["open_time"])
    return trades

def daily_equity(fills: List[Dict[str, Any]]) -> tuple[list[str], list[float], list[float]]:
    trades = build_trades(fills)
    by_day: Dict[date, float] = defaultdict(float)
    for t in trades:
        d = (t["close_time"] or t["open_time"]).date()
        by_day[d] += float(t["pnl"])

    days = sorted(by_day.keys())
    daily = [round(by_day[d], 2) for d in days]
    equity, run = [], 0.0
    for v in daily:
        run += v
        equity.append(round(run, 2))
    labels = [d.isoformat() for d in days]
    return labels, daily, equity

def daily_metrics_last_n(fills: list[dict], n: int = 30) -> dict:
    trades = build_trades(fills)
    by_day_pl = defaultdict(float)
    by_day_wins = defaultdict(int)
    by_day_total = defaultdict(int)

    for t in trades:
        d = (t["close_time"] or t["open_time"]).date()
        by_day_pl[d] += float(t["pnl"])
        by_day_wins[d] += 1 if t["pnl"] > 0 else 0
        by_day_total[d] += 1

    by_day_vol = defaultdict(int)
    for f in fills:
        by_day_vol[f["date"]] += abs(int(f["Quantity"]))

    days = sorted(set(by_day_total.keys()) | set(by_day_vol.keys()))
    if not days:
        return {"labels": [], "daily_pl": [], "equity": [], "volume": [], "winpct": []}
    days = days[-n:]

    labels = [d.isoformat() for d in days]
    daily_pl = [round(by_day_pl.get(d, 0.0), 2) for d in days]
    volume = [int(by_day_vol.get(d, 0)) for d in days]
    winpct = [
        round((by_day_wins.get(d, 0) / by_day_total.get(d, 1)) * 100.0, 1) if by_day_total.get(d, 0) else 0.0
        for d in days
    ]

    eq, run = [], 0.0
    for v in daily_pl:
        run += v
        eq.append(round(run, 2))

    return {"labels": labels, "daily_pl": daily_pl, "equity": eq, "volume": volume, "winpct": winpct}

def symbol_pl(fills: List[Dict[str, Any]]) -> list[dict]:
    agg: Dict[str, float] = defaultdict(float)
    for f in fills:
        agg[f["Symbol"]] += float(f["NetPL"])
    items = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:30]
    return [{"Symbol": k, "PnL": round(v, 2)} for k, v in items]

def side_counts(fills: List[Dict[str, Any]]) -> list[dict]:
    c = Counter(f["Side"] for f in fills)
    return [{"Side": k, "Count": v} for k, v in c.items()]

def filter_fills(fills: List[Dict[str, Any]], symbol: str = "", dfrom: str = "", dto: str = "") -> List[Dict[str, Any]]:
    sym = symbol.strip().upper()
    df, dt = None, None
    if dfrom:
        try: df = datetime.fromisoformat(dfrom).date()
        except Exception: pass
    if dto:
        try: dt = datetime.fromisoformat(dto).date()
        except Exception: pass
    out = []
    for f in fills:
        if sym and sym not in f["Symbol"]:
            continue
        if df and f["date"] < df:
            continue
        if dt and f["date"] > dt:
            continue
        out.append(f)
    return out

def preview_table(fills: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    out = []
    for f in fills[:limit]:
        out.append({
            "datetime": f["datetime"].strftime("%Y-%m-%d %I:%M:%S %p"),
            "Symbol": f["Symbol"], "Side": f["Side"],
            "Quantity": f["Quantity"], "Price": f["Price"],
            "Fees": f["Fees"], "NetPL": f["NetPL"],
            "Account": f.get("Account", ""), "Exchange": f.get("Exchange", ""),
        })
    return out

def pack_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
