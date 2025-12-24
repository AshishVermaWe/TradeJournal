from __future__ import annotations
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
from html import escape
import os, json, calendar, math, uuid
from datetime import datetime, date, timedelta, timezone
from string import Template
from email.parser import BytesParser
from email.policy import default
import urllib.request, urllib.parse
from zoneinfo import ZoneInfo
from typing import Iterable

# ---------------------------- persistent storage paths ----------------------------
# Use a Render disk mounted at /var/data (or set DATA_DIR via env var)
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
os.makedirs(DATA_DIR, exist_ok=True)

STORE_PATH = os.path.join(DATA_DIR, "trades.json")
NOTES_PATH = os.path.join(DATA_DIR, "journal_notes.json")

# Create files if missing
if not os.path.exists(STORE_PATH):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        f.write("[]")
if not os.path.exists(NOTES_PATH):
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("{}")

from parser import (
    parse_upload, daily_equity, symbol_pl, side_counts,
    pack_json, build_trades, daily_metrics_last_n
)

ROOT = os.path.dirname(os.path.abspath(__file__))
TPL  = lambda name: os.path.join(ROOT, "templates", name)
STATIC_DIR = os.path.join(ROOT, "static")
<<<<<<< HEAD
DATA_DIR   = os.path.join(ROOT, "data")
STORE_PATH = os.path.join(DATA_DIR, "trades.json")
NOTES_PATH = os.path.join(DATA_DIR, "journal_notes.json")
RULES_PATH = os.path.join(DATA_DIR, "rules.json")
CHECKLIST_PATH = os.path.join(DATA_DIR, "rules_checklist.json")
os.makedirs(DATA_DIR, exist_ok=True)
=======
>>>>>>> 7a1ba79955c653abca65f9247a04ae6d1fa5a11a

DEFAULT_TZ = "America/New_York"  # ET (EST/EDT)

# ---------------------------- in-memory cache ----------------------------
LAST = {
    "fills": [],
    "stats": None,
    "labels": [],
    "daily": [],
    "equity": [],
    "sym": [],
    "side": [],
    "trades": [],
    "diag": {
        "fills_total": 0.0,
        "trades_total": 0.0,
        "mismatch": 0.0,
        "trade_count": 0,
        "fill_count": 0,
        "note": ""
    }
}

# ---------------------------- util ----------------------------
def read_file(path: str, mode="rb"):
    if "b" in mode:
        with open(path, mode) as f:
            return f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

def parse_multipart(environ) -> dict:
    ctype = environ.get("CONTENT_TYPE", "")
    if "multipart/form-data" not in ctype or "boundary=" not in ctype:
        return {}
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length <= 0 or length > 100 * 1024 * 1024:
        return {}
    body = environ["wsgi.input"].read(length)
    headers = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=default).parsebytes(headers + body)
    result = {}
    if msg.is_multipart():
        for part in msg.iter_parts():
            cd = part.get("Content-Disposition", "") or ""
            if "form-data" not in cd:
                continue
            name = part.get_param("name", header="Content-Disposition")
            filename = part.get_param("filename", header="Content-Disposition")
            payload = part.get_payload(decode=True) or b""
            if name:
                result[name] = {"filename": filename, "content": payload}
    return result

def render(template: str, **ctx):
    base = read_file(TPL("base.html"), "r")
    body = read_file(TPL(template), "r")
    page = Template(base).safe_substitute(CONTENT=body)
    return Template(page).safe_substitute(**{k: str(v) for k, v in ctx.items()}).encode("utf-8")

def _trade_summary(trades: list[dict]) -> dict:
    n = len(trades)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    win_rate = (len(wins) * 100.0 / n) if n else 0.0
    total_pl = round(sum(float(t["pnl"]) for t in trades), 2)
    best = round(max(wins, default=0.0), 2)
    worst = round(min(losses, default=0.0), 2)
    avg_pl = round((total_pl / n), 2) if n else 0.0
    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pl": total_pl,
        "best": best,
        "worst": worst,
        "avg_pl": avg_pl,
    }

def _diag_update(fills: list[dict], trades: list[dict]):
    fills_total = round(sum(float(f["NetPL"]) for f in fills), 2)
    trades_total = round(sum(float(t["pnl"]) for t in trades), 2)
    mismatch = round(trades_total - fills_total, 2)
    LAST["diag"] = {
        "fills_total": fills_total,
        "trades_total": trades_total,
        "mismatch": mismatch,
        "trade_count": len(trades),
        "fill_count": len(fills),
        "note": ("OK" if abs(mismatch) <= 0.01 else "Totals differ. Check data.")
    }

def update_last(fills: list[dict]):
    LAST["fills"] = fills
    labels, daily, equity = daily_equity(fills)
    LAST["labels"], LAST["daily"], LAST["equity"] = labels, daily, equity
    LAST["sym"] = symbol_pl(fills)
    LAST["side"] = side_counts(fills)
    trades = build_trades(fills)
    LAST["trades"] = trades
    LAST["stats"] = _trade_summary(trades)
    _diag_update(fills, trades)

# ---------------------------- storage ----------------------------
def _norm_fill(f: dict) -> dict:
    g = dict(f)
    g["datetime"] = f["datetime"].isoformat(timespec="seconds")
    g["date"] = f["date"].isoformat()
    return g

def _denorm_fill(g: dict) -> dict:
    f = dict(g)
    f["datetime"] = datetime.fromisoformat(g["datetime"])
    f["date"] = date.fromisoformat(g["date"])
    return f

def store_load() -> list[dict]:
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [_denorm_fill(x) for x in raw]
    except Exception:
        return []

def store_save(fills: list[dict]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as fh:
        json.dump([_norm_fill(f) for f in fills], fh, ensure_ascii=False, indent=2)

def _fill_key(f: dict) -> tuple:
    return (
        f.get("uid", ""),
        f["datetime"].isoformat(timespec="seconds"),
        f["Symbol"], f["Side"], int(f["Quantity"]), float(f["Price"]),
        f.get("Account","")
    )

def store_merge(old_fills: list[dict], new_fills: list[dict]) -> list[dict]:
    seen = { _fill_key(f) for f in old_fills }
    out = list(old_fills)
    for f in new_fills:
        k = _fill_key(f)
        if k not in seen:
            out.append(f)
            seen.add(k)
    out.sort(key=lambda x: (x["datetime"], x.get("seq", 0), x.get("uid", "")))
    return out

# ---------------------------- calendar helpers ----------------------------
def trades_on_date(trades: list[dict], day: date) -> list[dict]:
    out: list[dict] = []
    for t in trades:
        if any(ex["datetime"].date() == day for ex in t.get("executions", [])):
            out.append(t)
    def first_exec_that_day(t) -> datetime:
        times = [ex["datetime"] for ex in t.get("executions", []) if ex["datetime"].date() == day]
        return min(times) if times else (t["open_time"])
    out.sort(key=first_exec_that_day)
    return out

def _daily_map_from_fills(fills: Iterable[dict]) -> dict[date, dict]:
    """
    Map of day -> {'pl': realized_net_for_day, 'count': fills_count}
    """
    m: dict[date, dict] = {}
    by_day = {}
    for f in fills:
        d = f["date"]
        by_day.setdefault(d, []).append(f)
    for d, arr in by_day.items():
        ser = _day_realized_series(d.isoformat(), arr)
        pl = ser["cum"][-1] if ser["cum"] else 0.0
        m[d] = {"pl": float(pl), "count": len(arr)}
    return m

def _month_name(y, m):
    import calendar as _cal
    return _cal.month_name[m]

def _month_nav(year, month):
    prev_y, prev_m = (year-1, 12) if month == 1 else (year, month-1)
    next_y, next_m = (year+1, 1)  if month == 12 else (year, month+1)
    return prev_y, prev_m, next_y, next_m

def _dashboard_kpis():
    trades = LAST.get("trades") or []
    net_pl = round(sum(float(t["pnl"]) for t in trades), 2)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    avg_win = round(sum(wins)/len(wins), 2) if wins else 0.0
    avg_loss = round(sum(losses)/len(losses), 2) if losses else 0.0

    day_pl = {}
    for t in trades:
        d = (t["close_time"] or t["open_time"]).date()
        day_pl[d] = day_pl.get(d, 0.0) + float(t["pnl"])
    days = sorted(day_pl.keys())
    day_win_pct = round((sum(1 for d in days if day_pl[d] > 0) * 100.0 / len(days)), 2) if days else 0.0
    avg_trades_day = round((len(trades) / len(days)), 2) if days else 0.0

    largest_win  = round(max([t["pnl"] for t in trades], default=0.0), 2)
    largest_loss = round(min([t["pnl"] for t in trades], default=0.0), 2)

    l30 = daily_metrics_last_n(LAST["fills"], n=30)
    return {
        "net_pl": net_pl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "day_win_pct": day_win_pct,
        "avg_trades_day": avg_trades_day,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "L30": l30,
    }

# ---------- TRADES_EXT payload for reports ----------
def _build_trades_ext_payload() -> list[dict]:
    """
    Flatten LAST['trades'] into the per-trade list used by the
    stats widgets on /reports (TRADES_EXT).
    """
    trades = LAST.get("trades") or []
    out: list[dict] = []

    for t in trades:
        execs = t.get("executions", []) or []

        if execs:
            first_dt = min(e["datetime"] for e in execs)
            last_dt  = max(e["datetime"] for e in execs)
        else:
            first_dt = t.get("open_time")
            last_dt  = t.get("close_time") or t.get("open_time")

        day = first_dt.date().isoformat() if first_dt else ""

        out.append({
            "symbol": t.get("symbol"),
            "date": day,                                   # "YYYY-MM-DD"
            "entry_time": first_dt.isoformat() if first_dt else None,
            "exit_time":  last_dt.isoformat() if last_dt else None,
            "pnl": float(t.get("pnl") or 0.0),
            "qty": int(t.get("volume") or 0),
            "commission": 0.0,
            "fees": float(t.get("fees") or 0.0),
            "mae": t.get("mae"),
            "mfe": t.get("mfe"),
            "fills": int(t.get("executions_count", len(execs))),
            "tags": t.get("tags") or [],
        })

    return out

# ---------------------------- market data (for charts) ----------------------------
def _day_bounds_utc(yyyymmdd: str, tz_name: str):
    tz = ZoneInfo(tz_name)
    y, m, d = int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    start_local = datetime(y, m, d, 0, 0, 0, tzinfo=tz)
    end_local   = start_local + timedelta(days=1)
    return int(start_local.astimezone(timezone.utc).timestamp()), int(end_local.astimezone(timezone.utc).timestamp())

def _yahoo_chart_url(symbol: str, yyyymmdd: str | None, tz_name: str):
    base = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(symbol)
    today_local = datetime.now(ZoneInfo(tz_name)).strftime("%Y%m%d")
    if yyyymmdd is None:
        yyyymmdd = today_local
    if yyyymmdd == today_local:
        qs = {"interval": "1m", "range": "1d", "includePrePost": "true"}
        return base + "?" + urllib.parse.urlencode(qs)
    p1, p2 = _day_bounds_utc(yyyymmdd, tz_name)
    qs = {"interval": "1m", "period1": str(p1), "period2": str(p2), "includePrePost": "true"}
    return base + "?" + urllib.parse.urlencode(qs)

def fetch_intraday_1m(symbol: str, yyyymmdd: str | None, tz_name: str):
    url = _yahoo_chart_url(symbol, yyyymmdd, tz_name)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    err = data.get("chart", {}).get("error")
    if err:
        return []

    res = data.get("chart", {}).get("result", [])
    if not res:
        return []

    series = res[0]
    ts = series.get("timestamp", []) or []
    ind = series.get("indicators", {}).get("quote", [{}])[0]
    o = ind.get("open", []); h = ind.get("high", []); l = ind.get("low", []); c = ind.get("close", []); v = ind.get("volume", [])

    candles = []
    for i in range(min(len(ts), len(o), len(h), len(l), len(c), len(v))):
        if o[i] is None or h[i] is None or l[i] is None or c[i] is None:
            continue
        candles.append({
            "time": int(ts[i]),
            "open": float(o[i]), "high": float(h[i]), "low": float(l[i]),
            "close": float(c[i]), "volume": int(v[i] or 0),
        })
    return candles

def _resample_5m(candles_1m: list[dict]) -> list[dict]:
    if not candles_1m:
        return []
    out: list[dict] = []
    cur = None
    cur_bucket = None
    for c in sorted(candles_1m, key=lambda x: int(x["time"])):
        t = int(c["time"])
        bucket = (t // 300) * 300
        if bucket != cur_bucket:
            if cur:
                out.append(cur)
            cur = {
                "time": bucket,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low":  float(c["low"]),
                "close":float(c["close"]),
                "volume": float(c.get("volume", 0.0))
            }
            cur_bucket = bucket
        else:
            cur["high"]   = max(cur["high"], float(c["high"]))
            cur["low"]    = min(cur["low"],  float(c["low"]))
            cur["close"]  = float(c["close"])
            cur["volume"] = float(cur["volume"]) + float(c.get("volume", 0.0))
    if cur:
        out.append(cur)
    return out

def route_api_candles(environ, start_response):
    qs = urllib.parse.parse_qs(environ.get("QUERY_STRING", ""))
    symbol = (qs.get("symbol", [""])[0] or "").upper()
    date_ymd = qs.get("date", [None])[0]
    interval = (qs.get("interval", ["1m"])[0] or "1m").lower()
    tz_name  = (qs.get("tz", [DEFAULT_TZ])[0] or DEFAULT_TZ)

    if interval not in ("1m", "5m"):
        interval = "1m"

    if not symbol:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error":"symbol required"}).encode()]

    try:
        m1 = fetch_intraday_1m(symbol, date_ymd, tz_name)

        if date_ymd and len(date_ymd) == 8 and date_ymd.isdigit():
            start_s, end_s = _day_bounds_utc(date_ymd, tz_name)
            m1 = [c for c in m1 if start_s <= int(c["time"]) < end_s]

        candles = m1 if interval == "1m" else _resample_5m(m1)

        for c in candles:
            c["time"]   = int(c["time"])
            c["open"]   = float(c["open"])
            c["high"]   = float(c["high"])
            c["low"]    = float(c["low"])
            c["close"]  = float(c["close"])
            c["volume"] = float(c.get("volume", 0.0))

        body = json.dumps({"symbol": symbol, "candles": candles})
        start_response("200 OK", [("Content-Type", "application/json"),
                                  ("Cache-Control", "no-store, max-age=0")])
        return [body.encode("utf-8")]
    except Exception as e:
        start_response("502 Bad Gateway", [("Content-Type", "application/json")])
        return [json.dumps({"error": str(e)}).encode()]

# ---------------------------- Journal helpers/APIs ----------------------------
def _collect_symbols_and_tags_from_trades(trades: list[dict]) -> tuple[str, str]:
    symbols = set()
    tags = set()
    for t in trades:
        s = (t.get("symbol") or "").strip()
        if s:
            symbols.add(s)
        tg = t.get("tags")
        if isinstance(tg, list):
            for k in tg:
                k = (k or "").strip()
                if k:
                    tags.add(k)
        elif isinstance(tg, str):
            k = tg.strip()
            if k:
                tags.add(k)

    def opts(all_label, values):
        out = [f'<option value="">{all_label}</option>']
        for v in sorted(values):
            ev = escape(v, quote=True)
            out.append(f'<option value="{ev}">{ev}</option>')
        return "\n".join(out)

    return opts("All Symbols", symbols), opts("All Tags", tags)

def _trade_passes_filters(t: dict, symbol: str, tag: str, side: str, duration: str) -> bool:
    if symbol and (t.get("symbol") or "").strip() != symbol:
        return False
    if side and (t.get("side") or "").strip().lower() != side.lower():
        return False
    if duration and (t.get("duration") or "").strip().lower() != duration.lower():
        return False
    if tag:
        tg = t.get("tags")
        if isinstance(tg, list):
            if tag not in tg:
                return False
        elif isinstance(tg, str):
            if tag != tg:
                return False
        else:
            return False
    return True

def _day_key_for_trade(t: dict) -> str:
    execs = t.get("executions", [])
    if execs:
        first_dt = min(ex["datetime"] for ex in execs)
        return first_dt.date().isoformat()
    if t.get("open_time"):
        return t["open_time"].date().isoformat()
    return ""

def _group_trades_by_day(trades: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in trades:
        key = _day_key_for_trade(t)
        if not key:
            continue
        out.setdefault(key, []).append(t)
    return out

def _compute_day_stats(day_trades: list[dict]) -> dict:
    total_trades = len(day_trades)
    total_volume = sum(int(t.get("volume") or 0) for t in day_trades)
    wins = sum(1 for t in day_trades if float(t.get("pnl") or 0) > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    fees = sum(float(t.get("fees") or 0.0) for t in day_trades)
    gross = sum(float(t.get("pnl") or 0.0) for t in day_trades)
    net_pl = gross - fees
    return {
        "total_trades": total_trades,
        "total_volume": total_volume,
        "win_rate": round(win_rate, 1),
        "fees": round(fees, 2),
        "net_pl": round(net_pl, 2),
    }

def _row_for_day_table(t: dict) -> dict:
    time_str = ""
    if t.get("open_time"):
        time_str = t["open_time"].strftime("%I:%M:%S %p")
    side = (t.get("side") or "").strip()
    if not side:
        try:
            ex0 = min(t.get("executions", []), key=lambda e: e["datetime"]) if t.get("executions") else None
            if ex0:
                side = "Long" if str(ex0.get("Side","")).lower().startswith("b") else "Short"
        except Exception:
            pass
    try:
        ex0 = min(t.get("executions", []), key=lambda e: e["datetime"]) if t.get("executions") else None
        if ex0:
            time_str = ex0["datetime"].strftime("%I:%M:%S %p")
    except Exception:
        pass

    return {
        "id": t.get("id"),
        "symbol": t.get("symbol"),
        "side": side,
        "time": time_str,
        "volume": int(t.get("volume") or 0),
        "execs": int(t.get("executions_count", len(t.get("executions", [])))),
        "pl": float(t.get("pnl") or 0.0),
        "notes": t.get("notes") or "",
        "tags": t.get("tags") or [],
    }

# --- Realized P&L helpers for a day (market-local) ---
def _per_exec_realized_deltas(fills_sorted: list[dict]) -> list[tuple[datetime, float]]:
    deltas: list[tuple[datetime, float]] = []
    pos = 0
    avg = 0.0

    def side_sign(s: str) -> int:
        return 1 if (s or "").lower().startswith("b") else -1

    for e in fills_sorted:
        qty = int(e["Quantity"])
        px  = float(e["Price"])
        fee = float(e.get("Fees", 0.0))
        sgn = side_sign(e["Side"])
        realized = 0.0

        if pos == 0:
            pos = sgn * qty
            avg = px
            realized -= fee
        elif (pos > 0 and sgn > 0) or (pos < 0 and sgn < 0):
            new_abs = abs(pos) + qty
            if new_abs:
                avg = ((avg * abs(pos)) + (px * qty)) / new_abs
            pos += sgn * qty
            realized -= fee
        else:
            close_qty = min(qty, abs(pos))
            if pos > 0 and sgn < 0:
                realized += (px - avg) * close_qty
            elif pos < 0 and sgn > 0:
                realized += (avg - px) * close_qty
            realized -= fee

            pos += sgn * qty
            if pos == 0:
                avg = 0.0
            else:
                if (pos > 0 and sgn > 0) or (pos < 0 and sgn < 0):
                    avg = px

        deltas.append((e["_dt_local"], realized))

    return deltas

<<<<<<< HEAD
def _day_realized_series(the_day_iso: str, fills: Iterable[dict] | None = None) -> dict:
=======
def _day_realized_series(the_day_iso: str) -> dict:
    """
    Build intraday realized P&L series for YYYY-MM-DD using LAST['fills'].
    Filters by the *market-local* calendar day, groups by (Account, Symbol),
    computes per-exec realized deltas, and cumulative-sums them.
    Returns {"ts":[epoch_utc_seconds...], "cum":[...]}.
    """
>>>>>>> 7a1ba79955c653abca65f9247a04ae6d1fa5a11a
    try:
        the_day = date.fromisoformat(the_day_iso)
    except Exception:
        return {"ts": [], "cum": []}

    market_tz = ZoneInfo(DEFAULT_TZ)
    fills_iter = fills if fills is not None else (LAST.get("fills", []) or [])

    candidates = []
    for f in fills_iter:
        dt = f["datetime"]
        if dt.tzinfo is None:
            dt_local = dt.replace(tzinfo=market_tz)
        else:
            dt_local = dt.astimezone(market_tz)
        if dt_local.date() == the_day:
            g = dict(f)
            g["_dt_local"] = dt_local
            candidates.append(g)

    from collections import defaultdict
    buckets = defaultdict(list)
    for g in candidates:
        key = (g.get("Account",""), g.get("Symbol",""))
        buckets[key].append(g)
    for key in buckets:
        buckets[key].sort(key=lambda e: e["_dt_local"])

    deltas_all: list[tuple[datetime, float]] = []
    for key, seq in buckets.items():
        deltas_all.extend(_per_exec_realized_deltas(seq))
    deltas_all.sort(key=lambda t: t[0])

    ts, cum = [], []
    running = 0.0

    if deltas_all:
        first_ts = int(deltas_all[0][0].astimezone(timezone.utc).timestamp())
        ts.append(first_ts)
        cum.append(0.0)

    for (dt_local, dpl) in deltas_all:
        running += float(dpl)
        ts.append(int(dt_local.astimezone(timezone.utc).timestamp()))
        cum.append(round(running, 2))

    if not ts:
        anchor = datetime(the_day.year, the_day.month, the_day.day, 10, 0, tzinfo=market_tz)
        ts = [int(anchor.astimezone(timezone.utc).timestamp())]
        cum = [0.0]

    return {"ts": ts, "cum": cum}

def _day_pl_series(the_day_iso: str) -> dict:
<<<<<<< HEAD
    return _day_realized_series(the_day_iso, LAST.get("fills", []))

# ---------------------------- Rules storage ----------------------------
RULES_DEFAULT = {
    "defined_rules": (
        "Keep the rules visible before each session. Stay within planned risk, trade only validated setups, "
        "and pause when rules are broken more than twice."
    ),
    "items": [
        {
            "id": "risk_guardrails",
            "title": "Protect capital first",
            "category": "Risk",
            "status": "Active",
            "detail": "Max 1R per trade, stop for the day at -2R, and size using the planned stop distance.",
        },
        {
            "id": "process_setup",
            "title": "Trade only planned setups",
            "category": "Process",
            "status": "Focus",
            "detail": "No impulsive entries. Write a brief plan before sending the order: thesis, level, stop, and target.",
        },
        {
            "id": "mindset_opens",
            "title": "Slow open, patient adds",
            "category": "Mindset",
            "status": "Active",
            "detail": "Skip the first 3 minutes unless an A+ setup appears. Adds only if initial risk is paid.",
        },
    ],
}

def _normalize_rule_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    title = (item.get("title") or "").strip()
    if not title:
        return None
    rid = (item.get("id") or uuid.uuid4().hex)
    category = (item.get("category") or "General").strip() or "General"
    status = (item.get("status") or "Active").strip() or "Active"
    detail = (item.get("detail") or "").strip()
    return {"id": rid, "title": title, "category": category, "status": status, "detail": detail}

def _normalize_rules_payload(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {"defined_rules": "", "items": []}
    defined = str(raw.get("defined_rules") or "").strip()
    items_in = raw.get("items")
    if not isinstance(items_in, list):
        items_in = []
    items: list[dict] = []
    for itm in items_in:
        norm = _normalize_rule_item(itm)
        if norm:
            items.append(norm)
    return {"defined_rules": defined, "items": items}

def load_rules_data() -> dict:
    data = None
    if os.path.exists(RULES_PATH):
        try:
            with open(RULES_PATH, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            data = _normalize_rules_payload(stored)
        except Exception:
            data = None
    if data is None:
        data = _normalize_rules_payload(RULES_DEFAULT)
    return data

def save_rules_data(payload: dict) -> None:
    tmp = RULES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, RULES_PATH)

# ---------------------------- Checklist storage ----------------------------
CHECKLIST_TEMPLATE = [
    {"id": "plan_review", "label": "Reviewed premarket plan and levels"},
    {"id": "risk", "label": "Respected position size and max daily loss"},
    {"id": "setups_only", "label": "Traded only A+ setups (no impulse trades)"},
    {"id": "stops", "label": "Executed stops without hesitation"},
    {"id": "journal", "label": "Captured notes/screenshots after trades"},
]
CHECKLIST_DEFAULT = {"entries": []}

def _win_rates_by_day() -> dict[str, float]:
    trades = LAST.get("trades", []) or []
    by_day = _group_trades_by_day(trades)
    out: dict[str, float] = {}
    for d, arr in by_day.items():
        stats = _compute_day_stats(arr)
        out[d] = float(stats.get("win_rate", 0.0) or 0.0)
    return out

def _pnl_by_day() -> dict[str, float]:
    fills = LAST.get("fills", []) or []
    dm = _daily_map_from_fills(fills)
    return {d.isoformat(): float(info.get("pl", 0.0) or 0.0) for d, info in dm.items()}

def _normalize_checklist_entry(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {"date": "", "checks": [], "win_pct": 0.0, "breaches": 0, "notes": "", "quality": "Some trading opportunities", "score": 0.0, "completed": False, "pnl": None}
    ds = (raw.get("date") or "").strip()
    try:
        d = date.fromisoformat(ds)
        dstr = d.isoformat()
    except Exception:
        dstr = ""

    checks_in = raw.get("checks")
    if not isinstance(checks_in, list):
        checks_in = []
    checks: list[dict] = []
    seen_ids: set[str] = set()

    def _add_check(item_id: str, label: str, checked: bool):
        if not item_id:
            return
        if item_id in seen_ids:
            return
        seen_ids.add(item_id)
        checks.append({"id": item_id, "label": label or item_id, "checked": bool(checked)})

    for itm in checks_in:
        if not isinstance(itm, dict):
            continue
        iid = (itm.get("id") or itm.get("label") or "").strip()
        label = (itm.get("label") or iid).strip()
        _add_check(iid, label, bool(itm.get("checked")))

    if not checks:
        for t in CHECKLIST_TEMPLATE:
            _add_check(t["id"], t["label"], False)

    try:
        win_raw = raw.get("win_pct", raw.get("rr", 0.0))
        win_pct = round(float(win_raw or 0.0), 1)
    except Exception:
        win_pct = 0.0
    try:
        breaches = max(0, int(raw.get("breaches") or 0))
    except Exception:
        breaches = 0
    notes = str(raw.get("notes") or "").strip()
    pnl_val = raw.get("pnl", raw.get("net_pl", raw.get("pl")))
    try:
        pnl = round(float(pnl_val), 2)
    except Exception:
        pnl = None
    quality_raw = str(raw.get("quality") or "Some trading opportunities").strip()

    def _map_quality(q: str) -> str:
        v = (q or "").strip().lower()
        if v in ("choppy", "tired", "emotional"):
            return "Choppy"
        if v in ("sharp", "nice", "amazing"):
            return "Amazing"
        if v in ("steady", "so so", "so-so", "neutral", "some trading opportunities"):
            return "Some trading opportunities"
        return q or "Some trading opportunities"

    quality = _map_quality(quality_raw)
    score = round((sum(1 for c in checks if c["checked"]) * 100.0) / max(len(checks), 1), 1)
    completed = all(c["checked"] for c in checks) if checks else False

    return {
        "date": dstr,
        "checks": checks,
        "win_pct": win_pct,
        "breaches": breaches,
        "notes": notes,
        "quality": quality,
        "pnl": pnl,
        "score": score,
        "completed": completed,
    }

def load_checklist_data() -> dict:
    data = {"entries": []}
    if os.path.exists(CHECKLIST_PATH):
        try:
            with open(CHECKLIST_PATH, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            entries = stored.get("entries") if isinstance(stored, dict) else (stored if isinstance(stored, list) else [])
            out: list[dict] = []
            seen: set[str] = set()
            for e in entries:
                norm = _normalize_checklist_entry(e)
                if norm["date"] and norm["date"] not in seen:
                    out.append(norm)
                    seen.add(norm["date"])
            out.sort(key=lambda x: x["date"], reverse=True)
            data["entries"] = out
        except Exception:
            data = {"entries": []}
    if not isinstance(data.get("entries"), list):
        data["entries"] = []
    return data

def save_checklist_data(payload: dict) -> None:
    tmp = CHECKLIST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKLIST_PATH)

=======
    # Keep journal's intraday chart on realized deltas
    return _day_realized_series(the_day_iso)

>>>>>>> 7a1ba79955c653abca65f9247a04ae6d1fa5a11a
# ---------------------------- WSGI app ----------------------------
def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    # static
    if path.startswith("/static/"):
        file_path = os.path.join(ROOT, path.lstrip("/"))
        if not os.path.isfile(file_path):
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not found"]
        ext = os.path.splitext(file_path)[1].lower()
        mime = {
            ".css":  "text/css",
            ".js":   "application/javascript",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg":  "image/svg+xml",
            ".ico":  "image/x-icon",
            ".json": "application/json",
            ".html": "text/html; charset=utf-8",
            ".woff": "font/woff", ".woff2":"font/woff2", ".ttf": "font/ttf",
        }.get(ext, "application/octet-stream")
        start_response("200 OK", [("Content-Type", mime), ("Cache-Control", "no-cache")])
        return [read_file(file_path, "rb")]

    # healthcheck
    if path == "/healthz":
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b'{"ok":true}']

    # root
    if path == "/" and method == "GET":
        start_response("302 Found", [("Location", "/dashboard")])
        return [b""]

    # upload page
    if path == "/upload" and method == "GET":
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("upload.html")]

    # upload handler
    if path == "/upload" and method == "POST":
        try:
            fields = parse_multipart(environ)
            fileinfo = fields.get("file")
            merge_flag = (fields.get("merge", {}).get("content", b"on").decode(errors="ignore").lower() in ("on","true","1","yes"))
            reset_flag = (fields.get("reset", {}).get("content", b"").decode(errors="ignore").lower() == "reset")

            if reset_flag:
                store_save([])
                update_last([])
                start_response("302 Found", [("Location", "/dashboard")])
                return [b""]

            if not fileinfo or not fileinfo.get("content"):
                raise ValueError("No file uploaded")

            incoming = parse_upload(fileinfo["content"])
            current = store_load() if merge_flag else []
            merged = store_merge(current, incoming)
            store_save(merged)
            update_last(merged)

            start_response("302 Found", [("Location", "/dashboard")])
            return [b""]

        except Exception as e:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [f"Error: {e}".encode("utf-8")]

    # export normalized json
    if path == "/storage/export" and method == "GET":
        payload = json.dumps([_norm_fill(f) for f in LAST.get("fills",[])], ensure_ascii=False, indent=2).encode("utf-8")
        headers = [("Content-Type","application/json; charset=utf-8"),
                   ("Content-Disposition","attachment; filename=trades.json")]
        start_response("200 OK", headers)
        return [payload]

    # diagnostics
    if path == "/diag" and method == "GET":
        data = {
            "diag": LAST["diag"],
            "kpis": _dashboard_kpis(),
            "trades_preview": [
                {
                    "id": t["id"], "symbol": t["symbol"], "side": t["side"],
                    "pnl": round(float(t["pnl"]), 2), "volume": t["volume"],
                    "executions": t.get("executions_count", len(t.get("executions", []))),
                    "open": t["open_time"].isoformat(),
                    "close": (t["close_time"].isoformat() if t["close_time"] else None)
                } for t in LAST.get("trades", [])[:5]
            ]
        }
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        start_response("200 OK", [("Content-Type","application/json; charset=utf-8")])
        return [payload]

    # dashboard
    if path == "/dashboard" and method == "GET":
        stats = LAST["stats"] or {
            "total_trades":0,"wins":0,"losses":0,"win_rate":0,
            "total_pl":0,"best":0,"worst":0,"avg_pl":0
        }
        metrics = daily_metrics_last_n(LAST["fills"], n=30)
        trades_ext = _build_trades_ext_payload()
        mismatch = LAST["diag"]["mismatch"]
        banner = ""
        if abs(mismatch) > 0.01:
            banner = (
                f"<div class='alert warn'>"
                f"Totals mismatch: trades={LAST['diag']['trades_total']:.2f} "
                f"vs fills={LAST['diag']['fills_total']:.2f} ({mismatch:+.2f})."
                f"</div>"
            )

        ctx = {
            "TRADES": stats["total_trades"],
            "WINRATE": stats["win_rate"],
            "TOTALPL": stats["total_pl"],
            "BEST": stats["best"],
            "WORST": stats["worst"],
            "AVG": stats["avg_pl"],
            "LABELS": pack_json(LAST["labels"]),
            "DAILY":  pack_json(LAST["daily"]),
            "EQUITY": pack_json(LAST["equity"]),
            # shared widget data (same as /reports)
            "SYM_DATA":   pack_json(LAST.get("sym") or []),
            "SIDE_DATA":  pack_json(LAST.get("side") or []),
            "L30_LABELS": pack_json(metrics["labels"]),
            "L30_DPL":    pack_json(metrics["daily_pl"]),
            "L30_EQ":     pack_json(metrics["equity"]),
            "L30_VOL":    pack_json(metrics["volume"]),
            "L30_WIN":    pack_json(metrics["winpct"]),
            "TRADES_EXT": pack_json(trades_ext),
            "BANNER": banner
        }
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("dashboard.html", **ctx)]

    # calendar month
    if path == "/calendar/month" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        today = date.today()
        year = int(qs.get("year", [today.year])[0])
        month = int(qs.get("month", [today.month])[0])

        dm = _daily_map_from_fills(LAST["fills"])
        cal = calendar.Calendar(firstweekday=6)
        weeks = cal.monthdatescalendar(year, month)

        # Precompute trade counts per day to reuse for the grid and weekly summary
        trade_counts: dict[date, int] = {}
        for d in dm.keys():
            trade_counts[d] = len(trades_on_date(LAST.get("trades", []) or [], d))

        cells = []
        month_total = 0.0
        for idx, wk in enumerate(weeks, start=1):
            week_cells = []
            week_pl = 0.0
            week_trades = 0
            for d in wk:
                if d.month != month:
                    week_cells.append(f"<div class='day mutedcell'><div class='num'>{d.day}</div></div>")
                    continue
                info = dm.get(d, {"pl": 0.0, "count": 0})
                pl = float(info["pl"])
                gcount = trade_counts.get(d, 0)
                cls = "pl pos" if pl > 0 else ("pl neg" if pl < 0 else "pl")
                amt = f"${pl:.2f}"
                trades_txt = f"{gcount} trade{'s' if gcount != 1 else ''}"
                href = f"/trades/day?date={d.isoformat()}"
                week_cells.append(
                    f"<a class='day link-block' href='{href}'>"
                    f"<div class='num'>{d.day}</div>"
                    f"<div class='{cls}'>{amt}</div>"
                    f"<div class='muted'>{trades_txt}</div>"
                    f"</a>"
                )
                month_total += pl
                week_pl += pl
                week_trades += gcount

            label = f"Week {idx}"
            wcls = "pos" if week_pl > 0 else ("neg" if week_pl < 0 else "")
            week_cells.append(
                f"<div class='week-total'>"
                f"<div class='week-title'>{label}</div>"
                f"<div class='week-amount {wcls}'>${week_pl:.2f}</div>"
                f"<div class='week-trades'>{week_trades} trade{'s' if week_trades != 1 else ''}</div>"
                f"</div>"
            )
            cells.extend(week_cells)

        prev_y, prev_m, next_y, next_m = _month_nav(year, month)
        ctx = {
            "YEAR": year,
            "MONTH_NAME": _month_name(year, month),
            "MONTH_TOTAL": f"${month_total:.2f}",
            "MONTH_CLASS": "pos" if month_total >= 0 else "neg",
            "DAY_CELLS": "".join(cells),
            "PREV_MONTH": f"{prev_m}&year={prev_y}",
            "NEXT_MONTH": f"{next_m}&year={next_y}",
        }
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("calendar_month.html", **ctx)]

    # calendar year
    if path == "/calendar" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        today = date.today()
        year = int(qs.get("year", [today.year])[0])
        dm = _daily_map_from_fills(LAST["fills"])
        cards = []
        for month in range(1, 13):
            cal = calendar.Calendar(firstweekday=6)
            weeks = cal.monthdayscalendar(year, month)
            mini = []
            mini.append("".join(f"<div class='d'>{d}</div>" for d in ["S","M","T","W","T","F","S"]))
            for wk in weeks:
                row = []
                for daynum in wk:
                    if daynum == 0:
                        row.append("<div class='d'></div>")
                    else:
                        d = date(year, month, daynum)
                        info = dm.get(d, None)
                        if info:
                            cls = " haspl " + ("pos" if float(info["pl"]) >= 0 else "neg")
                            gcount = len(trades_on_date(LAST.get("trades", []) or [], d))
                            row.append(f"<div class='d{cls}' title='${float(info['pl']):.2f} / {gcount} trades'>{daynum}</div>")
                        else:
                            row.append(f"<div class='d'>{daynum}</div>")
                mini.append("".join(row))
            mini_html = "".join(f"<div class='mini'>{row}</div>" for row in mini)
            card = (
              "<div class='month-card'>"
              f"<div class='title'><div>{_month_name(year, month)} {year}</div>"
              f"<a class='btn' href='/calendar/month?year={year}&month={month}'>Open</a></div>"
              f"{mini_html}</div>"
            )
            cards.append(card)

        ctx = {"YEAR": year, "PREV": year-1, "NEXT": year+1, "MONTH_CARDS": "".join(cards)}
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("calendar_year.html", **ctx)]

    # reports
    if path == "/reports" and method == "GET":
        metrics = daily_metrics_last_n(LAST["fills"], n=30)
        trades_ext = _build_trades_ext_payload()
        ctx = {
            "SYM_DATA":   pack_json(LAST["sym"] or []),
            "SIDE_DATA":  pack_json(LAST["side"] or []),
            "L30_LABELS": pack_json(metrics["labels"]),
            "L30_DPL":    pack_json(metrics["daily_pl"]),
            "L30_EQ":     pack_json(metrics["equity"]),
            "L30_VOL":    pack_json(metrics["volume"]),
            "L30_WIN":    pack_json(metrics["winpct"]),
            "TRADES_EXT": pack_json(trades_ext),
        }
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("reports.html", **ctx)]

    # trades list
    if path == "/trades" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        symbol = (qs.get("symbol", [""])[0] or "").upper()
        dfrom = qs.get("from", [""])[0]
        dto   = qs.get("to", [""])[0]

        trades = LAST.get("trades", []) or []

        def t_date(t) -> date:
            return t.get("date") or (t["open_time"].date() if t.get("open_time") else None)

        def in_range(t):
            if symbol and symbol not in t["symbol"]:
                return False
            td = t_date(t)
            if not td:
                return False
            if dfrom:
                try:
                    if td < date.fromisoformat(dfrom): return False
                except: pass
            if dto:
                try:
                    if td > date.fromisoformat(dto): return False
                except: pass
            return True

        rows = [t for t in trades if in_range(t)]

        def row_html(t):
            status = "Open" if t["close_time"] is None else "Closed"
            cls = "pos" if t["pnl"] >= 0 else "neg"
            vol = t["volume"]
            pnl = f"${t['pnl']:.2f}"
            ex = t.get("executions_count", len(t.get("executions", [])))
            open_ts = t["open_time"].strftime("%Y-%m-%d %I:%M:%S %p")
            return (
                "<tr>"
                f"<td>{open_ts}</td>"
                f"<td><a href='/trade?id={t['id']}' class='link'>{t['symbol']}</a></td>"
                f"<td>{t['side']}</td>"
                f"<td class='num'>{vol}</td>"
                f"<td class='num {cls}'>{pnl}</td>"
                f"<td class='num'>{ex}</td>"
                f"<td>{status}</td>"
                "</tr>"
            )

        TABLE = "".join(row_html(t) for t in rows) or "<tr><td colspan='7'>No trades. Upload a file first.</td></tr>"
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render(
            "trades_list.html",
            SYMBOL=escape(symbol),
            FROM=escape(dfrom),
            TO=escape(dto),
            TABLE=TABLE
        )]

    # trades for a specific day
    if path == "/trades/day" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        ds = qs.get("date", [""])[0]
        try:
            the_day = date.fromisoformat(ds)
        except Exception:
            start_response("400 Bad Request", [("Content-Type","text/plain")])
            return [b"Bad or missing 'date' (use YYYY-MM-DD)"]

        trades = trades_on_date(LAST.get("trades", []) or [], the_day)

        def row_html(t):
            status = "Open" if t["close_time"] is None else "Closed"
            cls = "pos" if t["pnl"] >= 0 else "neg"
            pnl = f"${t['pnl']:.2f}"
            vol = t["volume"]
            ex_count = t.get("executions_count", len(t.get("executions", [])))
            times = [ex["datetime"] for ex in t.get("executions", []) if ex["datetime"].date()==the_day]
            opened = (min(times) if times else (t.get("open_time"))).strftime("%Y-%m-%d %I:%M:%S %p")
            return (
                "<tr>"
                f"<td>{opened}</td>"
                f"<td><a class='link' href='/trade?id={t['id']}'>{t['symbol']}</a></td>"
                f"<td>{t['side']}</td>"
                f"<td class='num'>{vol}</td>"
                f"<td class='num {cls}'>{pnl}</td>"
                f"<td class='num'>{ex_count}</td>"
                f"<td>{status}</td>"
                "</tr>"
            )

        TABLE = "".join(row_html(t) for t in trades) or "<tr><td colspan='7'>No trades on this day.</td></tr>"
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("trades_day.html", DATE=the_day.isoformat(), YEAR=str(the_day.year), MONTH=str(the_day.month).zfill(2), TABLE=TABLE)]

    # API: intraday candles for chart
    if path == "/api/candles" and method == "GET":
        return route_api_candles(environ, start_response)

    # ---------- Journal page ----------
    if path == "/journal" and method == "GET":
        trades = LAST.get("trades", []) or []
        sym_opts, tag_opts = _collect_symbols_and_tags_from_trades(trades)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("journal.html", SYMBOL_OPTIONS=sym_opts, TAG_OPTIONS=tag_opts)]

    # API: Journal intraday P&L for a day
    if path == "/api/journal/day_pl" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        ds = (qs.get("date", [""])[0] or "").strip()
        out = _day_pl_series(ds)
        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps(out).encode("utf-8")]

    # API: Journal days (pagination + filters)
    if path == "/api/journal/days" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        page = max(1, int((qs.get("page", ["1"])[0] or "1")))
        per_page = max(1, min(200, int((qs.get("per_page", ["50"])[0] or "50"))))
        symbol  = (qs.get("symbol",  [""])[0] or "").strip()
        tag     = (qs.get("tag",     [""])[0] or "").strip()
        side    = (qs.get("side",    [""])[0] or "").strip()
        duration= (qs.get("duration",[""])[0] or "").strip()

        trades = LAST.get("trades", []) or []
        filtered = [t for t in trades if _trade_passes_filters(t, symbol, tag, side, duration)]
        by_day = _group_trades_by_day(filtered)

        all_days = sorted(by_day.keys(), reverse=True)
        total_days = len(all_days)
        pages = max(1, math.ceil(total_days / per_page))
        if page > pages: page = pages
        start_i = (page - 1) * per_page
        end_i = start_i + per_page
        days_slice = all_days[start_i:end_i]

        out = []
        for d in days_slice:
            d_trades = by_day[d]
            stats = _compute_day_stats(d_trades)
            rows = [_row_for_day_table(t) for t in d_trades]
            out.append({"date": d, "stats": stats, "trades": rows})

        payload = {
            "page": page,
            "pages": pages,
            "total_days": total_days,
            "days": out,
        }
        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps(payload).encode("utf-8")]

    # API: Journal notes (GET/POST)
    if path == "/api/journal/notes" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        day = (qs.get("date", [""])[0] or "").strip()
        notes = {}
        if os.path.exists(NOTES_PATH):
            try:
                with open(NOTES_PATH, "r", encoding="utf-8") as fh:
                    notes = json.load(fh)
            except Exception:
                notes = {}
        text = notes.get(day, "")
        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"text": text}).encode("utf-8")]

    if path == "/api/journal/notes" and method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except Exception:
            size = 0
        body = (environ["wsgi.input"].read(size) if size > 0 else b"").decode("utf-8", "ignore")
        data = {}
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        day = (data.get("date") or "").strip()
        text = data.get("text") or ""
        if not day:
            start_response("400 Bad Request", [("Content-Type", "application/json; charset=utf-8")])
            return [json.dumps({"error": "missing date"}).encode("utf-8")]

        notes = {}
        if os.path.exists(NOTES_PATH):
            try:
                with open(NOTES_PATH, "r", encoding="utf-8") as fh:
                    notes = json.load(fh)
            except Exception:
                notes = {}
        notes[day] = text
        tmp = NOTES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(notes, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, NOTES_PATH)

        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"ok": True}).encode("utf-8")]

    # Rules page
    if path == "/rules" and method == "GET":
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("rules.html")]

    # API: Rules board (GET/POST)
    if path == "/api/rules" and method == "GET":
        payload = load_rules_data()
        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps(payload, ensure_ascii=False).encode("utf-8")]

    if path == "/api/rules" and method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except Exception:
            size = 0
        body = (environ["wsgi.input"].read(size) if size > 0 else b"").decode("utf-8", "ignore")
        incoming = {}
        try:
            incoming = json.loads(body) if body else {}
        except Exception:
            incoming = {}

        payload = _normalize_rules_payload(incoming)
        try:
            save_rules_data(payload)
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json; charset=utf-8")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"ok": True}).encode("utf-8")]

    # API: Rules checklist (GET/POST)
    if path == "/api/rules/checklist" and method == "GET":
        payload = load_checklist_data()
        pnl_map = _pnl_by_day()
        entries = []
        for e in payload.get("entries", []):
            row = dict(e)
            if (row.get("pnl") is None or row.get("pnl") == "") and row.get("date") in pnl_map:
                row["pnl"] = pnl_map[row["date"]]
            entries.append(row)
        resp = {
            "template": CHECKLIST_TEMPLATE,
            "entries": entries,
            "today": date.today().isoformat(),
            "win_rates": _win_rates_by_day(),
            "pnl_by_day": pnl_map,
        }
        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps(resp, ensure_ascii=False).encode("utf-8")]

    if path == "/api/rules/checklist" and method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except Exception:
            size = 0
        body = (environ["wsgi.input"].read(size) if size > 0 else b"").decode("utf-8", "ignore")
        incoming = {}
        try:
            incoming = json.loads(body) if body else {}
        except Exception:
            incoming = {}

        entry = _normalize_checklist_entry(incoming)
        if not entry["date"]:
            start_response("400 Bad Request", [("Content-Type", "application/json; charset=utf-8")])
            return [json.dumps({"error": "missing date"}).encode("utf-8")]

        data = load_checklist_data()
        entries = [e for e in data.get("entries", []) if e.get("date") != entry["date"]]
        entries.append(entry)
        entries.sort(key=lambda x: x.get("date",""), reverse=True)
        try:
            save_checklist_data({"entries": entries})
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json; charset=utf-8")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

        start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
        return [json.dumps({"ok": True, "entry": entry, "entries": entries}, ensure_ascii=False).encode("utf-8")]

    if path == "/api/rules/checklist/clear" and method == "POST":
        try:
            save_checklist_data({"entries": []})
            start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
            return [b'{"ok":true}']
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json; charset=utf-8")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

    # single trade detail
    if path == "/trade" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING") or "")
        tid = qs.get("id", [""])[0]
        trades = LAST.get("trades", []) or []
        trade = next((t for t in trades if t["id"] == tid), None)

        if not trade:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [f"Trade {tid} not found".encode("utf-8")]

        sym = trade["symbol"]
        side = trade["side"]
        vol = trade["volume"]
        pnl = float(trade["pnl"])
        fees = float(trade["fees"])
        status = "Open" if trade["close_time"] is None else "Closed"
        open_ts = trade["open_time"].strftime("%Y-%m-%d %I:%M:%S %p")
        close_ts = trade["close_time"].strftime("%Y-%m-%d %I:%M:%S %p") if trade["close_time"] else "-"

        executions = sorted(trade.get("executions", []), key=lambda e: e["datetime"])
        account = executions[0].get("Account", "") if executions else ""
        ex_count = len(executions)

        MARKET_TZ = ZoneInfo(DEFAULT_TZ)
        def sgn(x: str) -> int:
            return 1 if x.lower().startswith("b") else -1

        def compute_exec_pnl(exec_list: list[dict]) -> list[dict]:
            """
            Walk executions in order and emit realized PnL, cumulative PnL, and position after each fill.
            """
            pos = 0
            avg = 0.0
            running = 0.0
            out: list[dict] = []

            for e in exec_list:
                dt_local = e["datetime"].replace(tzinfo=MARKET_TZ)
                ts_utc = int(dt_local.astimezone(timezone.utc).timestamp())

                qty = int(e["Quantity"])
                px = float(e["Price"])
                fee = float(e["Fees"])
                side_sgn = sgn(e["Side"])

                realized = 0.0

                if pos == 0:
                    pos = side_sgn * qty
                    avg = px
                    realized -= fee
                elif (pos > 0 and side_sgn > 0) or (pos < 0 and side_sgn < 0):
                    new_abs = abs(pos) + qty
                    if new_abs:
                        avg = ((avg * abs(pos)) + (px * qty)) / new_abs
                    pos += side_sgn * qty
                    realized -= fee
                else:
                    close_qty = min(qty, abs(pos))
                    if pos > 0 and side_sgn < 0:
                        realized += (px - avg) * close_qty
                    elif pos < 0 and side_sgn > 0:
                        realized += (avg - px) * close_qty
                    realized -= fee

                    pos += side_sgn * qty
                    if pos == 0:
                        avg = 0.0
                    else:
                        if (pos > 0 and side_sgn > 0) or (pos < 0 and side_sgn < 0):
                            avg = px

                running += realized
                out.append({
                    "ts": ts_utc,
                    "side": e["Side"],
                    "qty": qty,
                    "price": px,
                    "fees": fee,
                    "realized": round(realized, 2),
                    "cum": round(running, 2),
                    "pos_after": pos,
                })

            return out

        ex_stats = compute_exec_pnl(executions)
        EXECS_JSON = json.dumps(ex_stats, ensure_ascii=False)
        trade_day_ymd = (trade["open_time"]).strftime("%Y%m%d")

        def ex_row(e: dict, stats: dict) -> str:
            realized = float(stats.get("realized", 0.0))
            pos_after = int(stats.get("pos_after", 0))
            cls = "pos" if realized > 0 else ("neg" if realized < 0 else "")
            return (
                "<tr>"
                f"<td>{e['datetime'].strftime('%Y-%m-%d %I:%M:%S %p')}</td>"
                f"<td>{e['Symbol']}</td>"
                f"<td>{e['Side']}</td>"
                f"<td class='num'>{int(e['Quantity'])}</td>"
                f"<td class='num'>{float(e['Price']):.4f}</td>"
                f"<td class='num'>{float(e['Fees']):.2f}</td>"
                f"<td class='num {cls}'>${realized:.2f}</td>"
                f"<td class='num'>{pos_after}</td>"
                f"<td>{e.get('Account','')}</td>"
                f"<td>{e.get('Exchange','')}</td>"
                "</tr>"
            )

        EX_ROWS = "".join(ex_row(e, ex_stats[i] if i < len(ex_stats) else {}) for i, e in enumerate(executions))

        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [render("trade_detail.html",
                    TID=tid, SYMBOL=sym, SIDE=side, VOL=str(vol),
                    PNL=f"${pnl:.2f}", PNL_CLASS=("pos" if pnl>=0 else "neg"),
                    FEES=f"${fees:.2f}", STATUS=status,
                    OPEN=open_ts, CLOSE=close_ts,
                    EX_COUNT=str(ex_count), ACCOUNT=account,
                    SIDE_BADGE_BG="", STATUS_BADGE_BG="", SIDE_CLASS="", STATUS_CLASS="",
                    TRADE_DATE_YYYYMMDD=trade_day_ymd,
                    EXECS_JSON=EXECS_JSON,
                    EX_ROWS=EX_ROWS)]

    # fallback
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not found"]

# ---------------------------- initial load ----------------------------
_loaded = store_load()
<<<<<<< HEAD
if _loaded:
    update_last(_loaded)

if __name__ == "__main__":
    port = 8000
    print(f"Serving on http://127.0.0.1:{port}")
    with make_server("127.0.0.1", port, app) as httpd:
=======
update_last(_loaded if _loaded else [])

# ---------------------------- entrypoint ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = "0.0.0.0"  # important for Docker/Render
    with make_server(host, port, app) as httpd:
        print(f"Serving on http://{host}:{port}")
>>>>>>> 7a1ba79955c653abca65f9247a04ae6d1fa5a11a
        httpd.serve_forever()
