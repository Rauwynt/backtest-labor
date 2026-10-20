"""Backtest-Labor: Anlagestrategien mit historischen Daten testen.
Start:  pip install streamlit yfinance pandas numpy
        streamlit run backtest_app.py
"""
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Backtest-Labor", layout="wide")
STRATS = ["Buy & Hold", "SMA-Crossover", "Momentum", "RSI Mean-Reversion", "60/40 Rebalancing"]
SIGNAL_STRATS = ["Buy & Hold", "SMA-Crossover", "Momentum", "RSI Mean-Reversion"]


# ---------- Daten ----------
@st.cache_data(show_spinner=False, ttl=3600)
def load(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna()


# ---------- Signale (Exposure 0..1 der Aktienposition) ----------
def sig_sma(p, f, s):
    return (p.rolling(f).mean() > p.rolling(s).mean()).astype(float)


def sig_mom(p, n):
    return (p / p.shift(n) - 1 > 0).astype(float)


def sig_rsi(p, n, lo, hi):
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    out, s = np.zeros(len(p)), 0.0
    for i, r in enumerate(rsi.values):
        if r < lo:
            s = 1.0
        elif r > hi:
            s = 0.0
        out[i] = s
    return pd.Series(out, index=p.index)


def target(name, p, P):
    """Gibt (Gewicht Aktie, Gewicht Anleihe, Rebalancing-Frequenz) zurück."""
    one = pd.Series(1.0, index=p.index)
    zero = one * 0
    if name == "Buy & Hold":
        return one, zero, None
    if name == "60/40 Rebalancing":
        return one * 0.6, one * 0.4, P["reb"]
    if name == "SMA-Crossover":
        s = sig_sma(p, P["f"], P["s"])
    elif name == "Momentum":
        s = sig_mom(p, P["n"])
    else:
        s = sig_rsi(p, P["rn"], P["lo"], P["hi"])
    # Signal von gestern -> Handel heute (kein Look-ahead-Bias)
    return s.shift(1).fillna(0), zero, None


# ---------- Simulation ----------
def simulate(px, wr, wb, freq, c, monthly, init):
    idx, n = px.index, len(px.index)
    pr, pb = px["r"].values, px["b"].values
    wr, wb = wr.values, wb.values
    m = idx.to_period("M")
    first_m = np.r_[True, m[1:] != m[:-1]]
    y = idx.year.values
    first_y = np.r_[True, y[1:] != y[:-1]]
    per = first_m if freq == "Monatlich" else first_y if freq == "Jährlich" else np.zeros(n, bool)

    sh, basis = [0.0, 0.0], [0.0, 0.0]
    cash, contributed = init, init
    fees = taxes = ytd_gain = ytd_tax = 0.0
    trades, year, prev_w = 0, idx[0].year, None
    vals, contrib = np.zeros(n), np.zeros(n)

    for i in range(n):
        if idx[i].year != year:
            year, ytd_gain, ytd_tax = idx[i].year, 0.0, 0.0
        add = monthly if (i > 0 and first_m[i]) else 0.0
        cash += add
        contributed += add
        contrib[i] = add
        w = (wr[i], wb[i])
        if i == 0 or add > 0 or w != prev_w or per[i]:
            price = (pr[i], pb[i])
            total = cash + sh[0] * price[0] + sh[1] * price[1]
            deltas = [w[k] * total - sh[k] * price[k] for k in (0, 1)]
            for k in sorted((0, 1), key=lambda k: deltas[k]):  # erst verkaufen, dann kaufen
                d = deltas[k]
                if abs(d) < c["min_trade"]:
                    continue
                if d < 0:
                    amt = -d
                    fee = c["fix"] + amt * (c["pct"] + c["spread"])
                    tax = 0.0
                    if c["tax"] > 0:
                        ytd_gain += amt * (1 - basis[k] / price[k])
                        due = c["tax"] * max(0.0, ytd_gain - c["allow"])
                        tax, ytd_tax = due - ytd_tax, due
                    cash += amt - fee - tax
                    sh[k] -= amt / price[k]
                    fees += fee
                    taxes += tax
                    trades += 1
                else:
                    x = min(d, (cash - c["fix"]) / (1 + c["pct"] + c["spread"]))
                    if x < c["min_trade"]:
                        continue
                    fee = c["fix"] + x * (c["pct"] + c["spread"])
                    new = x / price[k]
                    basis[k] = (basis[k] * sh[k] + x) / (sh[k] + new)
                    sh[k] += new
                    cash -= x + fee
                    fees += fee
                    trades += 1
            prev_w = w
        vals[i] = cash + sh[0] * pr[i] + sh[1] * pb[i]

    v = pd.Series(vals, idx)
    r = ((v - pd.Series(contrib, idx)) / v.shift(1) - 1).replace([np.inf, -np.inf], np.nan).iloc[1:].fillna(0)
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    yrs = max(len(r) / 252, 1e-9)
    vol = r.std() * np.sqrt(252)
    met = {
        "Endwert (€)": v.iloc[-1],
        "Eingezahlt (€)": contributed,
        "Gewinn (€)": v.iloc[-1] - contributed,
        "Rendite p.a. (%)": (curve.iloc[-1] ** (1 / yrs) - 1) * 100,
        "Volatilität (%)": vol * 100,
        "Sharpe": r.mean() * 252 / vol if vol > 0 else 0.0,
        "Max. Drawdown (%)": dd.min() * 100,
        "Gebühren (€)": fees,
        "Steuern (€)": taxes,
        "Trades": trades,
    }
    return {"met": met, "equity": v, "dd": dd}


def finish(vals, contrib, idx, contributed, fees, taxes, trades):
    v = pd.Series(vals, idx)
    r = ((v - pd.Series(contrib, idx)) / v.shift(1) - 1).replace([np.inf, -np.inf], np.nan).iloc[1:].fillna(0)
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    yrs = max(len(r) / 252, 1e-9)
    vol = r.std() * np.sqrt(252)
    met = {
        "Endwert (€)": v.iloc[-1], "Eingezahlt (€)": contributed, "Gewinn (€)": v.iloc[-1] - contributed,
        "Rendite p.a. (%)": (curve.iloc[-1] ** (1 / yrs) - 1) * 100, "Volatilität (%)": vol * 100,
        "Sharpe": r.mean() * 252 / vol if vol > 0 else 0.0, "Max. Drawdown (%)": dd.min() * 100,
        "Gebühren (€)": fees, "Steuern (€)": taxes, "Trades": trades,
    }
    return {"met": met, "equity": v, "dd": dd}


def simulate_pair(d, P, c, monthly, init):
    """Pairs Trading: Long/Short auf Basis des Z-Scores des log. Kursverhältnisses A/B."""
    idx, n = d.index, len(d)
    pa, pb = d.iloc[:, 0].values, d.iloc[:, 1].values
    s_ = np.log(d.iloc[:, 0] / d.iloc[:, 1])
    z = ((s_ - s_.rolling(P["pw"]).mean()) / s_.rolling(P["pw"]).std()).shift(1)  # nur Daten bis gestern
    zv = z.values
    m = idx.to_period("M")
    first_m = np.r_[True, m[1:] != m[:-1]]
    rate = c["pct"] + c["spread"]
    cash, contributed = init, init
    qa = qb = eA = eB = acc = 0.0
    state, trades, year = 0, 0, idx[0].year
    fees = taxes = ytd_gain = ytd_tax = 0.0
    vals, contrib, pos = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if idx[i].year != year:
            year, ytd_gain, ytd_tax = idx[i].year, 0.0, 0.0
        add = monthly if (i > 0 and first_m[i]) else 0.0
        cash += add
        contributed += add
        contrib[i] = add
        if state != 0:  # Leihgebühr für die Short-Seite
            lf = (max(-qa, 0) * pa[i] + max(-qb, 0) * pb[i]) * c["borrow"] / 252
            cash -= lf
            fees += lf
            acc -= lf
        zi, want = zv[i], state
        if not np.isnan(zi):
            if state == 0:
                want = -1 if zi > P["pe"] else 1 if zi < -P["pe"] else 0
            elif (state == -1 and (zi < P["px"] or (P["ps"] > 0 and zi > P["ps"]))) or \
                 (state == 1 and (zi > -P["px"] or (P["ps"] > 0 and zi < -P["ps"]))):
                want = 0
        if want != state:
            if state != 0:  # Position schließen
                vA, vB = qa * pa[i], qb * pb[i]
                fee = 2 * c["fix"] + (abs(vA) + abs(vB)) * rate
                cash += vA + vB - fee
                fees += fee
                trades += 2
                pnl = (vA - eA) + (vB - eB) - fee + acc
                if c["tax"] > 0:
                    ytd_gain += pnl
                    due = c["tax"] * max(0.0, ytd_gain - c["allow"])
                    tax, ytd_tax = due - ytd_tax, due
                    cash -= tax
                    taxes += tax
                qa = qb = 0.0
                state = 0
            if want != 0 and cash > c["min_trade"]:  # neue Position: want=+1 -> A long / B short
                N = P["pl"] / 100 * cash
                qa, qb = want * N / pa[i], -want * N / pb[i]
                eA, eB = qa * pa[i], qb * pb[i]
                fee = 2 * c["fix"] + 2 * N * rate
                cash -= eA + eB + fee
                fees += fee
                trades += 2
                acc = -fee
                state = want
        vals[i] = cash + qa * pa[i] + qb * pb[i]
        pos[i] = state
    res = finish(vals, contrib, idx, contributed, fees, taxes, trades)
    res["z"] = z
    res["pos"] = pd.Series(pos, idx)
    res["data"] = d
    return res


# ---------- Speichern ----------
def read_saved():
    return st.session_state.setdefault("saved", [])


def write_saved(items):
    st.session_state["saved"] = items


PK = {"SMA-Crossover": ["f", "s"], "Momentum": ["n"], "RSI Mean-Reversion": ["rn", "lo", "hi"],
      "60/40 Rebalancing": ["reb"], "Buy & Hold": [], "Pairs Trading": ["pw", "pe", "px", "ps", "pl"]}
PNAMES = {"f": "SMA schnell", "s": "SMA langsam", "n": "Momentum-Lookback", "rn": "RSI-Periode",
          "lo": "RSI kaufen unter", "hi": "RSI verkaufen über", "reb": "Rebalancing",
          "pw": "Paare: Fenster (Tage)", "pe": "Paare: Einstieg (Std.abw.)", "px": "Paare: Ausstieg (Std.abw.)",
          "ps": "Paare: Stop-Loss (Std.abw., 0=aus)", "pl": "Paare: Kapital pro Seite (%)"}


def sub(names):
    """Nur die Parameter der tatsächlich genutzten Strategien."""
    return {PNAMES[k]: P[k] for s_ in names for k in PK.get(s_, [])}


# ---------- Sidebar ----------
def tog(label, on, val, step, unit):
    a = st.checkbox(label, on)
    v = st.number_input(f"{label} ({unit})", 0.0, value=val, step=step, disabled=not a)
    return v if a else 0.0


with st.sidebar:
    st.header("Daten")
    t_r = st.text_input("Aktie/ETF (Ticker)", "URTH")
    t_b = st.text_input("Anleihen-ETF (nur für 60/40)", "AGG")
    start = st.date_input("Start", dt.date(2013, 1, 1), min_value=dt.date(1990, 1, 1))
    end = st.date_input("Ende", dt.date.today())

    st.header("Kapital")
    init = st.number_input("Startkapital (€)", 0.0, value=10000.0, step=1000.0)
    monthly = st.number_input("Monatliche Sparrate (€)", 0.0, value=0.0, step=50.0)

    st.header("Kosten (einzeln schaltbar)")
    cost = {
        "fix": tog("Fixe Ordergebühr", True, 1.0, 0.5, "€ pro Order"),
        "pct": tog("Prozentuale Gebühr", False, 0.1, 0.05, "% vom Volumen") / 100,
        "spread": tog("Spread/Slippage", False, 0.05, 0.01, "% pro Trade") / 100,
        "borrow": tog("Leihgebühr Leerverkauf (Pairs)", True, 0.5, 0.1, "% p.a. auf Short-Wert") / 100,
        "ter": tog("Fondskosten (TER)", True, 0.2, 0.05, "% p.a.") / 100,
        "tax": tog("Steuer auf Gewinne", False, 26.375, 0.5, "%") / 100,
        "allow": tog("Sparer-Pauschbetrag", False, 1000.0, 100.0, "€ pro Jahr"),
        "min_trade": 10.0,
    }

    st.header("Strategien")
    P = {}
    with st.expander("SMA-Crossover"):
        P["f"] = st.number_input("Schnelle SMA (Tage)", 5, 400, 50)
        P["s"] = st.number_input("Langsame SMA (Tage)", 10, 600, 200)
    with st.expander("Momentum"):
        P["n"] = st.number_input("Lookback (Handelstage)", 10, 500, 250)
    with st.expander("RSI Mean-Reversion"):
        P["rn"] = st.number_input("RSI-Periode", 2, 50, 14)
        P["lo"] = st.number_input("Kaufen unter", 1, 50, 30)
        P["hi"] = st.number_input("Verkaufen über", 50, 99, 70)
    with st.expander("Pairs Trading"):
        P["pw"] = st.number_input("Fenster für Mittelwert/Streuung (Tage)", 10, 500, 60)
        P["pe"] = st.number_input("Einstieg ab Abweichung von (Standardabw.)", 0.5, 6.0, 2.0, 0.1)
        P["px"] = st.number_input("Ausstieg bei Rückkehr auf (Standardabw.)", 0.0, 5.0, 0.5, 0.1)
        P["ps"] = st.number_input("Stop-Loss ab Abweichung von (0 = aus)", 0.0, 10.0, 0.0, 0.5)
        P["pl"] = st.number_input("Kapitaleinsatz pro Seite (% des Depots)", 10.0, 100.0, 50.0, 5.0)
    with st.expander("60/40 Rebalancing"):
        P["reb"] = st.selectbox("Rebalancing", ["Jährlich", "Monatlich"])

tab_sim, tab_saved = st.tabs(["Simulation", "Gespeicherte Ergebnisse"])

# ---------- Simulation ----------
with tab_sim:
    st.caption("Lehr-/Experimentierwerkzeug, keine Anlageberatung. Vergangene Renditen sind keine Garantie.")
    sel = st.multiselect("Strategien (jede wird einzeln simuliert)", STRATS, ["Buy & Hold", "SMA-Crossover"])
    combo_on = st.checkbox("Zusätzlich: Strategien kombinieren")
    combo = []
    if combo_on:
        combo = st.multiselect("Kombination aus", SIGNAL_STRATS[1:], ["SMA-Crossover", "Momentum"])
        mode = st.radio("Verknüpfung", ["Alle müssen Long sein (UND)", "Durchschnitt der Signale"], horizontal=True)

    PAIRS = {"Coca-Cola / PepsiCo": ("KO", "PEP"), "Visa / Mastercard": ("V", "MA"),
             "Home Depot / Lowe's": ("HD", "LOW"), "ExxonMobil / Chevron": ("XOM", "CVX"),
             "UPS / FedEx": ("UPS", "FDX"), "McDonald's / Yum! Brands": ("MCD", "YUM")}
    pair_names = st.multiselect("Pairs Trading: Firmenpaare (jedes Paar wird einzeln simuliert)", list(PAIRS),
                                ["Coca-Cola / PepsiCo"])
    pair_list = [PAIRS[k] for k in pair_names]
    if st.checkbox("Eigenes Paar hinzufügen"):
        c1, c2 = st.columns(2)
        ta, tb = c1.text_input("Ticker A", "GS").strip().upper(), c2.text_input("Ticker B", "MS").strip().upper()
        if ta and tb and ta != tb:
            pair_list.append((ta, tb))
    if P["px"] >= P["pe"]:
        st.warning("Der Ausstiegswert sollte kleiner sein als der Einstiegswert.")

    if st.button("Simulation starten", type="primary"):
        try:
            p = load(t_r.strip(), start, end)
            b = load(t_b.strip(), start, end) if "60/40 Rebalancing" in sel else None
        except Exception as e:
            st.error(f"Daten konnten nicht geladen werden: {e}")
            st.stop()
        if len(p) < 30:
            st.error("Zu wenige Kursdaten für diesen Zeitraum/Ticker.")
            st.stop()
        df = pd.DataFrame({"r": p})
        df["b"] = b if b is not None else 1.0
        df = df.dropna()
        p = df["r"]
        shown = df[["r"]].rename(columns={"r": t_r.strip()})
        if b is not None:
            shown["Anleihen: " + t_b.strip()] = df["b"]
        st.session_state["data"] = shown
        net = (1 - cost["ter"]) ** (np.arange(len(df)) / 252)  # TER als laufender Abzug
        px = df.mul(net, axis=0)

        runs = {}
        for name in sel:
            wr, wb, fq = target(name, p, P)
            runs[name] = simulate(px, wr, wb, fq, cost, monthly, init)
            runs[name]["params"] = sub([name])
        if combo_on and len(combo) >= 2:
            ex = pd.concat([target(s, p, P)[0] for s in combo], axis=1)
            w = ex.min(axis=1) if mode.startswith("Alle") else ex.mean(axis=1)
            kname = "Kombi: " + " + ".join(combo) + (" (UND)" if mode.startswith("Alle") else " (Ø)")
            runs[kname] = simulate(px, w, w * 0, None, cost, monthly, init)
            runs[kname]["params"] = {**sub(combo), "Verknüpfung": mode}
        for a, b2 in pair_list:
            try:
                d = pd.concat([load(a, start, end), load(b2, start, end)], axis=1, keys=[a, b2]).dropna()
            except Exception as e:
                st.error(f"Paar {a}/{b2}: Daten konnten nicht geladen werden ({e})")
                continue
            if len(d) < P["pw"] + 30:
                st.warning(f"Paar {a}/{b2}: zu wenige gemeinsame Kursdaten.")
                continue
            r_ = simulate_pair(d, P, cost, monthly, init)
            r_["params"] = {**sub(["Pairs Trading"]), "Paar": f"{a}/{b2}"}
            r_["settings_override"] = {"ticker": f"{a}/{b2}", "start": str(d.index[0].date()),
                                       "end": str(d.index[-1].date())}
            runs[f"Pairs: {a}/{b2}"] = r_
        if not runs:
            st.warning("Bitte mindestens eine Strategie oder ein Paar auswählen.")
            st.stop()
        st.session_state["runs"] = runs
        st.session_state["info"] = {
            "ticker": t_r.strip(), "anleihen_ticker": t_b.strip() if "60/40 Rebalancing" in sel else "-",
            "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "startkapital": init, "sparrate": monthly,
            "cost": {"Fixe Gebühr €": cost["fix"], "Prozentuale Gebühr %": cost["pct"] * 100,
                     "Spread %": cost["spread"] * 100, "TER % p.a.": cost["ter"] * 100,
                     "Steuer %": cost["tax"] * 100, "Pauschbetrag €": cost["allow"],
                     "Leihgebühr % p.a.": cost["borrow"] * 100}}

    runs = st.session_state.get("runs")
    if runs:
        data = st.session_state.get("data")
        if data is not None:
            with st.expander("Verwendete Kursdaten", expanded=True):
                st.caption("Quelle: Yahoo Finance (yfinance), dividendenbereinigte Schlusskurse in der Handelswährung des Tickers.")
                st.write(f"Zeitraum: {data.index[0].date()} bis {data.index[-1].date()} ({len(data)} Handelstage)")
                st.line_chart(data)
                st.dataframe(pd.concat([data.head(5), data.tail(5)]).round(2))
                st.download_button("Kursdaten als CSV", data.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                                   "kursdaten.csv", "text/csv")
        st.subheader("Vergleich")
        st.dataframe(pd.DataFrame({k: v["met"] for k, v in runs.items()}).T.round(2), use_container_width=True)
        st.write("**Depotwert (€)**")
        st.line_chart(pd.DataFrame({k: v["equity"] for k, v in runs.items()}))
        st.write("**Drawdown**")
        st.line_chart(pd.DataFrame({k: v["dd"] for k, v in runs.items()}))
        for k, v in runs.items():
            if "z" in v:
                with st.expander(f"Details {k}: Z-Score, Positionen und Kursdaten"):
                    pe_, px_ = st.session_state.get("pe_shown", P["pe"]), P["px"]
                    zz = pd.DataFrame({"Z-Score": v["z"], "Einstieg +": P["pe"], "Einstieg -": -P["pe"],
                                       "Ausstieg +": px_, "Ausstieg -": -px_})
                    st.line_chart(zz)
                    st.caption("Positionen: +1 = A long / B short, -1 = A short / B long, 0 = nicht investiert. "
                               "Quelle der Kurse: Yahoo Finance, dividendenbereinigt.")
                    st.line_chart(v["pos"].rename("Position"))
                    st.dataframe(pd.concat([v["data"].head(5), v["data"].tail(5)]).round(2))
                    st.download_button("Kursdaten als CSV", v["data"].to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                                       f"kursdaten_{k.replace('/', '_').replace(' ', '')}.csv", "text/csv", key=f"dl_{k}")

        st.subheader("Ergebnisse festhalten")
        pick = st.multiselect("Welche Läufe speichern?", list(runs), list(runs))
        note = st.text_input("Notiz (z. B. 'mit Steuer, 50 € Sparplan')")
        if st.button("Speichern") and pick:
            items = read_saved()
            for k in pick:
                eq = runs[k]["equity"].resample("W").last()
                items.append({
                    "label": f"{k}" + (f" | {note}" if note else ""),
                    "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "params": runs[k].get("params", {}),
                    "settings": {**st.session_state["info"], **runs[k].get("settings_override", {})},
                    "met": {a: float(x) for a, x in runs[k]["met"].items()},
                    "equity": {d.strftime("%Y-%m-%d"): round(float(x), 2) for d, x in eq.items()},
                })
            write_saved(items)
            st.success(f"{len(pick)} Ergebnis(se) gespeichert. Zum Sichern im Tab 'Gespeicherte Ergebnisse' als Datei exportieren.")

# ---------- Gespeicherte Ergebnisse ----------
with tab_saved:
    up = st.file_uploader("Ergebnisdateien importieren (.json von anderen Geräten)", type="json",
                          accept_multiple_files=True)
    if up:
        cur = read_saved()
        known = {(it["time"], it["label"]) for it in cur}
        for f in up:
            for it in json.load(f):
                if (it["time"], it["label"]) not in known:
                    cur.append(it)
                    known.add((it["time"], it["label"]))
        write_saved(cur)
    items = read_saved()
    if not items:
        st.info("Noch nichts gespeichert.")
    else:
        rows = pd.DataFrame([{"Nr": i, "Label": it["label"], "Zeit": it["time"],
                              "Ticker": it["settings"]["ticker"], "Zeitraum": f'{it["settings"]["start"]} – {it["settings"]["end"]}',
                              "Kapital": f'{it["settings"].get("startkapital")} € + {it["settings"].get("sparrate")} €/Monat',
                              "Kosten": str(it["settings"]["cost"]), "Parameter": str(it.get("params", {})), **{a: round(x, 2) for a, x in it["met"].items()}}
                             for i, it in enumerate(items)]).set_index("Nr")
        st.dataframe(rows, use_container_width=True)
        st.download_button("Alle Ergebnisse exportieren (.json)", json.dumps(items, ensure_ascii=False, indent=1),
                           f"backtest_ergebnisse_{dt.date.today()}.json", "application/json")
        st.download_button("Tabelle als CSV (Excel)", rows.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                           "backtest_tabelle.csv", "text/csv")
        chosen = st.multiselect("Zum Vergleichen auswählen (Nr)", list(rows.index))
        if chosen:
            chart = pd.DataFrame({f'{i}: {items[i]["label"]}': pd.Series(items[i]["equity"]) for i in chosen})
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart)
            if st.button("Ausgewählte löschen"):
                write_saved([it for i, it in enumerate(items) if i not in chosen])
                st.rerun()
