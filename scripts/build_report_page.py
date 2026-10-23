"""
Build the shareable HTML report from data/results.json.

    python scripts/build_report_page.py
    -> data/predictions_report.html

Same data the control panel uses, so the two can never disagree.
"""
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results.json")
BACKTEST = os.path.join(BASE, "data", "history", "backtest.json")
GRADED = os.path.join(BASE, "data", "history", "graded.json")
OUT = os.path.join(BASE, "data", "predictions_report.html")

LEAGUE_ORDER = ["Champions League", "Europa League", "Premier League",
                "Serie A", "Bundesliga", "La Liga", "Ligue 1"]
MARKET_LABEL = {"1x2": "1X2", "ou25": "O/U 2.5", "btts": "BTTS"}
OUTCOME_LABEL = {"home": "Home", "draw": "Draw", "away": "Away",
                 "over": "Over", "under": "Under", "yes": "Yes", "no": "No"}


def esc(s):
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def pct(v, digits=0):
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def pick_1x2(m):
    p = m["markets"]["1x2"]
    opts = [("home", p.get("home") or 0), ("draw", p.get("draw") or 0), ("away", p.get("away") or 0)]
    return max(opts, key=lambda x: x[1])


def accuracy_section():
    """Out-of-time accuracy + whatever has been graded so far."""
    bt = gd = None
    if os.path.exists(BACKTEST):
        with open(BACKTEST, encoding="utf-8") as f:
            bt = json.load(f)
    if os.path.exists(GRADED):
        with open(GRADED, encoding="utf-8") as f:
            gd = json.load(f)

    if not bt and not gd:
        return ""

    labels = {"1x2": "1X2", "ou25": "O/U 2.5", "ou15": "O/U 1.5", "ou35": "O/U 3.5", "btts": "BTTS"}
    rows = []
    if bt:
        for k in ("1x2", "ou25", "ou15", "ou35", "btts"):
            m = bt["markets"].get(k)
            if not m:
                continue
            graded_here = (gd or {}).get("totals", {}).get(k, {})
            live = (f"{graded_here['accuracy'] * 100:.1f}% "
                    f"<span class='sub'>({graded_here['correct']}/{graded_here['graded']})</span>"
                    if graded_here.get("graded") else "<span class='dash'>—</span>")
            rows.append(f"""<tr>
              <td><b>{labels[k]}</b></td>
              <td class="num">{m['accuracy'] * 100:.1f}%</td>
              <td class="num">{m['baseline'] * 100:.1f}%</td>
              <td class="num"><span class="edge {'pos' if m['lift'] >= 0 else 'neg'}">{m['lift'] * 100:+.1f}</span></td>
              <td class="num">{live}</td>
              <td class="num sub">{m['n']}</td>
            </tr>""")

    method = bt["method"] if bt else ""
    stamp = bt["generated"] if bt else ""
    return f"""
  <section>
    <h2>How good is the model, really?</h2>
    <div class="h2sub">{esc(method)} · {esc(stamp)}</div>
    <div class="card"><table>
      <thead><tr><th>Market</th><th class="num">Out-of-time</th><th class="num">Always-pick</th>
      <th class="num">Lift (pts)</th><th class="num">Graded live</th><th class="num">n</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    <div class="callout" style="margin-top:12px">
      <b>Read it this way:</b> “always-pick” is the accuracy of blindly choosing the most common
      outcome. A lift near zero means the market adds nothing — O/U 1.5 is exactly that case, since
      over 1.5 already happens ~78% of the time. 1X2 and BTTS are where the signal actually is.
    </div>
  </section>"""


def build():
    with open(RESULTS, encoding="utf-8") as f:
        data = json.load(f)

    matches = data["matches"]
    vb = data.get("value_bets", [])
    with_odds = data.get("with_odds", 0)
    credible = [v for v in vb if 0.05 < v["edge"] <= 0.15 and 0.40 <= v["model"] <= 0.85]
    leagues = {m["league"] for m in matches}
    avg_edge = (sum(abs(v["edge"]) for v in vb) / len(vb)) if vb else 0

    kpis = [
        ("Fixtures scored", len(matches), "7 markets each"),
        ("Live odds", with_odds, "1X2 · O/U 2.5 · BTTS"),
        ("Edges above 5%", len(vb), f"{len(credible)} in the credible 5–15% band"),
        ("Average edge", f"{avg_edge * 100:.1f}%", "absolute, all markets"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><span>{k}</span><b>{v}</b><small>{n}</small></div>' for k, v, n in kpis)

    # ---- 1X2 table -------------------------------------------------------
    # Fixtures with a live price first (biggest edge on top), then the ones the
    # odds feed has not opened yet — otherwise the page leads with a wall of
    # "no odds" Champions League rows.
    scored, unscored = [], []
    for m in matches:
        p = m["markets"]["1x2"]
        side, prob = pick_1x2(m)
        name = m["home"] if side == "home" else m["away"] if side == "away" else "Draw"
        mk = m["odds"]["markets"].get("1x2") if m.get("odds") else None
        e = m["edges"].get(f"1x2_{side}")
        h, d, a = (p.get("home") or 0) * 100, (p.get("draw") or 0) * 100, (p.get("away") or 0) * 100
        edge_cls = "hot" if (e or 0) >= 0.25 else "pos" if (e or 0) > 0.02 else "neg"
        market_cell = (f"{pct(mk['prob'].get('home'))} · {pct(mk['prob'].get('draw'))} · "
                       f"{pct(mk['prob'].get('away'))}") if mk else '<span class="dash">no odds</span>'
        edge_cell = (f'<span class="edge {edge_cls}">{e * 100:+.1f}%</span>'
                     if e is not None else '<span class="dash">—</span>')
        price = ((mk or {}).get("price") or {}).get(side)
        price_cell = f' <span class="price">@{price:.2f}</span>' if price else ""
        row = f"""<tr>
          <td><div class="tm"><b>{esc(m['home'])}</b><span>{esc(m['away'])}</span>
              <div class="lg">{esc(m['league'])}</div></div></td>
          <td><div class="seg">
            <div class="s1" style="width:{h}%">{f'{h:.0f}%' if h >= 11 else ''}</div>
            <div class="sx" style="width:{d}%">{f'{d:.0f}%' if d >= 11 else ''}</div>
            <div class="s2" style="width:{a}%">{f'{a:.0f}%' if a >= 11 else ''}</div>
          </div></td>
          <td><span class="badge {side}">{esc(name)}</span> <em>{pct(prob)}</em>{price_cell}</td>
          <td class="mk">{market_cell}</td>
          <td class="num">{edge_cell}</td>
        </tr>"""
        if e is None:
            unscored.append(row)
        else:
            scored.append((e, row))
    scored.sort(key=lambda x: -x[0])
    rows_1x2 = [r for _, r in scored] + unscored

    # ---- value bets ------------------------------------------------------
    rows_vb = []
    for v in vb[:40]:
        home, away = v["match"].split(" vs ", 1)
        cls = "hot" if v["edge"] >= 0.25 else "pos"
        rows_vb.append(f"""<tr>
          <td><div class="tm"><b>{esc(home)}</b><span>{esc(away)}</span>
              <div class="lg">{esc(v['league'])}</div></div></td>
          <td class="mk">{MARKET_LABEL.get(v['market'], v['market'])}</td>
          <td><span class="badge">{OUTCOME_LABEL.get(v['outcome'], v['outcome'])}</span></td>
          <td class="num">{pct(v['model'])}</td>
          <td class="num">{pct(v['market_prob'])}</td>
          <td class="num"><span class="edge {cls}">+{v['edge'] * 100:.1f}%</span></td>
          <td class="num">{v['price']:.2f}</td>
        </tr>""")

    # ---- per-league cards ------------------------------------------------
    per_league = {}
    for m in matches:
        per_league.setdefault(m["league"], []).append(m)

    league_html = []
    ordered = [l for l in LEAGUE_ORDER if l in per_league] + \
              [l for l in per_league if l not in LEAGUE_ORDER]
    for league in ordered:
        cards = []
        for m in per_league[league]:
            p, g, b, dc = m["markets"]["1x2"], m["markets"], m["markets"]["btts"], m["markets"]["dc"]
            mk = m["odds"]["markets"].get("ou25") if m.get("odds") else None
            market_line = (f'<div class="mline"><span class="k">Market</span>'
                           f'<span class="v">O2.5 {pct(mk["prob"].get("over"))}</span></div>'
                           if mk else "")
            cards.append(f"""<div class="mcard">
              <div class="mhead"><span class="mname">{esc(m['home'])} <em>vs</em> {esc(m['away'])}</span></div>
              <div class="mline"><span class="k">1X2</span><span class="v">{pct(p.get('home'))} · X {pct(p.get('draw'))} · {pct(p.get('away'))}</span></div>
              <div class="mline"><span class="k">Goals</span><span class="v">O1.5 {pct(g['ou15'].get('over'))} · O2.5 {pct(g['ou25'].get('over'))} · O3.5 {pct(g['ou35'].get('over'))}</span></div>
              <div class="mline"><span class="k">BTTS</span><span class="v">Yes {pct(b.get('yes'))}</span></div>
              <div class="mline"><span class="k">Derived</span><span class="v">DC 1X {pct(dc.get('1x'))} · 12 {pct(dc.get('12'))} · X2 {pct(dc.get('x2'))}</span></div>
              {market_line}
            </div>""")
        league_html.append(f"""<details {'open' if league in ('Champions League', 'Europa League') else ''}>
          <summary><span class="lname">{esc(league)}</span><span class="lcount">{len(per_league[league])} matches</span></summary>
          <div class="mgrid">{''.join(cards)}</div>
        </details>""")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Football Predictions — model vs market</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f3f4f7;color:#0d1017;line-height:1.55;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter","PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:44px 24px 80px}}
.eyebrow{{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:#878e9b;font-weight:700}}
h1{{font-size:34px;line-height:1.15;margin:10px 0 12px;letter-spacing:-.03em}}
.meta{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:12.5px;color:#4b5563}}
.dot{{width:4px;height:4px;border-radius:50%;background:#c4c8d0}}
.pill{{background:#fff;border:1px solid #e5e7eb;border-radius:999px;padding:3px 11px;font-size:11.5px;color:#374151}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0 30px}}
.kpi{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:15px 17px}}
.kpi span{{font-size:11px;color:#878e9b;font-weight:700;text-transform:uppercase;letter-spacing:.07em}}
.kpi b{{display:block;font-size:26px;letter-spacing:-.03em;margin:4px 0 2px;font-variant-numeric:tabular-nums}}
.kpi small{{font-size:11.5px;color:#9aa0aa}}
section{{margin:34px 0}}
h2{{font-size:20px;margin:0 0 4px;letter-spacing:-.02em}}
.h2sub{{font-size:12.5px;color:#878e9b;margin-bottom:14px}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:16px;overflow:hidden}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:#878e9b;
   font-weight:700;padding:11px 16px;background:#fafafb;border-bottom:1px solid #eef0f3;white-space:nowrap}}
td{{padding:12px 16px;border-bottom:1px solid #f2f4f6;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
.tm{{display:flex;flex-direction:column;gap:1px;min-width:170px}}
.tm b{{font-weight:600}}
.tm span{{font-size:11.5px;color:#878e9b}}
.lg{{font-size:10.5px;color:#a9afb9;margin-top:2px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.seg{{display:flex;height:24px;border-radius:6px;overflow:hidden;min-width:210px;background:#f1f2f5}}
.seg div{{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#fff;white-space:nowrap}}
.seg .s1{{background:#4f46e5}} .seg .sx{{background:#c7cbd4;color:#3f4652}} .seg .s2{{background:#0f766e}}
.badge{{display:inline-block;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;
  background:#eef0ff;color:#3730a3;white-space:nowrap}}
.badge.away{{background:#e8f5f2;color:#0f5f59}} .badge.draw{{background:#f1f2f5;color:#4b5563}}
em{{font-style:normal;color:#878e9b;font-weight:600;font-variant-numeric:tabular-nums;font-size:11.5px}}
.mk{{font-size:11.5px;color:#3f4652;font-variant-numeric:tabular-nums;white-space:nowrap}}
.edge{{font-weight:700;font-variant-numeric:tabular-nums}}
.edge.pos{{color:#0f766e}} .edge.hot{{color:#b45309}} .edge.neg{{color:#9aa0aa}}
.dash{{color:#c2c7d0}}
.price{{display:inline-block;margin-left:4px;font-size:10.5px;font-weight:600;color:#3f4652;
  background:#f1f2f5;border-radius:5px;padding:1px 5px;font-variant-numeric:tabular-nums}}
.callout{{background:#fff8ed;border:1px solid #fcd9a4;border-radius:14px;padding:15px 17px;font-size:13px}}
.callout b{{color:#92400e}}
details{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;margin-bottom:10px;overflow:hidden}}
summary{{cursor:pointer;padding:13px 17px;display:flex;justify-content:space-between;align-items:center;
  font-weight:600;font-size:14px;list-style:none}}
summary::-webkit-details-marker{{display:none}}
summary::before{{content:"▸";color:#9aa0aa;margin-right:9px;font-size:11px}}
details[open] summary::before{{content:"▾"}}
.lname{{flex:1}} .lcount{{font-size:11.5px;color:#878e9b;font-weight:500}}
.mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:11px;padding:0 17px 17px}}
.mcard{{border:1px solid #eef0f3;border-radius:11px;padding:11px 13px;background:#fcfcfd}}
.mhead{{margin-bottom:7px}}
.mname{{font-weight:650;font-size:13px}}
.mname em{{color:#9aa0aa;font-style:normal;font-weight:400;font-size:11px;padding:0 3px}}
.mline{{display:flex;gap:9px;font-size:11.5px;padding:1.5px 0}}
.mline .k{{color:#8b919c;min-width:52px;flex:none}}
.mline .v{{color:#2f333a;font-variant-numeric:tabular-nums}}
footer{{margin-top:44px;font-size:11.5px;color:#8b919c;border-top:1px solid #e5e7eb;padding-top:16px}}
@media(max-width:820px){{.kpis{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:26px}}
  .mgrid{{grid-template-columns:1fr}}.card{{overflow-x:auto}}table{{min-width:640px}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">AIFootballPredictions · model vs market</div>
  <h1>Predictions &amp; live-odds comparison</h1>
  <div class="meta">
    <span>Generated {esc(data['generated'])}</span><span class="dot"></span>
    <span>1X2 · O/U 1.5 · O/U 2.5 · O/U 3.5 · BTTS · Double chance · DNB</span><span class="dot"></span>
    <span class="pill">{len(leagues)} competitions</span>
  </div>

  <div class="kpis">{kpi_html}</div>

  <section>
    <h2>Read the edges with care</h2>
    <div class="h2sub">Every “edge” below is model probability minus the de-vigged market probability.</div>
    <div class="callout">
      <b>Why the numbers still look extreme:</b> the pipeline no longer leaks the scoreline, so the
      average edge fell from 25% to {avg_edge * 100:.1f}% — but the ensemble is still uncalibrated, and a
      model that gets 1X2 right 54% of the time will happily print 69% on a match it has misread.
      Treat the edge column as a <b>ranking of disagreement with the market</b>, not as expected value.
      The rows worth a second look sit in the <b>5–15%</b> band with a model probability between
      <b>40% and 85%</b> — {len(credible)} of {len(vb)} qualify.
    </div>
  </section>

  {accuracy_section()}

  <section>
    <h2>1X2 — model vs market</h2>
    <div class="h2sub">Segmented bar is the model's home / draw / away split. Market column is the de-vigged consensus.</div>
    <div class="card"><table>
      <thead><tr><th>Match</th><th>Home · Draw · Away</th><th>Pick</th><th>Market</th><th class="num">Edge</th></tr></thead>
      <tbody>{''.join(rows_1x2)}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Top edges across all markets</h2>
    <div class="h2sub">Top 40 by edge. 1X2, O/U 2.5 and BTTS are compared against live Flashscore prices.</div>
    <div class="card"><table>
      <thead><tr><th>Match</th><th>Market</th><th>Selection</th><th class="num">Model</th>
      <th class="num">Market</th><th class="num">Edge</th><th class="num">Price</th></tr></thead>
      <tbody>{''.join(rows_vb)}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Every fixture, every market</h2>
    <div class="h2sub">Grouped by competition.</div>
    {''.join(league_html)}
  </section>

  <footer>
    Sources: football-data.co.uk (results, features) · football-data.org (fixtures) ·
    Flashscore (live odds) · Wikipedia (Europa League fixtures).<br>
    Research output only — not betting advice. Probabilities are uncalibrated; past performance
    does not predict future results.
  </footer>
</div>
</body>
</html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Report -> {OUT}")
    print(f"matches={len(matches)} with_odds={with_odds} edges={len(vb)} credible={len(credible)}")


if __name__ == "__main__":
    build()
