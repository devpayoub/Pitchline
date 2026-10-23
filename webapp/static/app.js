/* AI Football Predictions — control panel front-end */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  let state = { status: "idle" };
  let logCursor = 0;
  let results = null;
  let accuracy = null;
  let live = null;
  let liveTimer = null;
  let liveStates = {};
  let activeTab = "1x2";
  let activeLeague = "all";
  let activeDay = "all";
  let sortMode = "edge";

  const LEAGUE_ORDER = ["Champions League", "Europa League", "Premier League",
    "Serie A", "Bundesliga", "La Liga", "Ligue 1"];

  const pct = v => (v === null || v === undefined) ? "—" : `${Math.round(v * 100)}%`;
  const one = v => (v === null || v === undefined) ? "—" : `${(v * 100).toFixed(1)}%`;

  /* ------------------------------------------------------------------ */
  /* state polling                                                       */
  /* ------------------------------------------------------------------ */
  async function pollState() {
    try {
      const r = await fetch("/api/state");
      const next = await r.json();
      const changed = next.status !== state.status || next.job !== state.job;
      state = next;
      renderChrome();
      if (next.status === "running") {
        $("#console").hidden = false;
        pollLogs();
      }
      if (changed && (next.status === "done" || next.status === "failed" || next.status === "stopped")) {
        pollLogs();
        if (next.results_ready) loadResults();
      }
    } catch (e) { /* server restarting */ }
  }

  async function pollLogs() {
    try {
      const r = await fetch(`/api/logs?since=${logCursor}`);
      const data = await r.json();
      if (data.lines && data.lines.length) appendLogs(data.lines);
      logCursor = data.total;
    } catch (e) { }
  }

  function appendLogs(lines) {
    const body = $("#consoleBody");
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
    const frag = document.createDocumentFragment();
    for (const l of lines) {
      const div = document.createElement("div");
      div.className = `l-${l.level || "out"}`;
      div.textContent = l.text;
      frag.appendChild(div);
    }
    body.appendChild(frag);
    while (body.childNodes.length > 1200) body.removeChild(body.firstChild);
    if (atBottom) body.scrollTop = body.scrollHeight;
  }

  /* ------------------------------------------------------------------ */
  /* chrome: status pill, buttons, progress                              */
  /* ------------------------------------------------------------------ */
  const LABEL = { idle: "Idle", running: "Running", done: "Done", failed: "Failed", stopped: "Stopped" };
  const JOBNAME = { download: "Downloading data", train: "Training models", predict: "Predicting & comparing" };

  function renderChrome() {
    const pill = $("#status");
    pill.className = `status ${state.status}`;
    $("span", pill).textContent = state.status === "running"
      ? (JOBNAME[state.job] || "Running") : LABEL[state.status] || "Idle";

    $("#stamp").textContent = state.finished
      ? `last run ${state.finished.slice(5, 16)}`
      : (state.started ? `started ${state.started.slice(5, 16)}` : "—");

    $$(".action").forEach(card => {
      const job = card.dataset.job;
      const btn = $("button[data-run]", card);
      const running = state.status === "running";
      card.classList.toggle("running", running && state.job === job);
      card.classList.toggle("dim", running && state.job !== job);
      btn.disabled = running;
      btn.textContent = (running && state.job === job) ? "Running…" : "Run";
    });

    $("#consoleJob").textContent = JOBNAME[state.job] || "Running";
    $("#consoleStep").textContent = `${state.step} / ${state.total}`;
    const p = state.total ? (state.step / state.total) * 100 : 0;
    $("#progressBar").style.width = `${p}%`;
    $("#btnStop").disabled = state.status !== "running";

    if (state.status === "running" && state.job) $("#console").hidden = false;
  }

  /* ------------------------------------------------------------------ */
  /* actions                                                             */
  /* ------------------------------------------------------------------ */
  async function runJob(job) {
    logCursor = 0;
    $("#consoleBody").innerHTML = "";
    $("#console").hidden = false;
    const r = await fetch(`/api/run/${job}`, { method: "POST" });
    const data = await r.json();
    if (!data.ok) alert(data.message || "Could not start");
    pollState();
  }

  $$("button[data-run]").forEach(btn =>
    btn.addEventListener("click", () => runJob(btn.dataset.run)));

  $("#btnStop").addEventListener("click", async () => {
    await fetch("/api/stop", { method: "POST" });
    pollState();
  });

  $("#btnFold").addEventListener("click", () => {
    $("#console").hidden = true;
  });

  $("#btnReload").addEventListener("click", loadResults);

  /* ------------------------------------------------------------------ */
  /* results                                                             */
  /* ------------------------------------------------------------------ */
  async function loadLive(force) {
    try {
      const r = await fetch(`/api/live${force ? "?force=1" : ""}`);
      live = await r.json();
      liveStates = {};
      for (const m of (live.matches || [])) {
        liveStates[`${m.home} vs ${m.away}`] = m;
      }
      if (activeTab === "live") renderTable();
      else if (results) renderTable();   // keep the ticks in the market tables fresh
    } catch (e) {
      live = { error: "live feed unreachable" };
    }
  }

  function startLivePolling() {
    if (liveTimer) return;
    // 30s on the Live tab, 60s elsewhere: the ticks in the market tables need to
    // stay current too, but there is no point hammering the feed.
    liveTimer = setInterval(() => {
      if (state.status === "running") return;
      loadLive(false);
    }, 45000);
  }

  async function loadAccuracy() {
    try {
      const r = await fetch("/api/accuracy");
      accuracy = await r.json();
      if (activeTab === "accuracy") renderTable();
    } catch (e) { }
  }

  async function loadResults() {
    try {
      const r = await fetch("/api/results");
      if (!r.ok) return;
      results = await r.json();
      renderResults();
    } catch (e) { }
  }

  function renderResults() {
    if (!results || !results.matches) return;
    const ms = results.matches;
    const leagues = [...new Set(ms.map(m => m.league))];
    const ordered = LEAGUE_ORDER.filter(l => leagues.includes(l))
      .concat(leagues.filter(l => !LEAGUE_ORDER.includes(l)));
    const withOdds = results.with_odds || 0;
    const vb = (results.value_bets || []).length;
    const credible = (results.value_bets || []).filter(isCredible).length;

    $("#resultsSub").textContent =
      `${ms.length} fixtures across ${leagues.length} competitions · model generated ${results.generated}`;
    $("#btnReport").hidden = !state.report_ready;

    $("#kpis").innerHTML = [
      ["Fixtures scored", ms.length, "7 markets each"],
      ["Live odds", withOdds, "1X2 · O/U 2.5 · BTTS"],
      ["Edges over 5%", vb, `${credible} in the credible 5–15% band`],
      ["Odds coverage", withOdds ? `${Math.round(withOdds / ms.length * 100)}%` : "—", `${ms.length - withOdds} without prices`],
    ].map(([k, v, s]) => `<div class="kpi"><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join("");

    $("#heroStats").innerHTML = [
      ["Fixtures", ms.length],
      ["Credible edges", credible],
      ["Leagues", leagues.length],
    ].map(([k, v]) => `<div class="hstat"><b>${v}</b><span>${k}</span></div>`).join("");

    // league filter chips
    const counts = {};
    ms.forEach(m => counts[m.league] = (counts[m.league] || 0) + 1);
    $("#filters").innerHTML =
      `<button class="chip ${activeLeague === "all" ? "active" : ""}" data-lg="all">All<span class="n">${ms.length}</span></button>` +
      ordered.map(l => `<button class="chip ${activeLeague === l ? "active" : ""}" data-lg="${esc(l)}">${esc(l)}<span class="n">${counts[l]}</span></button>`).join("");
    $$("#filters .chip").forEach(c =>
      c.addEventListener("click", () => { activeLeague = c.dataset.lg; renderResults(); }));

    renderDayFilters(ms);

    renderTable();
  }

  function dayKey(m) {
    const d = String(m.date || "").slice(0, 10);
    return d || "unknown";
  }

  function dayLabel(key) {
    if (key === "unknown") return "No date";
    const d = new Date(`${key}T12:00:00`);
    if (isNaN(d)) return key;
    return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
  }

  function renderDayFilters(ms) {
    const el = $("#dayFilters");
    // Live and Accuracy are their own views — a day filter makes no sense there.
    if (activeTab === "live" || activeTab === "accuracy") { el.innerHTML = ""; return; }

    const counts = {};
    ms.forEach(m => { const k = dayKey(m); counts[k] = (counts[k] || 0) + 1; });
    const keys = Object.keys(counts).filter(k => k !== "unknown").sort();
    if (counts.unknown) keys.push("unknown");
    if (keys.length < 2) { el.innerHTML = ""; return; }

    if (activeDay !== "all" && !keys.includes(activeDay)) activeDay = "all";

    el.innerHTML =
      `<span class="daylabel">Day</span>` +
      `<button class="chip ${activeDay === "all" ? "active" : ""}" data-day="all">All<span class="n">${ms.length}</span></button>` +
      keys.map(k => `<button class="chip ${activeDay === k ? "active" : ""}" data-day="${esc(k)}">${esc(dayLabel(k))}<span class="n">${counts[k]}</span></button>`).join("");

    $$("#dayFilters .chip").forEach(c =>
      c.addEventListener("click", () => { activeDay = c.dataset.day; renderResults(); }));
  }

  function visible() {
    const rows = results.matches.filter(m =>
      (activeLeague === "all" || m.league === activeLeague) &&
      (activeDay === "all" || dayKey(m) === activeDay));
    const byLeague = (a, b) => {
      const ia = LEAGUE_ORDER.indexOf(a.league), ib = LEAGUE_ORDER.indexOf(b.league);
      const la = ia < 0 ? 99 : ia, lb = ib < 0 ? 99 : ib;
      return la - lb || String(a.date).localeCompare(String(b.date));
    };
    if (sortMode === "league") return rows.sort(byLeague);
    if (sortMode === "kickoff") return rows.sort((a, b) => String(a.date).localeCompare(String(b.date)) || byLeague(a, b));
    // biggest edge first; fixtures without a market price fall to the bottom
    return rows.sort((a, b) => bestEdge(b) - bestEdge(a) || byLeague(a, b));
  }

  const isCredible = v => v.edge > 0.05 && v.edge <= 0.15 && v.model >= 0.40 && v.model <= 0.85;

  function bestEdge(m) {
    const vals = Object.values(m.edges || {});
    return vals.length ? Math.max(...vals) : -9;
  }

  function renderTable() {
    const wrap = $("#tableWrap");
    renderNote();
    if (activeTab === "live") {
      wrap.innerHTML = liveView();
      const btn = $("#btnLiveRefresh");
      if (btn) btn.addEventListener("click", () => loadLive(true));
      return;
    }
    if (activeTab === "accuracy") {
      wrap.innerHTML = accuracyView();
      const btn = $("#btnGrade");
      if (btn) btn.addEventListener("click", () => runJob("grade"));
      return;
    }
    const rows = visible();
    if (!rows.length) {
      wrap.innerHTML = `<div class="empty"><h3>Nothing to show yet</h3><p>Run <b>Predict &amp; compare</b> and the match cards will land here.</p></div>`;
      return;
    }
    wrap.innerHTML = `<table>${activeTab === "value" ? valueTable() : matchTable(rows)}</table>`;
  }

  const NOTES = {
    "1x2": "Home · draw · away from the ensemble. The market column is the de-vigged consensus of every bookmaker Flashscore lists — a big gap means the model disagrees with the price, not that the price is wrong.",
    goals: "Over/under probabilities come from three separate binary models. Over 1.5 sits near 80–99% for most fixtures because the base rate is high — read it as a sanity check, not a pick.",
    btts: "Both teams to score, compared with the live BTTS price.",
    live: "",
    accuracy: "",
    value: "Edges above 25% are almost always miscalibration or a club whose history comes from another division — not value. Trust the 5–15% band with a model probability between 40% and 85%.",
  };

  function renderNote() {
    const el = $("#tabNote");
    el.textContent = NOTES[activeTab] || "";
    el.hidden = !NOTES[activeTab];
  }

  function pick1x2(m) {
    const p = m.markets["1x2"] || {};
    const opts = [["home", p.home], ["draw", p.draw], ["away", p.away]];
    return opts.reduce((a, b) => (b[1] || 0) > (a[1] || 0) ? b : a, opts[0]);
  }

  function seg(p) {
    const h = (p.home || 0) * 100, d = (p.draw || 0) * 100, a = (p.away || 0) * 100;
    return `<div class="seg">
      <div class="s1" style="width:${h}%">${h >= 11 ? Math.round(h) + "%" : ""}</div>
      <div class="sx" style="width:${d}%">${d >= 11 ? Math.round(d) + "%" : ""}</div>
      <div class="s2" style="width:${a}%">${a >= 11 ? Math.round(a) + "%" : ""}</div>
    </div>`;
  }

  function matchTable(rows) {
    if (activeTab === "1x2") {
      return `<thead><tr>
        <th>Match</th><th>Home · Draw · Away</th><th>Pick</th><th>Market</th><th class="num">Edge</th>
      </tr></thead><tbody>` + rows.map(m => {
        const p = m.markets["1x2"] || {};
        const [side, prob] = pick1x2(m);
        const name = side === "home" ? m.home : side === "away" ? m.away : "Draw";
        const mk = m.odds && m.odds.markets["1x2"];
        const e = m.edges[`1x2_${side}`];
        return `<tr>
          <td><div class="tm"><b>${esc(m.home)}</b><span>${esc(m.away)}</span><div class="lg">${esc(m.league)}${m.source === "baseline" ? '<span class="base-tag">baseline</span>' : ""}</div></div></td>
          <td>${seg(p)}</td>
          <td><span class="pick"><span class="badge ${side}">${esc(name)}</span><em>${pct(prob)}</em>${priceOf(m, "1x2", side)}${liveTick(m, "1x2")}</span></td>
          <td class="mk">${mk ? `${pct(mk.prob.home)} <span>·</span> ${pct(mk.prob.draw)} <span>·</span> ${pct(mk.prob.away)}` : `<span class="dash">no odds</span>`}</td>
          <td class="num">${edgeCell(e)}</td>
        </tr>`;
      }).join("") + `</tbody>`;
    }

    if (activeTab === "goals") {
      return `<thead><tr>
        <th>Match</th><th>O/U 1.5</th><th>O/U 2.5</th><th>O/U 3.5</th><th class="num">Edge 2.5</th>
      </tr></thead><tbody>` + rows.map(m => {
        const g = k => {
          const o = m.markets[k];
          if (!o) return `<span class="dash">—</span>`;
          const over = (o.over || 0) >= 0.5;
          return `<span class="pick"><span class="badge ${over ? "" : "away"}">${over ? "Over" : "Under"}</span><em>${pct(over ? o.over : o.under)}</em>${priceOf(m, k, over ? "over" : "under")}${liveTick(m, k)}</span>`;
        };
        const side = (m.markets.ou25.over || 0) >= 0.5 ? "over" : "under";
        return `<tr>
          <td><div class="tm"><b>${esc(m.home)}</b><span>${esc(m.away)}</span><div class="lg">${esc(m.league)}${m.source === "baseline" ? '<span class="base-tag">baseline</span>' : ""}</div></div></td>
          <td>${g("ou15")}</td><td>${g("ou25")}</td><td>${g("ou35")}</td>
          <td class="num">${edgeCell(m.edges[`ou25_${side}`])}</td>
        </tr>`;
      }).join("") + `</tbody>`;
    }

    if (activeTab === "btts") {
      return `<thead><tr>
        <th>Match</th><th>Yes / No</th><th>Pick</th><th>Market</th><th class="num">Edge</th>
      </tr></thead><tbody>` + rows.map(m => {
        const b = m.markets.btts || {};
        const yes = (b.yes || 0) >= 0.5;
        const side = yes ? "yes" : "no";
        const mk = m.odds && m.odds.markets["btts"];
        const y = (b.yes || 0) * 100;
        return `<tr>
          <td><div class="tm"><b>${esc(m.home)}</b><span>${esc(m.away)}</span><div class="lg">${esc(m.league)}${m.source === "baseline" ? '<span class="base-tag">baseline</span>' : ""}</div></div></td>
          <td><div class="seg"><div class="s1" style="width:${y}%">${y >= 12 ? "Yes " + Math.round(y) + "%" : ""}</div>
              <div class="sx" style="width:${100 - y}%">${(100 - y) >= 12 ? "No " + Math.round(100 - y) + "%" : ""}</div></div></td>
          <td><span class="pick"><span class="badge ${yes ? "" : "away"}">${yes ? "Yes" : "No"}</span><em>${pct(yes ? b.yes : b.no)}</em>${priceOf(m, "btts", yes ? "yes" : "no")}${liveTick(m, "btts")}</span></td>
          <td class="mk">${mk ? `${pct(mk.prob.yes)} / ${pct(mk.prob.no)}` : `<span class="dash">no odds</span>`}</td>
          <td class="num">${edgeCell(m.edges[`btts_${side}`])}</td>
        </tr>`;
      }).join("") + `</tbody>`;
    }
    return "";
  }

  function valueTable() {
    let vb = (results.value_bets || []).filter(v => {
      if (activeLeague !== "all" && v.league !== activeLeague) return false;
      if (activeDay === "all") return true;
      const m = (results.matches || []).find(x => `${x.home} vs ${x.away}` === v.match);
      return m ? dayKey(m) === activeDay : false;
    });
    if (sortMode === "league") {
      vb = vb.slice().sort((a, b) => {
        const ia = LEAGUE_ORDER.indexOf(a.league), ib = LEAGUE_ORDER.indexOf(b.league);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || b.edge - a.edge;
      });
    }
    if (!vb.length) {
      return `<tbody><tr><td><div class="empty"><h3>No edges above 5%</h3>
        <p>The model agrees with the market on every fixture in this view.</p></div></td></tr></tbody>`;
    }
    const MKT = { "1x2": "1X2", ou25: "O/U 2.5", btts: "BTTS" };
    const OUT = { home: "Home", draw: "Draw", away: "Away", over: "Over", under: "Under", yes: "Yes", no: "No" };
    return `<thead><tr>
      <th>Match</th><th>Market</th><th>Selection</th><th class="num">Model</th>
      <th class="num">Market</th><th class="num">Edge</th><th class="num">Best price</th>
    </tr></thead><tbody>` + vb.map(v => {
      const [home, away] = v.match.split(" vs ");
      return `<tr>
        <td><div class="tm"><b>${esc(home)}</b><span>${esc(away)}</span><div class="lg">${esc(v.league)}</div></div></td>
        <td class="mk">${MKT[v.market] || v.market}</td>
        <td><span class="pick"><span class="badge">${OUT[v.outcome] || v.outcome}</span>${valueTick(v)}</span></td>
        <td class="num">${pct(v.model)}</td>
        <td class="num">${pct(v.market_prob)}</td>
        <td class="num">${edgeCell(v.edge)}</td>
        <td class="num">${v.price ? v.price.toFixed(2) : "—"}</td>
      </tr>`;
    }).join("") + `</tbody>`;
  }

  const MARKET_LABEL = { "1x2": "1X2", ou25: "O/U 2.5", ou15: "O/U 1.5", ou35: "O/U 3.5", btts: "BTTS" };

  const MARKET_SHORT = { "1x2": "1X2", ou25: "O/U 2.5", ou15: "O/U 1.5", ou35: "O/U 3.5", btts: "BTTS" };
  const PICK_LABEL = { home: "Home", draw: "Draw", away: "Away", over: "Over", under: "Under", yes: "Yes", no: "No" };

  function pickText(m, market, pick) {
    if (market === "1x2") return pick === "home" ? m.home : pick === "away" ? m.away : "Draw";
    return PICK_LABEL[pick] || pick;
  }

  const TICK = { won: "✓", lost: "✗", pending: "·" };

  /** Best available price for the picked side, e.g. "@1.85". */
  function priceOf(m, market, pick) {
    const mk = m.odds && m.odds.markets && m.odds.markets[market];
    const p = mk && mk.price ? mk.price[pick] : null;
    return p ? `<span class="price">@${p.toFixed(2)}</span>` : "";
  }

  /** Small ✓ / ✗ / · next to a pick, once the live feed has settled that market. */
  function liveTick(m, market) {
    const st = liveStates[`${m.home} vs ${m.away}`];
    if (!st) return "";
    const info = (st.markets || {})[market];
    if (!info || !info.state || info.state === "pending") return "";
    return `<span class="tick ${info.state}" title="${info.state === "won" ? "already won" : "already lost"}">${TICK[info.state]}</span>`;
  }

  function liveView() {
    if (!live) return `<div class="empty"><h3>Loading live scores…</h3></div>`;
    if (live.error) return `<div class="empty"><h3>Live feed unavailable</h3><p>${esc(live.error)}</p></div>`;

    const ms = live.matches || [];
    const totals = live.totals || {};
    const decided = Object.values(totals).reduce((a, t) => a + t.decided, 0);
    const won = Object.values(totals).reduce((a, t) => a + t.won, 0);
    const inPlay = ms.filter(m => m.status === "live").length;

    const kpis = [
      ["In play", inPlay, "matches running now"],
      ["Decided picks", decided, "markets already settled"],
      ["Correct", won, "green ticks so far"],
      ["Hit rate", decided ? `${Math.round(won / decided * 100)}%` : "—", "of settled picks"],
    ].map(([k, v, s]) => `<div class="kpi"><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join("");

    const groups = [
      ["live", "In play"],
      ["finished", "Finished today"],
      ["scheduled", "Later today"],
    ];

    let body = "";
    for (const [status, label] of groups) {
      const rows = ms.filter(m => m.status === status);
      if (!rows.length) continue;
      body += `<div class="lgroup"><div class="lgroup-head"><h3>${label}</h3><span>${rows.length}</span></div>`;
      body += rows.map(m => liveCard(m)).join("");
      body += `</div>`;
    }

    if (!body) {
      body = `<div class="acc-block"><div class="acc-empty">No archived fixture is on the board right now.
        Predictions for the next matchday will appear here once the matches kick off.</div></div>`;
    }

    return `<div class="live">
      <div class="live-head">
        <div>
          <h3>Live scores</h3>
          <p class="results-sub">Updated ${esc(live.generated || "")} · auto-refresh every 45s · archive ${esc(live.archive || "")}</p>
        </div>
        <button class="btn ghost" id="btnLiveRefresh">Refresh now</button>
      </div>
      <div class="kpis">${kpis}</div>
      ${body}
      <p class="live-note">A goals market settles as soon as the score passes the line — the tick appears while the match is
        still running. 1X2 and the “no” side of BTTS/Under only settle at full time.</p>
    </div>`;
  }

  function liveCard(m) {
    const chips = [];
    for (const [market, info] of Object.entries(m.markets || {})) {
      if (market === "value") continue;
      chips.push(`<span class="lchip ${info.state}">
        <em>${MARKET_SHORT[market] || market}</em>
        ${esc(pickText(m, market, info.pick))}
        ${info.price ? `<i>@${info.price.toFixed(2)}</i>` : ""}
        <b>${TICK[info.state] || "·"}</b></span>`);
    }
    for (const v of ((m.markets || {}).value || {}).picks || []) {
      chips.push(`<span class="lchip ${v.state} value">
        <em>${MARKET_SHORT[v.market] || v.market}</em>
        ${esc(PICK_LABEL[v.outcome] || v.outcome)}
        <b>${TICK[v.state] || "·"}</b>
        ${v.price ? `<i>@${v.price.toFixed(2)}</i>` : ""}</span>`);
    }

    const stage = m.status === "live"
      ? `<span class="lstage live">● ${esc(m.stage || "live")}</span>`
      : m.status === "finished"
        ? `<span class="lstage done">FT</span>`
        : `<span class="lstage soon">upcoming</span>`;

    return `<div class="lcard">
      <div class="lhead">
        <div class="lteams">
          <b>${esc(m.home)}</b><em>vs</em><b class="away">${esc(m.away)}</b>
          <span class="llg">${esc(m.league)}${m.source === "baseline" ? '<span class="base-tag">baseline</span>' : ""}</span>
        </div>
        <div class="lright">
          <span class="lscore">${m.hg}–${m.ag}</span>
          ${stage}
        </div>
      </div>
      <div class="lchips">${chips.join("")}</div>
    </div>`;
  }

  function accuracyView() {
    if (!accuracy) return `<div class="empty"><h3>Loading…</h3></div>`;
    const bt = accuracy.backtest, gd = accuracy.graded;
    const archives = accuracy.archives || [];
    let html = '<div class="acc">';

    // ---- out-of-time backtest ----
    html += `<div class="acc-block">
      <div class="acc-head"><h3>Out-of-time backtest</h3>
        <span>${bt ? esc(bt.method) + " · " + esc(bt.generated) : "not run yet — press Train models"}</span></div>`;
    if (bt && bt.markets) {
      html += '<div class="acc-grid">' + ["1x2", "ou25", "ou15", "ou35", "btts"].map(k => {
        const m = bt.markets[k];
        if (!m) return "";
        const lift = m.lift * 100;
        return `<div class="acc-card">
          <div class="k">${MARKET_LABEL[k]}</div>
          <b>${(m.accuracy * 100).toFixed(1)}%</b>
          <small>always-pick ${(m.baseline * 100).toFixed(1)}% ·
            <span class="lift ${lift >= 0 ? "up" : "down"}">${lift >= 0 ? "+" : ""}${lift.toFixed(1)} pts</span></small>
          <small>n = ${m.n}</small></div>`;
      }).join("") + "</div>";
      html += `<div class="acc-note">Trained on the first 80% of every league by date and scored on the last 20%,
        so no scored match was seen during training. The cross-validation figure in the training log is higher
        because it mixes past and future matches.</div>`;
    } else {
      html += `<div class="acc-empty">No backtest yet — it runs as the last step of Train models.</div>`;
    }
    html += "</div>";

    // ---- graded archives ----
    const graded = gd && gd.totals ? (gd.totals["1x2"].graded || 0) : 0;
    html += `<div class="acc-block">
      <div class="acc-head"><h3>Graded predictions</h3>
        <span>${archives.length} archived run(s) · ${graded} match(es) graded</span></div>`;

    if (graded) {
      html += '<div class="acc-grid">' + ["1x2", "ou25", "ou15", "ou35", "btts", "value"].map(k => {
        const t = gd.totals[k];
        if (!t || !t.graded) return "";
        return `<div class="acc-card">
          <div class="k">${k === "value" ? "Value bets" : MARKET_LABEL[k]}</div>
          <b>${(t.accuracy * 100).toFixed(1)}%</b>
          <small>${t.correct} of ${t.graded} correct</small></div>`;
      }).join("") + "</div>";

      html += `<table style="margin-top:16px"><thead><tr>
        <th>Match</th><th>Score</th><th class="num">1X2</th><th class="num">O/U 2.5</th>
        <th class="num">BTTS</th></tr></thead><tbody>` + gd.recent.slice(0, 25).map(r => `
        <tr>
          <td><div class="tm"><b>${esc(r.home)}</b><span>${esc(r.away)}</span>
            <div class="lg">${esc(r.league)}</div></div></td>
          <td class="mk">${esc(r.score)}</td>
          <td class="num">${r.hits["1x2"] ? '<span class="hit">✓</span>' : '<span class="miss">✗</span>'}</td>
          <td class="num">${r.hits["ou25"] ? '<span class="hit">✓</span>' : '<span class="miss">✗</span>'}</td>
          <td class="num">${r.hits["btts"] ? '<span class="hit">✓</span>' : '<span class="miss">✗</span>'}</td>
        </tr>`).join("") + "</tbody></table>";
    } else {
      const waiting = archives.length ? archives[0].matches : 0;
      html += `<div class="acc-empty">Nothing graded yet.
        ${waiting ? `<b>${waiting}</b> predictions are archived and waiting for kick-off.` : "Run Predict &amp; compare to archive a run."}
        Once the matches finish, grade them here — or just press the button.</div>
        <button class="btn" id="btnGrade" style="margin-top:10px">Grade results now</button>`;
    }

    if (archives.length) {
      html += `<div class="acc-head" style="margin-top:18px"><h3 style="font-size:13px">Archive</h3></div>
        <div style="font-size:12px;color:var(--muted);line-height:1.9">` +
        archives.slice(0, 6).map(a =>
          `<div>${esc(a.generated)} · ${a.matches} matches · ${a.with_odds} with odds · ${a.value_bets} edges</div>`
        ).join("") + "</div>";
    }
    html += "</div></div>";
    return html;
  }

  function valueTick(v) {
    const st = liveStates[v.match];
    if (!st) return "";
    const picks = ((st.markets || {}).value || {}).picks || [];
    const hit = picks.find(p => p.market === v.market && p.outcome === v.outcome);
    if (!hit || !hit.state || hit.state === "pending") return "";
    return `<span class="tick ${hit.state}">${TICK[hit.state]}</span>`;
  }

  function edgeCell(e) {
    if (e === undefined || e === null) return `<span class="dash">—</span>`;
    const cls = e >= 0.25 ? "hot" : e > 0.02 ? "pos" : "neg";
    return `<span class="edge ${cls}">${e > 0 ? "+" : ""}${(e * 100).toFixed(1)}%</span>`;
  }

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* tabs + navbar */
  const TABS = ["1x2", "goals", "btts", "value", "live", "accuracy"];

  function selectTab(name, scroll) {
    activeTab = name;
    history.replaceState(null, "", `#${name}`);
    $$(".tab").forEach(x => x.classList.toggle("active", x.dataset.tab === name));
    const navKey = name === "accuracy" ? "accuracy" : name === "live" ? "live" : "home";
    $$(".navlink").forEach(x => x.classList.toggle("active", x.dataset.nav === navKey));
    if (name === "accuracy") loadAccuracy();
    if (name === "live") { loadLive(false); startLivePolling(); }
    if (results || name === "live") renderTable();
    if (scroll === "results") {
      const y = $("#results").getBoundingClientRect().top + window.scrollY - 78;
      window.scrollTo({ top: Math.max(y, 0), behavior: "smooth" });
    } else if (scroll === "top") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  $$(".tab").forEach(t => t.addEventListener("click", () => selectTab(t.dataset.tab, false)));

  $$(".navlink").forEach(n => n.addEventListener("click", () => {
    const go = n.dataset.nav;
    if (go === "accuracy") selectTab("accuracy", "results");
    else if (go === "live") selectTab("live", "results");
    else selectTab("1x2", "top");
  }));

  const hashTab = location.hash.replace("#", "");
  if (TABS.includes(hashTab)) selectTab(hashTab, false);

  /* sort */
  $$(".chip.sort").forEach(c => c.addEventListener("click", () => {
    sortMode = c.dataset.sort;
    $$(".chip.sort").forEach(x => x.classList.toggle("active", x === c));
    if (results) renderTable();
  }));

  /* ------------------------------------------------------------------ */
  pollState();
  setInterval(pollState, 1200);
  setInterval(() => { if (state.status === "running") pollLogs(); }, 1200);
  loadResults();
  loadAccuracy();
  loadLive(false);      // ticks are useful on every tab, not just Live
  startLivePolling();
})();
