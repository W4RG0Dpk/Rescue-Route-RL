/* RescueRoute Mission Control - isometric mission replay, comparison and results viewer. */
(() => {
  "use strict";

  const DATA = window.RR_DATA || { traces: [], benchmark: null, weather: null, methods: {}, profiles: [] };
  const NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "HOVER"];
  const DIRS = [[0, -1], [0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1], [1, 1], [-1, 1], [0, 0]];
  const B36 = "0123456789abcdefghijklmnopqrstuvwxyz";
  const SHORT = {
    random: "Random",
    bc_cnn_lstm: "BC student (no PPO)", ppo_mlp: "PPO (MLP)", ppo_cnn: "PPO + CNN", ppo_cnn_lstm: "PPO + CNN + LSTM", ppo_cnn_lstm_risk: "PPO + CNN + LSTM + Risk",
    "ppo_cnn_lstm+immediate": "PPO-LSTM + Immediate safety", "ppo_cnn_lstm+deterministic": "PPO-LSTM + Deterministic recoverability",
    "ppo_cnn_lstm+probabilistic": "PPO-LSTM + Probabilistic recoverability (full)",
  };
  const REASON = {
    SUCCESSFUL_DELIVERY: ["Delivered on time", "good"], RETURNED_TO_BASE: ["Mission aborted · UAV returned to base", "info"],
    SAFE_LANDING: ["Mission aborted · safe rooftop landing", "info"], FLOOD_ENGULFMENT: ["UAV lost · engulfed by flood", "bad"],
    FLOOD_HAZARD: ["UAV lost · flew into flood", "bad"], DIAGONAL_FLOOD_HAZARD: ["UAV lost · clipped flooded corner", "bad"],
    DEADLINE_EXCEEDED: ["Mission failed · deadline exceeded", "bad"], PAYLOAD_FAILURE: ["Mission failed · payload damaged", "bad"],
    BATTERY_DEPLETED: ["UAV lost · battery depleted", "bad"], STATIC_BLOCKED: ["UAV lost · building collision", "bad"],
    NFZ_VIOLATION: ["UAV lost · no-fly-zone violation", "bad"], BOUNDARY_CRASH: ["UAV lost · left the map", "bad"],
    DIAGONAL_CORNER_COLLISION: ["UAV lost · corner collision", "bad"],
  };
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => t * t * (3 - 2 * t);
  const hash = (i, j, s = 0) => { const x = Math.sin(i * 127.1 + j * 311.7 + s * 74.7) * 43758.5453; return x - Math.floor(x); };
  const pct = (x) => (x == null || isNaN(x) ? "–" : (100 * x).toFixed(1) + "%");
  const short = (m) => SHORT[m] || m;

  function riskColor(p, delta, alpha = 1) {
    if (p == null) return `rgba(58,69,96,${alpha})`;
    if (p <= delta) return `rgba(46,230,166,${alpha})`;
    if (p < 0.5) return `rgba(255,181,71,${alpha})`;
    return `rgba(255,77,109,${alpha})`;
  }

  // ======================================================================
  // Isometric renderer
  // ======================================================================
  class IsoRenderer {
    constructor(canvas, opts = {}) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.opts = Object.assign({ hud: true }, opts);
      this.layers = { forecast: true, fan: true, trail: true, weather: true };
      this.rain = Array.from({ length: 700 }, () => ({ x: Math.random(), y: Math.random(), z: 0.4 + Math.random() * 0.6 }));
      this.wind = Array.from({ length: 140 }, () => ({ x: Math.random(), y: Math.random(), life: Math.random() }));
      this.flash = 0;
      this.trace = null;
      new ResizeObserver(() => this.resize()).observe(canvas);
    }

    setTrace(trace) {
      this.trace = trace;
      this.W = trace.width;
      this.H = trace.height;
      this.cells = Array.from(trace.cells, Number);
      this.onset = trace.flood_onset;
      this.forecastCache = new Map();
      this.resize();
    }

    resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const r = this.c.getBoundingClientRect();
      if (r.width < 10 || r.height < 10) return;
      this.c.width = Math.round(r.width * dpr);
      this.c.height = Math.round(r.height * dpr);
      this.dpr = dpr;
      this.vw = r.width;
      this.vh = r.height;
      if (!this.trace) return;
      const n = this.W + this.H;
      const tw = Math.min((this.vw * 0.94) / (n / 2), ((this.vh - 70) * 0.9) / (n / 4));
      this.tw = tw;
      this.th = tw / 2;
      this.ox = this.vw / 2 + ((this.H - this.W) * this.tw) / 4;
      this.oy = (this.vh - (n * this.th) / 2) / 2 + 34;
      this.buildStatic();
    }

    iso(gx, gy, z = 0) {
      return [this.ox + ((gx - gy) * this.tw) / 2, this.oy + ((gx + gy) * this.th) / 2 - z];
    }

    diamond(ctx, i, j, z = 0, inset = 0) {
      const a = this.iso(i + inset, j + inset, z), b = this.iso(i + 1 - inset, j + inset, z);
      const c = this.iso(i + 1 - inset, j + 1 - inset, z), d = this.iso(i + inset, j + 1 - inset, z);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.closePath();
    }

    heightOf(i, j) {
      const code = this.cells[j * this.W + i];
      if (code === 4) return this.th * (2.0 + 1.9 * hash(i, j));
      if (code === 2) return this.th * (0.55 + 0.45 * hash(i, j, 3));
      return 0;
    }

    buildStatic() {
      const mk = () => { const cv = document.createElement("canvas"); cv.width = this.c.width; cv.height = this.c.height; const x = cv.getContext("2d"); x.scale(this.dpr, this.dpr); return [cv, x]; };
      const [g, gx] = mk();
      const [b, bx] = mk();
      this.ground = g;
      this.buildings = b;
      // ground slab edge
      const W = this.W, H = this.H;
      const p0 = this.iso(0, H), p1 = this.iso(W, H), p2 = this.iso(W, 0);
      gx.fillStyle = "#0a1324";
      gx.beginPath(); gx.moveTo(p0[0], p0[1]); gx.lineTo(p1[0], p1[1]); gx.lineTo(p2[0], p2[1]);
      gx.lineTo(p2[0], p2[1] + this.th * 0.9); gx.lineTo(p1[0], p1[1] + this.th * 0.9); gx.lineTo(p0[0], p0[1] + this.th * 0.9); gx.closePath(); gx.fill();
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
        const code = this.cells[j * W + i];
        const v = hash(i, j, 9) * 0.05;
        let fill = `rgb(${22 + v * 120},${32 + v * 140},${52 + v * 160})`;
        if (code === 1) fill = "#2a1420";
        if (code === 3) fill = "#0d2a3a";
        this.diamond(gx, i, j); gx.fillStyle = fill; gx.fill();
        gx.strokeStyle = "rgba(120,160,230,0.06)"; gx.lineWidth = 1; gx.stroke();
        if (code === 1) { // NFZ hatch
          gx.save(); this.diamond(gx, i, j); gx.clip();
          gx.strokeStyle = "rgba(255,77,109,0.35)"; gx.lineWidth = 1.2;
          const c = this.iso(i + 0.5, j + 0.5);
          for (let k = -3; k <= 3; k++) { gx.beginPath(); gx.moveTo(c[0] - this.tw + k * 6, c[1] - this.th); gx.lineTo(c[0] + k * 6 + this.tw, c[1] + this.th); gx.stroke(); }
          gx.restore();
        }
        if (code === 3) { // helipad
          this.diamond(gx, i, j, 0, 0.12); gx.strokeStyle = "rgba(76,201,255,0.85)"; gx.lineWidth = 1.4; gx.stroke();
          const c = this.iso(i + 0.5, j + 0.5);
          gx.fillStyle = "rgba(140,225,255,0.9)"; gx.font = `700 ${Math.max(7, this.th * 0.62)}px Inter`; gx.textAlign = "center"; gx.textBaseline = "middle";
          gx.save(); gx.translate(c[0], c[1]); gx.scale(1, 0.55); gx.fillText("H", 0, 0); gx.restore();
        }
      }
      // buildings + NFZ fields, painter's order
      const order = [];
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) order.push([i + j, i, j]);
      order.sort((a, b2) => a[0] - b2[0]);
      this.towers = [];
      for (const [, i, j] of order) {
        const code = this.cells[j * W + i];
        if (code === 4 || code === 2) this.drawBuilding(bx, i, j, code);
        if (code === 1) this.drawField(bx, i, j);
      }
    }

    drawBuilding(ctx, i, j, code) {
      const h = this.heightOf(i, j);
      const A = this.iso(i, j + 1), B = this.iso(i + 1, j + 1), C = this.iso(i + 1, j);
      const tower = code === 4;
      const base = tower ? [38, 46, 70] : [30, 64, 66];
      const shade = (k) => `rgb(${base[0] * k | 0},${base[1] * k | 0},${base[2] * k | 0})`;
      // left (south) face
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.lineTo(B[0], B[1] - h); ctx.lineTo(A[0], A[1] - h); ctx.closePath();
      ctx.fillStyle = shade(0.95); ctx.fill();
      // right (east) face
      ctx.beginPath(); ctx.moveTo(B[0], B[1]); ctx.lineTo(C[0], C[1]); ctx.lineTo(C[0], C[1] - h); ctx.lineTo(B[0], B[1] - h); ctx.closePath();
      ctx.fillStyle = shade(0.72); ctx.fill();
      // top
      this.diamond(ctx, i, j, h); ctx.fillStyle = shade(1.35); ctx.fill();
      ctx.strokeStyle = tower ? "rgba(150,170,230,0.18)" : "rgba(80,220,190,0.35)"; ctx.lineWidth = 1; ctx.stroke();
      if (!tower) { // rooftop landing ring
        this.diamond(ctx, i, j, h, 0.28); ctx.strokeStyle = "rgba(46,230,166,0.45)"; ctx.stroke();
      }
      // windows
      const rows = Math.max(1, Math.floor(h / (this.th * 0.45)));
      for (const [P, Q, k] of [[A, B, 0], [B, C, 1]]) {
        for (let r = 0; r < rows; r++) for (let c = 0; c < 3; c++) {
          const u = (c + 0.5) / 3, v = (r + 0.6) / (rows + 0.4);
          const x = lerp(P[0], Q[0], u), y = lerp(P[1], Q[1], u) - v * h;
          const lit = hash(i * 3 + c, j * 7 + r, k) ;
          if (lit < 0.42) continue;
          ctx.fillStyle = lit > 0.86 ? "rgba(255,214,140,0.9)" : lit > 0.7 ? "rgba(160,200,255,0.55)" : "rgba(120,150,210,0.22)";
          ctx.fillRect(x - this.tw * 0.035, y - this.th * 0.1, this.tw * 0.07, this.th * 0.17);
        }
      }
      if (tower) this.towers.push([i, j, h]);
    }

    drawField(ctx, i, j) {
      const h = this.th * 2.6;
      const A = this.iso(i, j + 1), B = this.iso(i + 1, j + 1), C = this.iso(i + 1, j);
      const g = ctx.createLinearGradient(0, B[1], 0, B[1] - h);
      g.addColorStop(0, "rgba(255,60,90,0.22)"); g.addColorStop(1, "rgba(255,60,90,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.lineTo(C[0], C[1]); ctx.lineTo(C[0], C[1] - h); ctx.lineTo(B[0], B[1] - h); ctx.lineTo(A[0], A[1] - h); ctx.closePath(); ctx.fill();
    }

    forecastAt(k) {
      if (!this.forecastCache.has(k)) {
        const s = this.trace.steps[k].forecast;
        const arr = new Float32Array(s.length);
        for (let n = 0; n < s.length; n++) arr[n] = B36.indexOf(s[n]) / 35;
        this.forecastCache.set(k, arr);
      }
      return this.forecastCache.get(k);
    }

    // pos at fractional time t (steps index space)
    dronePos(t) {
      const st = this.trace.steps;
      const k = clamp(Math.floor(t), 0, st.length - 1);
      const a = st[k].pos;
      const b = k + 1 < st.length ? st[k + 1].pos : this.trace.final_pos;
      const f = ease(clamp(t - k, 0, 1));
      return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), k, f];
    }

    render(t, clock) {
      const ctx = this.ctx, tr = this.trace;
      if (!tr || !this.ground) return;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      // sky
      const sky = ctx.createLinearGradient(0, 0, 0, this.vh);
      sky.addColorStop(0, "#081022"); sky.addColorStop(1, "#04070f");
      ctx.fillStyle = sky; ctx.fillRect(0, 0, this.vw, this.vh);
      ctx.drawImage(this.ground, 0, 0, this.vw, this.vh);

      const st = tr.steps;
      const [px, py, k, f] = this.dronePos(t);
      const step = st[k];
      const simStep = step.t + f; // flood state time
      const W = this.W;
      const delta = tr.delta ?? 0.15;

      // forecast overlay
      if (this.layers.forecast) {
        const fc = this.forecastAt(k);
        for (let n = 0; n < fc.length; n++) {
          const p = fc[n];
          if (p < 0.05) continue;
          const on = this.onset[n];
          if (on >= 0 && on <= simStep) continue;
          const i = n % W, j = (n / W) | 0;
          this.diamond(ctx, i, j, 0, 0.04);
          ctx.fillStyle = `rgba(255,${(170 - 120 * p) | 0},60,${0.08 + 0.45 * p})`;
          ctx.fill();
        }
      }

      // water
      for (let n = 0; n < this.onset.length; n++) {
        const on = this.onset[n];
        if (on < 0 || on > Math.ceil(simStep)) continue;
        let a = 1;
        if (on > simStep - 1 && on > 0) a = clamp(simStep - (on - 1), 0, 1);
        if (a <= 0) continue;
        const i = n % W, j = (n / W) | 0;
        const age = clamp((simStep - on) / 25, 0, 1);
        const sh = 0.5 + 0.5 * Math.sin(clock * 2.2 + i * 0.9 + j * 1.3);
        this.diamond(ctx, i, j, 0, 0.0);
        ctx.fillStyle = `rgba(${(30 - 14 * age) | 0},${(120 + 30 * sh - 40 * age) | 0},${(235 - 30 * age) | 0},${0.78 * a})`;
        ctx.fill();
        if (sh > 0.82) { this.diamond(ctx, i, j, 0, 0.3); ctx.fillStyle = `rgba(190,230,255,${0.18 * a})`; ctx.fill(); }
        if (on > simStep - 1.2 && on > 0) { this.diamond(ctx, i, j, 0, 0); ctx.strokeStyle = `rgba(160,220,255,${0.8 * a})`; ctx.lineWidth = 1.5; ctx.stroke(); }
      }

      // shield fan on neighbour tiles
      if (this.layers.fan && f < 0.55) {
        const cx = step.pos[0], cy = step.pos[1];
        for (let a = 0; a < 9; a++) {
          const [dx, dy] = DIRS[a];
          const i = cx + dx, j = cy + dy;
          if (i < 0 || j < 0 || i >= W || j >= this.H) continue;
          const imm = step.immediate_ok ? step.immediate_ok[a] : true;
          let col;
          if (!imm) col = "rgba(90,100,130,0.35)";
          else if (step.p_unrec) col = riskColor(step.p_unrec[a], delta, 0.42);
          else col = `rgba(76,201,255,${0.08 + 0.5 * step.probs[a]})`;
          this.diamond(ctx, i, j, 0, 0.08); ctx.fillStyle = col; ctx.fill();
          if (a === step.action) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
          else if (a === step.policy_action && step.policy_action !== step.action) { ctx.setLineDash([4, 3]); ctx.strokeStyle = "#ffb547"; ctx.lineWidth = 1.6; ctx.stroke(); ctx.setLineDash([]); }
        }
      }

      // buildings
      ctx.drawImage(this.buildings, 0, 0, this.vw, this.vh);
      // aviation lights
      const blink = 0.5 + 0.5 * Math.sin(clock * 3);
      for (const [i, j, h] of this.towers) {
        if (hash(i, j, 5) < 0.55) continue;
        const p = this.iso(i + 0.5, j + 0.5, h);
        ctx.fillStyle = `rgba(255,70,90,${0.35 + 0.65 * blink * hash(i, j, 6)})`;
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.6, 0, 6.29); ctx.fill();
      }

      // start + target markers
      const [sx, sy] = tr.start, [tx, ty] = tr.target;
      const sp = this.iso(sx + 0.5, sy + 0.5);
      ctx.strokeStyle = `rgba(76,201,255,${0.5 + 0.4 * blink})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(sp[0], sp[1], this.tw * 0.55, this.th * 0.55, 0, 0, 6.29); ctx.stroke();
      const tp = this.iso(tx + 0.5, ty + 0.5);
      const beam = ctx.createLinearGradient(0, tp[1], 0, tp[1] - this.th * 9);
      beam.addColorStop(0, "rgba(255,80,110,0.55)"); beam.addColorStop(1, "rgba(255,80,110,0)");
      ctx.fillStyle = beam; ctx.fillRect(tp[0] - this.tw * 0.16, tp[1] - this.th * 9, this.tw * 0.32, this.th * 9);
      for (let r = 0; r < 3; r++) {
        const ph = (clock * 0.7 + r / 3) % 1;
        ctx.strokeStyle = `rgba(255,90,120,${0.7 * (1 - ph)})`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.ellipse(tp[0], tp[1], this.tw * (0.3 + 1.1 * ph), this.th * (0.3 + 1.1 * ph), 0, 0, 6.29); ctx.stroke();
      }
      const cz = this.iso(tx + 0.5, ty + 0.5, this.th * 4.2 + Math.sin(clock * 2) * 2);
      const s = this.tw * 0.28;
      ctx.fillStyle = "#fff"; ctx.shadowColor = "rgba(255,80,110,0.9)"; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(cz[0], cz[1], s * 1.15, 0, 6.29); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = "#ff3d64";
      ctx.fillRect(cz[0] - s * 0.22, cz[1] - s * 0.75, s * 0.44, s * 1.5); ctx.fillRect(cz[0] - s * 0.75, cz[1] - s * 0.22, s * 1.5, s * 0.44);

      // trail
      const alt = this.th * 3.4;
      if (this.layers.trail) {
        const pts = [];
        for (let n = 0; n <= k; n++) pts.push(st[n].pos);
        pts.push([px, py]);
        ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.shadowColor = "rgba(76,201,255,0.8)"; ctx.shadowBlur = 10;
        for (let n = 1; n < pts.length; n++) {
          const a = this.iso(pts[n - 1][0] + 0.5, pts[n - 1][1] + 0.5, alt), b = this.iso(pts[n][0] + 0.5, pts[n][1] + 0.5, alt);
          const src = st[Math.min(n - 1, k)].source;
          const fb = src && src.startsWith("fallback");
          const sh = src === "shield" || src === "shield_min_risk";
          ctx.strokeStyle = fb ? `rgba(160,130,255,${0.25 + 0.75 * n / pts.length})` : sh ? `rgba(255,181,71,${0.3 + 0.7 * n / pts.length})` : `rgba(76,201,255,${0.2 + 0.8 * n / pts.length})`;
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
        ctx.shadowBlur = 0;
      }

      // drone
      this.drawDrone(px, py, alt, clock, step, f);

      // weather particles
      if (this.layers.weather) this.drawWeather(step, clock);

      if (this.flash > 0) { ctx.fillStyle = `rgba(200,220,255,${this.flash * 0.25})`; ctx.fillRect(0, 0, this.vw, this.vh); this.flash *= 0.86; if (this.flash < 0.02) this.flash = 0; }
      // vignette
      const vg = ctx.createRadialGradient(this.vw / 2, this.vh / 2, this.vh * 0.35, this.vw / 2, this.vh / 2, this.vh * 0.95);
      vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.5)");
      ctx.fillStyle = vg; ctx.fillRect(0, 0, this.vw, this.vh);
    }

    drawDrone(px, py, alt, clock, step, f) {
      const ctx = this.ctx;
      const bob = Math.sin(clock * 3.1) * 1.6;
      const g = this.iso(px + 0.5, py + 0.5);
      const p = this.iso(px + 0.5, py + 0.5, alt + bob);
      // shadow
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.beginPath(); ctx.ellipse(g[0], g[1], this.tw * 0.32, this.th * 0.32, 0, 0, 6.29); ctx.fill();
      ctx.strokeStyle = "rgba(76,201,255,0.18)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(g[0], g[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
      const sc = this.tw * 0.62;
      // halo
      const halo = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], sc * 1.6);
      halo.addColorStop(0, "rgba(76,201,255,0.35)"); halo.addColorStop(1, "rgba(76,201,255,0)");
      ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(p[0], p[1], sc * 1.6, 0, 6.29); ctx.fill();
      const act = step.action >= 0 ? DIRS[step.action] : [0, 0];
      const tilt = (act[0] - act[1]) * 0.12 * Math.sin(Math.PI * f);
      // payload
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[0], p[1] + sc * 0.75); ctx.stroke();
      const stab = step.stability;
      ctx.fillStyle = stab < (this.trace.required_stability || 0.4) ? "#ff4d6d" : "#ffb547";
      ctx.fillRect(p[0] - sc * 0.18, p[1] + sc * 0.65, sc * 0.36, sc * 0.3);
      ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(tilt);
      const arms = [[1, 0], [0, 1], [-1, 0], [0, -1]].map(([a, b]) => [(a - b) * sc * 0.62, (a + b) * sc * 0.31]);
      ctx.strokeStyle = "#1c2740"; ctx.lineWidth = 3.2;
      for (const [x, y] of arms) { ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(x, y); ctx.stroke(); }
      ctx.strokeStyle = "#cfe6ff"; ctx.lineWidth = 1.2;
      for (const [x, y] of arms) { ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(x, y); ctx.stroke(); }
      arms.forEach(([x, y], n) => {
        ctx.fillStyle = "rgba(200,230,255,0.16)";
        ctx.beginPath(); ctx.ellipse(x, y, sc * 0.36, sc * 0.17, 0, 0, 6.29); ctx.fill();
        const ang = clock * 40 + n;
        ctx.strokeStyle = "rgba(230,245,255,0.6)"; ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(x - Math.cos(ang) * sc * 0.34, y - Math.sin(ang) * sc * 0.16); ctx.lineTo(x + Math.cos(ang) * sc * 0.34, y + Math.sin(ang) * sc * 0.16); ctx.stroke();
        ctx.fillStyle = n < 2 ? "#2ee6a6" : "#ff4d6d";
        ctx.beginPath(); ctx.arc(x, y, 1.3, 0, 6.29); ctx.fill();
      });
      const body = ctx.createRadialGradient(-2, -2, 1, 0, 0, sc * 0.3);
      body.addColorStop(0, "#ffffff"); body.addColorStop(1, "#4cc9ff");
      ctx.fillStyle = body; ctx.shadowColor = "rgba(76,201,255,0.9)"; ctx.shadowBlur = 16;
      ctx.beginPath(); ctx.ellipse(0, 0, sc * 0.26, sc * 0.16, 0, 0, 6.29); ctx.fill();
      ctx.shadowBlur = 0; ctx.restore();
      if (this.opts.hud) { // status tag
        const label = `UAV · ${Math.round(100 * step.battery / this.trace.battery0)}%`;
        ctx.font = "700 11px Inter"; const w = ctx.measureText(label).width + 14;
        const lx = p[0] - w / 2, ly = p[1] - sc * 1.25 - 18;
        ctx.fillStyle = "rgba(8,14,28,0.85)"; ctx.strokeStyle = "rgba(76,201,255,0.6)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(lx, ly, w, 18, 9); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#cdeeff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(label, p[0], ly + 9.5);
      }
    }

    drawWeather(step, clock) {
      const ctx = this.ctx;
      const rad = (step.wind_dir * Math.PI) / 180;
      const wx = Math.sin(rad), wy = -Math.cos(rad); // grid (east, south)
      const sx = ((wx - wy) * this.tw) / 2, sy = ((wx + wy) * this.th) / 2;
      const nrm = Math.hypot(sx, sy) || 1;
      const ux = sx / nrm, uy = sy / nrm;
      const speed = step.wind_speed + step.gust;
      // wind streaks
      ctx.lineCap = "round";
      for (const p of this.wind) {
        p.x += ux * speed * 0.00018 * 6; p.y += uy * speed * 0.00018 * 6; p.life -= 0.006;
        if (p.life <= 0 || p.x < -0.05 || p.x > 1.05 || p.y < -0.05 || p.y > 1.05) { p.x = Math.random(); p.y = Math.random(); p.life = 1; }
        const x = p.x * this.vw, y = p.y * this.vh, L = 6 + speed * 1.6;
        ctx.strokeStyle = `rgba(200,225,255,${0.16 * Math.sin(Math.PI * p.life)})`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - ux * L, y - uy * L); ctx.stroke();
      }
      // rain
      const n = Math.floor(this.rain.length * clamp(step.rain * 1.4, 0.05, 1));
      const slant = clamp(ux * speed * 0.06, -0.6, 0.6);
      ctx.strokeStyle = "rgba(170,200,255,0.32)"; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let q = 0; q < n; q++) {
        const r = this.rain[q];
        r.y += 0.018 * r.z; r.x += slant * 0.006 * r.z;
        if (r.y > 1) { r.y = -0.05; r.x = Math.random(); }
        if (r.x > 1) r.x -= 1; if (r.x < 0) r.x += 1;
        const x = r.x * this.vw, y = r.y * this.vh, L = 10 * r.z;
        ctx.moveTo(x, y); ctx.lineTo(x + slant * L, y + L);
      }
      ctx.stroke();
      if (step.rain > 0.55 && Math.random() < 0.0035 * step.rain) this.flash = 1;
    }
  }

  // ======================================================================
  // App state
  // ======================================================================
  const state = { trace: null, t: 0, playing: false, speed: 1, lastStep: -1, live: false };
  const scene = new IsoRenderer($("scene"));

  function setTrace(tr) {
    state.trace = tr;
    state.t = 0;
    state.lastStep = -1;
    scene.setTrace(tr);
    $("scrub").max = String(tr.steps.length);
    $("endBanner").className = "end-banner";
    $("toasts").innerHTML = "";
    renderMissionCard(tr);
    drawTimeline();
    updateHud(0, true);
  }

  function renderMissionCard(tr) {
    const fin = tr.final;
    const [label, kind] = REASON[fin.reason] || [fin.reason, "info"];
    $("missionCard").innerHTML = `
      <div class="mc-title">${short(tr.method)}</div>
      <div class="mc-sub">Mission #${tr.seed} · profile <b>${tr.profile.replace(/_/g, " ")}</b></div>
      <div class="chips">
        <span class="chip ${tr.payload === "FRAGILE" ? "hot" : ""}">${tr.payload} payload ≥ ${tr.required_stability.toFixed(2)}</span>
        <span class="chip">deadline ${tr.deadline} steps</span>
        <span class="chip">battery ${tr.battery0.toFixed(0)}</span>
        <span class="chip acc">forecast α ${tr.alpha_fc.toFixed(2)}</span>
        <span class="chip ${fin.oracle_feasible ? "ok" : "hot"}">${fin.oracle_feasible ? "oracle-feasible" : "infeasible even for an oracle"}</span>
      </div>`;
    $("shieldMode").textContent = tr.method.includes("+") ? tr.method.split("+")[1] + (tr.delta != null ? ` · δ=${tr.delta}` : "") : "no shield";
  }

  function toast(msg, kind = "info") {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    $("toasts").appendChild(el);
    while ($("toasts").children.length > 3) $("toasts").firstChild.remove();
    setTimeout(() => el.remove(), 3600);
  }

  function gauge(id, frac, text, color, mark) {
    const g = $(id);
    g.querySelector(".g-fill").style.width = `${clamp(frac, 0, 1) * 100}%`;
    g.querySelector(".g-fill").style.background = color;
    g.querySelector(".g-val").textContent = text;
    const m = g.querySelector(".g-mark");
    if (m && mark != null) m.style.left = `calc(${mark * 100}% - 1px)`;
  }

  function updateHud(k, force) {
    const tr = state.trace;
    if (!tr) return;
    const st = tr.steps[clamp(k, 0, tr.steps.length - 1)];
    const free = Array.from(tr.cells).filter((c) => c !== "1" && c !== "4").length;
    const bf = st.battery / tr.battery0;
    gauge("gBattery", bf, `${st.battery.toFixed(1)} / ${tr.battery0.toFixed(0)}`, bf < 0.2 ? "var(--danger)" : bf < 0.4 ? "var(--warn)" : "linear-gradient(90deg,#2ee6a6,#4cc9ff)");
    const rem = tr.deadline - st.t;
    gauge("gDeadline", rem / tr.deadline, `${rem} steps left`, rem / tr.deadline < 0.2 ? "var(--danger)" : "linear-gradient(90deg,#7c5cff,#4cc9ff)");
    const ok = st.stability >= tr.required_stability;
    gauge("gPayload", st.stability, st.stability.toFixed(3), ok ? "linear-gradient(90deg,#2ee6a6,#9bffd9)" : "linear-gradient(90deg,#ff4d6d,#ffb547)", tr.required_stability);
    gauge("gFlood", st.flooded / free, `${((100 * st.flooded) / free).toFixed(1)}%`, "linear-gradient(90deg,#1f6fff,#4cc9ff)");
    $("wxWind").textContent = `${st.wind_speed.toFixed(1)} m/s → ${st.wind_dir.toFixed(0)}°`;
    $("wxGust").textContent = `+${st.gust.toFixed(1)} m/s`;
    $("wxRain").textContent = `${(st.rain * 100).toFixed(0)}% intensity`;
    $("wxAlpha").textContent = tr.alpha_fc.toFixed(2);
    drawCompass(st);
    // decision
    const delta = tr.delta ?? 0.15;
    const src = st.source;
    const actName = st.action < 0 ? "LAND" : NAMES[st.action];
    let txt = `<span class="src ${src}">${src.replace(/_/g, " ")}</span>`;
    if (st.action < 0) txt += `<b>Landing</b> on a dry landing cell.`;
    else if (st.policy_action !== st.action) txt += `Policy wanted <b>${NAMES[st.policy_action]}</b>${st.p_unrec && st.p_unrec[st.policy_action] != null ? ` (P<sub>unrec</sub> ${st.p_unrec[st.policy_action].toFixed(2)})` : " (immediately unsafe)"}; executing <b>${actName}</b>${st.p_unrec && st.p_unrec[st.action] != null ? ` (P<sub>unrec</sub> ${st.p_unrec[st.action].toFixed(2)})` : ""}.`;
    else txt += `Executing policy action <b>${actName}</b>${st.p_unrec && st.p_unrec[st.action] != null ? ` · P<sub>unrec</sub> ${st.p_unrec[st.action].toFixed(2)}` : ""}.`;
    if (st.note) txt += `<br><span class="muted">${st.note}</span>`;
    $("decision").innerHTML = txt;
    const order = [5, 0, 4, 3, 8, 2, 7, 1, 6]; // compass layout NW N NE / W H E / SW S SE
    $("actionGrid").innerHTML = order.map((a) => {
      const imm = st.immediate_ok ? st.immediate_ok[a] : true;
      const p = st.p_unrec ? st.p_unrec[a] : null;
      const col = !imm ? "var(--muted)" : p != null ? riskColor(p, delta) : "var(--accent)";
      const cls = ["act", a === st.action ? "chosen" : "", a === st.policy_action && st.policy_action !== st.action ? "wanted" : ""].join(" ");
      return `<div class="${cls}"><div class="nm">${a === 8 ? "◎" : NAMES[a]}</div>
        <div class="pr">π ${(st.probs[a] * 100).toFixed(0)}%</div>
        <div class="pr">${!imm ? "blocked" : p != null ? "P " + p.toFixed(2) : "—"}</div>
        <div class="risk" style="width:${!imm ? 100 : p != null ? Math.max(8, p * 100) : 100}%;background:${col}"></div></div>`;
    }).join("");
    $("stepInfo").textContent = `t = ${st.t} / ${tr.deadline}`;
  }

  function drawCompass(st) {
    const c = $("compass"), x = c.getContext("2d"), R = 50;
    x.clearRect(0, 0, 120, 120);
    x.save(); x.translate(60, 60);
    x.strokeStyle = "rgba(255,255,255,.12)"; x.lineWidth = 1;
    for (const r of [R, R * 0.66, R * 0.33]) { x.beginPath(); x.arc(0, 0, r, 0, 6.29); x.stroke(); }
    x.fillStyle = "#6f7d9c"; x.font = "600 10px Inter"; x.textAlign = "center"; x.textBaseline = "middle";
    [["N", 0, -R - 6], ["E", R + 6, 0], ["S", 0, R + 6], ["W", -R - 6, 0]].forEach(([s, a, b]) => x.fillText(s, a, b));
    const rad = (st.wind_dir * Math.PI) / 180;
    const L = clamp((st.wind_speed + st.gust) / 16, 0.15, 1) * R;
    const ex = Math.sin(rad) * L, ey = -Math.cos(rad) * L;
    const g = x.createLinearGradient(0, 0, ex, ey); g.addColorStop(0, "rgba(76,201,255,.1)"); g.addColorStop(1, "#4cc9ff");
    x.strokeStyle = g; x.lineWidth = 4; x.lineCap = "round";
    x.beginPath(); x.moveTo(-ex * 0.3, -ey * 0.3); x.lineTo(ex, ey); x.stroke();
    x.fillStyle = "#4cc9ff"; x.beginPath(); x.arc(ex, ey, 4, 0, 6.29); x.fill();
    x.restore();
  }

  function drawTimeline() {
    const tr = state.trace, c = $("timeline");
    const dpr = Math.min(2, window.devicePixelRatio || 1), r = c.getBoundingClientRect();
    c.width = r.width * dpr; c.height = r.height * dpr;
    const x = c.getContext("2d"); x.scale(dpr, dpr);
    const W = r.width, H = r.height, n = tr.steps.length;
    x.fillStyle = "rgba(255,255,255,.035)"; x.fillRect(0, 0, W, H);
    const X = (k) => (k / Math.max(1, n)) * W;
    // risk band
    x.beginPath(); x.moveTo(0, H);
    tr.steps.forEach((s, k) => {
      const v = s.p_unrec ? Math.min(...s.p_unrec.filter((q) => q != null).concat([1])) : 0;
      x.lineTo(X(k), H - v * (H - 6));
    });
    x.lineTo(X(n), H); x.closePath(); x.fillStyle = "rgba(255,77,109,.28)"; x.fill();
    const line = (fn, col) => { x.beginPath(); tr.steps.forEach((s, k) => { const y = H - 3 - fn(s) * (H - 8); k ? x.lineTo(X(k), y) : x.moveTo(X(k), y); }); x.strokeStyle = col; x.lineWidth = 1.6; x.stroke(); };
    line((s) => s.stability, "rgba(255,255,255,.85)");
    line((s) => s.battery / tr.battery0, "#4cc9ff");
    tr.steps.forEach((s, k) => {
      if (s.source === "policy" || s.source == null) return;
      x.fillStyle = s.source.startsWith("fallback") ? "#a48bff" : s.source === "unrecoverable" ? "#ff4d6d" : "#ffb547";
      x.fillRect(X(k), 0, Math.max(2, W / n - 1), 5);
    });
    x.fillStyle = "#6f7d9c"; x.font = "10px Inter";
    x.fillText("— stability   — battery   ▇ min P_unrec   ▮ intervention", 6, H - 4);
    state.tlW = W;
  }

  function drawPlayhead() {
    const c = $("timeline"), x = c.getContext("2d");
    drawTimeline();
    const n = state.trace.steps.length, W = state.tlW;
    const px = (state.t / n) * W;
    x.fillStyle = "#fff"; x.fillRect(px - 1, 0, 2, c.height);
  }

  function onStepChange(k, prevK) {
    const tr = state.trace;
    if (k <= prevK) return;
    const st = tr.steps[k];
    if (!st) return;
    if (st.source === "shield" || st.source === "shield_min_risk") {
      const pu = st.p_unrec;
      const from = pu && pu[st.policy_action] != null ? pu[st.policy_action].toFixed(2) : "blocked";
      const to = pu && pu[st.action] != null ? pu[st.action].toFixed(2) : "–";
      toast(`⚠ Shield override  ${NAMES[st.policy_action]} → ${NAMES[st.action]}   ·   P_unrec ${from} → ${to}`, "warn");
    } else if (st.source.startsWith("fallback") && !(tr.steps[k - 1] || {}).source?.startsWith("fallback")) {
      toast(st.action < 0 ? "🛬 Safe landing" : "↩ Emergency fallback engaged · " + (st.note || ""), "info");
    } else if (st.source === "unrecoverable") {
      toast("✖ No recoverable option left - least-risk action", "bad");
    }
    const prev = tr.steps[k - 1];
    if (prev && prev.stability >= tr.required_stability && st.stability < tr.required_stability) toast("Payload below delivery threshold - needs re-stabilising", "warn");
  }

  function showEnd() {
    const fin = state.trace.final;
    const [label, kind] = REASON[fin.reason] || [fin.reason, "info"];
    const rec = fin.success ? (fin.recovery_feasible ? " · recovery to a landing site feasible" : " · no safe recovery left") : "";
    $("endBanner").innerHTML = `${label}<small>${fin.steps} steps · energy ${fin.energy_used.toFixed(1)} · ${fin.interventions} shield interventions${rec}</small>`;
    $("endBanner").className = `end-banner show ${kind}`;
  }

  // ======================================================================
  // Compare tab
  // ======================================================================
  const cmp = { a: new IsoRenderer($("cmpA")), b: new IsoRenderer($("cmpB")), t: 0, playing: false, ta: null, tb: null };

  function missionKey(tr) { return `${tr.profile}#${tr.seed}`; }

  function setupCompare() {
    const groups = {};
    DATA.traces.forEach((tr, i) => { (groups[missionKey(tr)] ||= []).push(i); });
    const keys = Object.keys(groups).filter((k) => groups[k].length >= 2);
    $("cmpMission").innerHTML = keys.map((k) => `<option value="${k}">${k.replace("#", " · #").replace(/_/g, " ")}</option>`).join("");
    const fill = () => {
      const idx = groups[$("cmpMission").value] || [];
      const opts = idx.map((i) => `<option value="${i}">${short(DATA.traces[i].method)}</option>`).join("");
      $("cmpLeft").innerHTML = opts; $("cmpRight").innerHTML = opts;
      const im = idx.find((i) => DATA.traces[i].method.endsWith("+immediate")) ?? idx[0];
      const pr = idx.find((i) => DATA.traces[i].method.endsWith("+probabilistic")) ?? idx[idx.length - 1];
      $("cmpLeft").value = im; $("cmpRight").value = pr;
      loadCompare();
    };
    $("cmpMission").onchange = fill;
    $("cmpLeft").onchange = $("cmpRight").onchange = loadCompare;
    if (keys.length) fill();
  }

  function loadCompare() {
    cmp.ta = DATA.traces[+$("cmpLeft").value];
    cmp.tb = DATA.traces[+$("cmpRight").value];
    if (!cmp.ta || !cmp.tb) return;
    cmp.a.setTrace(cmp.ta); cmp.b.setTrace(cmp.tb);
    cmp.t = 0;
    $("cmpScrub").max = String(Math.max(cmp.ta.steps.length, cmp.tb.steps.length));
  }

  function cmpLabel(el, tr, t) {
    const k = clamp(Math.floor(t), 0, tr.steps.length - 1);
    const st = tr.steps[k];
    const done = t >= tr.steps.length;
    const [label, kind] = REASON[tr.final.reason] || [tr.final.reason, "info"];
    const col = kind === "good" ? "var(--safe)" : kind === "bad" ? "var(--danger)" : "#c4b5ff";
    el.innerHTML = `<b>${short(tr.method)}</b><br>t=${st.t} · battery ${st.battery.toFixed(0)} · stability ${st.stability.toFixed(2)}` +
      (done ? `<br><b style="color:${col}">${label}</b>` : `<br><span class="muted">${st.source.replace(/_/g, " ")}</span>`);
  }

  // ======================================================================
  // Benchmark + weather pages
  // ======================================================================
  function svgBars(container, groups, series, colors, opts = {}) {
    const W = Math.max(720, container.clientWidth || 900), H = opts.height || 320, padL = 44, padB = 70, padT = 16;
    const gw = (W - padL - 10) / groups.length, bw = (gw * 0.8) / series.length;
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    for (let v = 0; v <= 1.001; v += 0.25) {
      const y = padT + (1 - v) * (H - padT - padB);
      s += `<line x1="${padL}" x2="${W - 10}" y1="${y}" y2="${y}" stroke="rgba(255,255,255,.07)"/><text x="${padL - 8}" y="${y + 4}" font-size="10" text-anchor="end">${(v * 100) | 0}%</text>`;
    }
    groups.forEach((g, gi) => {
      series.forEach((ser, si) => {
        const v = ser.values[gi];
        if (v == null || isNaN(v)) return;
        const h = v * (H - padT - padB), x = padL + gi * gw + gw * 0.1 + si * bw, y = H - padB - h;
        s += `<rect x="${x}" y="${y}" width="${bw - 2}" height="${h}" rx="3" fill="${colors[si]}"><title>${ser.name} · ${g}: ${pct(v)}</title></rect>`;
      });
      const cx = padL + gi * gw + gw / 2;
      s += `<text x="${cx}" y="${H - padB + 16}" font-size="10.5" text-anchor="middle">` +
        g.split("_").map((w, li) => `<tspan x="${cx}" dy="${li ? 12 : 0}">${w}</tspan>`).join("") + `</text>`;
    });
    s += `</svg>`;
    let legend = `<div class="legend-row">` + series.map((ser, i) => `<span><i class="sw" style="background:${colors[i]}"></i>${ser.name}</span>`).join("") + `</div>`;
    container.innerHTML = s + legend;
  }

  function setupBench() {
    const B = DATA.benchmark;
    if (!B) { $("benchLede").textContent = "Run scripts/evaluate.py and scripts/export_demo.py to populate this page."; return; }
    const ov = Object.fromEntries(B.overall.map((r) => [r.method, r]));
    const full = ov["ppo_cnn_lstm+probabilistic"], imm = ov["ppo_cnn_lstm+immediate"], raw = ov["ppo_cnn_lstm"];
    $("benchLede").innerHTML = `${B.n_episodes.toLocaleString()} paired test episodes across ${new Set(B.per_profile.map((r) => r.profile)).size} stress profiles. Every method faced the same missions and the same hidden flood/wind realisations. Shield threshold δ = ${B.delta} (tuned on validation seeds only).`;
    const kpi = (v, l, d) => `<div class="kpi glass"><div class="kv">${v}</div><div class="kl">${l}</div><div class="kd">${d || ""}</div></div>`;
    if (full) {
      $("benchKpis").innerHTML =
        kpi(pct(full.success), "Mission success (full method)", imm ? `immediate filter: ${pct(imm.success)} · raw policy: ${raw ? pct(raw.success) : "–"}` : "") +
        kpi(pct(full.uav_lost), "UAV-loss rate (full method)", imm ? `immediate filter: ${pct(imm.uav_lost)}` : "") +
        kpi(pct(full.success_and_recovery), "Delivered and recoverable", imm ? `immediate filter: ${pct(imm.success_and_recovery)}` : "") +
        kpi(pct(full.safe_abort), "Safe aborts (UAV saved)", "missions abandoned to save the aircraft");
    }
    const profiles = [...new Set(B.per_profile.map((r) => r.profile))];
    const keyM = ["ppo_cnn", "ppo_cnn_lstm", "ppo_cnn_lstm+immediate", "ppo_cnn_lstm+deterministic", "ppo_cnn_lstm+probabilistic"].filter((m) => ov[m]);
    const cols = { "ppo_cnn_lstm+probabilistic": "#2a78d6", "ppo_cnn_lstm+immediate": "#d95926", ppo_cnn_lstm: "#199e70", ppo_cnn: "#c98500", "ppo_cnn_lstm+deterministic": "#d55181" };
    const per = {}; B.per_profile.forEach((r) => (per[r.method + "|" + r.profile] = r));
    svgBars($("benchProfile"), profiles, keyM.map((m) => ({ name: short(m), values: profiles.map((p) => per[m + "|" + p]?.success) })), keyM.map((m) => cols[m]));
    const methods = B.overall.map((r) => r.method);
    const outcomeKeys = [["success", "Delivered", "#199e70"], ["safe_abort", "Safe abort", "#c98500"], ["deadline_failure", "Deadline", "#6f7d9c"], ["payload_failure", "Payload", "#d55181"], ["uav_lost", "UAV lost", "#e66767"]];
    let s = `<svg width="100%" height="${methods.length * 30 + 30}" viewBox="0 0 1000 ${methods.length * 30 + 30}" preserveAspectRatio="none">`;
    methods.forEach((m, i) => {
      let x = 260;
      s += `<text x="250" y="${i * 30 + 20}" font-size="12" text-anchor="end">${short(m)}</text>`;
      outcomeKeys.forEach(([k, l, c]) => { const w = (ov[m][k] || 0) * 720; s += `<rect x="${x}" y="${i * 30 + 6}" width="${Math.max(0, w - 2)}" height="20" rx="3" fill="${c}"><title>${short(m)} · ${l}: ${pct(ov[m][k])}</title></rect>`; x += w; });
    });
    s += `</svg><div class="legend-row">` + outcomeKeys.map(([, l, c]) => `<span><i class="sw" style="background:${c}"></i>${l}</span>`).join("") + `</div>`;
    $("benchOutcomes").innerHTML = s;
    const head = ["Method", "Success", "Success + recovery", "UAV lost", "Flood", "Deadline", "Payload", "Safe abort", "Future failure", "Interventions"];
    const keys = ["success", "success_and_recovery", "uav_lost", "flood_failure", "deadline_failure", "payload_failure", "safe_abort", "future_failure", "intervention_rate"];
    $("benchTable").innerHTML = `<table><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` +
      methods.map((m) => `<tr class="${m.endsWith("probabilistic") ? "hl" : ""}"><td>${short(m)}</td>${keys.map((k) => `<td>${pct(ov[m][k])}</td>`).join("")}</tr>`).join("") + `</table>`;
  }

  function setupWeather() {
    const Wd = DATA.weather;
    if (!Wd) { $("wxLede").textContent = "Run scripts/validate_weather.py and scripts/export_demo.py to populate this page."; return; }
    $("wxLede").innerHTML = `ERA5 hourly reanalysis (Open-Meteo) for <b>${Wd.location}</b>, ${Wd.season} monsoon, fitted on ${Wd.train_years.join(", ")} and tested on the held-out year ${Wd.test_years.join(", ")}. The V1 columns run the simulator's own rain and wind classes, so this checks what the code actually does.`;
    const kpi = (v, l, d) => `<div class="kpi glass"><div class="kv">${v}</div><div class="kl">${l}</div><div class="kd">${d || ""}</div></div>`;
    $("wxKpis").innerHTML = Wd.kpis.map((k) => kpi(k.value, k.label, k.detail)).join("");
    $("wxRainFigs").innerHTML = Wd.rain_figures.map((f) => `<img src="${f}" alt="rain validation figure" loading="lazy">`).join("");
    $("wxWindFigs").innerHTML = Wd.wind_figures.map((f) => `<img src="${f}" alt="wind validation figure" loading="lazy">`).join("");
    const table = (rows, models) => `<table><tr><th>Statistic</th><th>Observed</th>${models.map((m) => `<th>${m}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr><td>${r.label}</td><td>${r.observed}</td>${r.cells.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + `</table>`;
    $("wxRainTable").innerHTML = table(Wd.rain_rows, Wd.rain_models);
    $("wxWindTable").innerHTML = table(Wd.wind_rows, Wd.wind_models);
  }

  // ======================================================================
  // Controls
  // ======================================================================
  function populateTraceSelect(selectIndex = 0) {
    $("selTrace").innerHTML = DATA.traces.map((tr, i) => {
      const [label] = REASON[tr.final.reason] || [tr.final.reason];
      return `<option value="${i}">#${tr.seed} ${tr.profile.replace(/_/g, " ")} · ${short(tr.method)} — ${label}</option>`;
    }).join("");
    $("selTrace").value = String(selectIndex);
  }

  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
    setTimeout(() => { scene.resize(); cmp.a.resize(); cmp.b.resize(); if (state.trace) drawTimeline(); }, 30);
  }));
  $("selTrace").onchange = () => { setTrace(DATA.traces[+$("selTrace").value]); state.playing = true; $("btnPlay").textContent = "❚❚"; };
  $("btnPlay").onclick = () => {
    if (state.t >= state.trace.steps.length) { state.t = 0; state.lastStep = -1; $("endBanner").className = "end-banner"; }
    state.playing = !state.playing; $("btnPlay").textContent = state.playing ? "❚❚" : "▶";
  };
  $("btnPrev").onclick = () => { state.playing = false; $("btnPlay").textContent = "▶"; state.t = Math.max(0, Math.ceil(state.t) - 1); state.lastStep = Math.floor(state.t); $("endBanner").className = "end-banner"; };
  $("btnNext").onclick = () => { state.playing = false; $("btnPlay").textContent = "▶"; state.t = Math.min(state.trace.steps.length, Math.floor(state.t) + 1); };
  $("scrub").oninput = (e) => { state.t = +e.target.value; state.lastStep = Math.floor(state.t); $("endBanner").className = state.t >= state.trace.steps.length ? $("endBanner").className : "end-banner"; };
  $("selSpeed").onchange = (e) => (state.speed = +e.target.value);
  ["Forecast", "Fan", "Trail", "Weather"].forEach((n) => $("tg" + n).addEventListener("change", (e) => { const key = n.toLowerCase(); scene.layers[key] = e.target.checked; cmp.a.layers[key] = cmp.b.layers[key] = e.target.checked; }));
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); $("btnPlay").click(); }
    if (e.code === "ArrowRight") $("btnNext").click();
    if (e.code === "ArrowLeft") $("btnPrev").click();
  });
  $("cmpPlay").onclick = () => { if (cmp.t >= +$("cmpScrub").max) cmp.t = 0; cmp.playing = !cmp.playing; $("cmpPlay").textContent = cmp.playing ? "❚❚" : "▶"; };
  $("cmpScrub").oninput = (e) => (cmp.t = +e.target.value);
  window.addEventListener("resize", () => state.trace && drawTimeline());

  // live mode (served by scripts/run_demo.py)
  async function tryLive() {
    try {
      const r = await fetch("/api/meta");
      if (!r.ok) return;
      const meta = await r.json();
      state.live = true;
      document.body.classList.add("live");
      $("modePill").textContent = "● live simulator";
      $("modePill").classList.add("live");
      $("selMethod").innerHTML = meta.methods.map((m) => `<option value="${m}">${short(m)}</option>`).join("");
      $("selMethod").value = meta.methods.includes("ppo_cnn_lstm+probabilistic") ? "ppo_cnn_lstm+probabilistic" : meta.methods[0];
      // Profile list annotated with the selected method's success rate on the held-out test missions.
      const rates = meta.success_rates || {};
      const fillProfiles = () => {
        const keep = $("selProfile").value || "normal";
        $("selProfile").innerHTML = meta.profiles.map((p) => {
          const r = rates[`${$("selMethod").value}|${p}`];
          return `<option value="${p}">${p.replace(/_/g, " ")}${r != null ? ` · ${Math.round(r * 100)}% success` : ""}</option>`;
        }).join("");
        $("selProfile").value = keep;
      };
      fillProfiles();
      $("selMethod").onchange = fillProfiles;
      $("inSeed").value = "2000001"; // a normal mission the full method delivers
      $("btnLucky").onclick = async () => {
        try {
          const r = await fetch(`/api/suggest?method=${encodeURIComponent($("selMethod").value)}&profile=${$("selProfile").value}`);
          const s = await r.json();
          if (s.error) throw new Error(s.error);
          if (s.fallback) toast(`No recorded success for that profile — using “${s.profile.replace(/_/g, " ")}” instead`, "info");
          $("selProfile").value = s.profile;
          $("inSeed").value = String(s.seed);
          $("btnRun").click();
        } catch (err) { toast("Could not pick a mission: " + err.message, "bad"); }
      };
      $("btnRun").onclick = async () => {
        $("btnRun").disabled = true;
        $("loading").classList.add("show");
        $("loadingText").textContent = $("selMethod").value.includes("probabilistic") ? "Imagining 16 forecast-conditioned futures per decision…" : "Simulating mission…";
        try {
          const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ method: $("selMethod").value, profile: $("selProfile").value, seed: +$("inSeed").value }) });
          const tr = await res.json();
          if (tr.error) throw new Error(tr.error);
          DATA.traces.unshift(tr);
          populateTraceSelect(0);
          setupCompare();
          setTrace(tr); state.playing = true; $("btnPlay").textContent = "❚❚";
        } catch (err) { toast("Simulation failed: " + err.message, "bad"); }
        $("btnRun").disabled = false;
        $("loading").classList.remove("show");
      };
    } catch (_) { /* offline replay */ }
  }

  // main loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const clock = now / 1000;
    if (state.trace && $("tab-replay").classList.contains("active")) {
      const n = state.trace.steps.length;
      if (state.playing) {
        state.t = Math.min(n, state.t + dt * 2.4 * state.speed);
        if (state.t >= n) { state.playing = false; $("btnPlay").textContent = "↻"; showEnd(); }
      }
      const k = Math.min(n - 1, Math.floor(state.t));
      if (k !== state.lastStep) { onStepChange(k, state.lastStep); state.lastStep = k; updateHud(k); }
      $("scrub").value = String(state.t);
      scene.render(Math.min(state.t, n - 1e-3), clock);
      drawPlayhead();
    }
    if (cmp.ta && $("tab-compare").classList.contains("active")) {
      const n = +$("cmpScrub").max;
      if (cmp.playing) { cmp.t = Math.min(n, cmp.t + dt * 2.4); if (cmp.t >= n) { cmp.playing = false; $("cmpPlay").textContent = "↻"; } }
      $("cmpScrub").value = String(cmp.t);
      cmp.a.render(Math.min(cmp.t, cmp.ta.steps.length - 1e-3), clock);
      cmp.b.render(Math.min(cmp.t, cmp.tb.steps.length - 1e-3), clock);
      cmpLabel($("cmpLabelA"), cmp.ta, cmp.t);
      cmpLabel($("cmpLabelB"), cmp.tb, cmp.t);
      $("cmpInfo").textContent = `t = ${Math.floor(cmp.t)}`;
    }
    requestAnimationFrame(frame);
  }

  // boot
  if (DATA.traces.length) {
    const start = Math.max(0, DATA.traces.findIndex((t) => t.featured));
    populateTraceSelect(start);
    setTrace(DATA.traces[start]);
    state.playing = true;
    $("btnPlay").textContent = "❚❚";
  }
  setupCompare();
  setupBench();
  setupWeather();
  tryLive();
  requestAnimationFrame(frame);
})();
