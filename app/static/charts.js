/* Dashboard charts - plain SVG/HTML, no chart library.
 *
 * Each chart is a .chart-card with a [data-chart] body. The data comes from
 * window.DASH_STATS (dashboard/routes.py _dashboard_stats). Every chart has:
 *   - a hover tooltip on each mark (hit target larger than the mark)
 *   - a "Table" toggle that swaps the plot for the same numbers as a table,
 *     which is also the relief for the light-mode colours under 3:1 contrast
 * Colours come from the --viz-* custom properties in input.css, so light and
 * dark are each their own validated steps, not an automatic flip.
 */
(function () {
  const S = window.DASH_STATS;
  if (!S) return;

  const esc = (v) => String(v).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  // labels are escaped once here, so every innerHTML below is safe
  ['division_bars', 'land_types', 'activity', 'flood'].forEach((k) =>
    (S[k] || []).forEach((r) => { r.label = esc(r.label); }));

  const NS = 'http://www.w3.org/2000/svg';
  const CAT = ['--viz-1', '--viz-2', '--viz-3', '--viz-4', '--viz-5'];
  const SEQ = ['--viz-seq-1', '--viz-seq-2', '--viz-seq-3', '--viz-seq-4'];

  const fmt = {
    lkr: (v) => v >= 1e9 ? `LKR ${(v / 1e9).toFixed(2)}B`
      : v >= 1e6 ? `LKR ${(v / 1e6).toFixed(2)}M`
      : `LKR ${Math.round(v).toLocaleString('en-US')}`,
    count: (v) => String(v),
  };

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
  }

  /* ---------------------------------------------------------------- tooltip */
  function tooltip(card) {
    let tip = card.querySelector('.chart-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'chart-tooltip';
      tip.hidden = true;
      card.appendChild(tip);
    }
    return {
      show(evt, html) {
        tip.innerHTML = html;
        tip.hidden = false;
        const r = card.getBoundingClientRect();
        let x = evt.clientX - r.left + 12;
        const y = evt.clientY - r.top - 34;
        if (x + tip.offsetWidth > r.width - 8) x = evt.clientX - r.left - tip.offsetWidth - 12;
        tip.style.left = `${x}px`;
        tip.style.top = `${Math.max(4, y)}px`;
      },
      hide() { tip.hidden = true; },
    };
  }

  /* -------------------------------------------------------- horizontal bars */
  function hbar(body, rows, { format, colors, sub }) {
    const card = body.closest('.chart-card');
    const tip = tooltip(card);
    const max = Math.max(...rows.map((r) => r.value), 1);
    const wrap = document.createElement('div');
    wrap.className = 'flex flex-col gap-2';
    rows.forEach((r, i) => {
      const row = document.createElement('div');
      row.className = 'grid items-center gap-3';
      row.style.gridTemplateColumns = 'minmax(6.5rem, 34%) 1fr auto';
      row.innerHTML =
        `<span class="truncate text-xs" style="color:var(--viz-ink-2)">${r.label}</span>` +
        `<span class="relative block h-3.5 rounded-sm" style="background:var(--viz-grid)">` +
        `<span class="absolute inset-y-0 left-0 block" style="width:${Math.max(r.value / max * 100, r.value ? 1.5 : 0)}%;` +
        `background:var(${colors(i)});border-radius:0 4px 4px 0"></span></span>` +
        `<span class="text-xs tabular-nums" style="color:var(--viz-ink)">${fmt[format](r.value)}</span>`;
      row.addEventListener('mousemove', (e) => tip.show(e,
        `<strong>${r.label}</strong><br>${fmt[format](r.value)}${sub ? ` · ${sub(r)}` : ''}`));
      row.addEventListener('mouseleave', tip.hide);
      wrap.appendChild(row);
    });
    body.appendChild(wrap);
  }

  /* ---------------------------------------------------------------- columns */
  function columns(body, rows, { format }) {
    const card = body.closest('.chart-card');
    const tip = tooltip(card);
    const W = 520, H = 190, L = 26, B = 22, T = 8;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
                            'aria-label': 'Estimates per month' }, body);
    const max = Math.max(...rows.map((r) => r.value), 1);
    const top = Math.max(1, Math.ceil(max / 2) * 2);
    const y = (v) => H - B - (v / top) * (H - B - T);
    [0, top / 2, top].forEach((t) => {
      el('line', { x1: L, x2: W, y1: y(t), y2: y(t),
                   stroke: t === 0 ? 'var(--viz-axis)' : 'var(--viz-grid)', 'stroke-width': 1 }, svg);
      const lab = el('text', { x: L - 6, y: y(t) + 3, 'text-anchor': 'end', 'font-size': 10,
                               fill: 'var(--viz-muted)' }, svg);
      lab.textContent = t;
    });
    const step = (W - L) / rows.length;
    const bw = Math.min(46, step - 10);
    rows.forEach((r, i) => {
      const cx = L + step * i + step / 2;
      const x = cx - bw / 2, y0 = H - B, y1 = y(r.value), rad = Math.min(4, (y0 - y1) / 2);
      if (r.value > 0) {
        el('path', { fill: 'var(--viz-1)', d:
          `M${x},${y0} V${y1 + rad} Q${x},${y1} ${x + rad},${y1} H${x + bw - rad} ` +
          `Q${x + bw},${y1} ${x + bw},${y1 + rad} V${y0} Z` }, svg);
      }
      const lab = el('text', { x: cx, y: H - 6, 'text-anchor': 'middle', 'font-size': 10,
                               fill: 'var(--viz-muted)' }, svg);
      lab.textContent = r.label.split(' ')[0];
      const hit = el('rect', { x: L + step * i, y: T, width: step, height: H - B - T,
                               fill: 'transparent' }, svg);
      hit.addEventListener('mousemove', (e) => tip.show(e,
        `<strong>${r.label}</strong><br>${fmt[format](r.value)} estimate${r.value === 1 ? '' : 's'}`));
      hit.addEventListener('mouseleave', tip.hide);
    });
  }

  /* ------------------------------------------------------------------ donut */
  function donut(body, rows) {
    const card = body.closest('.chart-card');
    const tip = tooltip(card);
    const total = rows.reduce((s, r) => s + r.value, 0) || 1;
    const box = document.createElement('div');
    box.className = 'flex flex-col items-center gap-4 sm:flex-row sm:items-center';
    body.appendChild(box);
    const R = 70, r0 = 44, C = 80;
    const svg = el('svg', { viewBox: '0 0 160 160', width: 160, height: 160, role: 'img',
                            'aria-label': 'Estimates by land type', class: 'flex-none' }, box);
    let a = -Math.PI / 2;
    const pt = (ang, rad) => [C + rad * Math.cos(ang), C + rad * Math.sin(ang)];
    rows.forEach((r, i) => {
      const frac = r.value / total;
      const b = a + frac * Math.PI * 2;
      const large = b - a > Math.PI ? 1 : 0;
      let d;
      if (frac >= 0.9999) {
        d = `M${C - R},${C} a${R},${R} 0 1,0 ${2 * R},0 a${R},${R} 0 1,0 ${-2 * R},0 ` +
            `M${C - r0},${C} a${r0},${r0} 0 1,1 ${2 * r0},0 a${r0},${r0} 0 1,1 ${-2 * r0},0 Z`;
      } else {
        const [x1, y1] = pt(a, R), [x2, y2] = pt(b, R), [x3, y3] = pt(b, r0), [x4, y4] = pt(a, r0);
        d = `M${x1},${y1} A${R},${R} 0 ${large} 1 ${x2},${y2} L${x3},${y3} ` +
            `A${r0},${r0} 0 ${large} 0 ${x4},${y4} Z`;
      }
      const seg = el('path', { d, fill: `var(${CAT[i]})`, stroke: 'var(--viz-surface)',
                               'stroke-width': 2, 'fill-rule': 'evenodd' }, svg);
      seg.addEventListener('mousemove', (e) => {
        seg.setAttribute('opacity', 0.85);
        tip.show(e, `<strong>${r.label}</strong><br>${r.value} estimate${r.value === 1 ? '' : 's'} · ${Math.round(frac * 100)}%`);
      });
      seg.addEventListener('mouseleave', () => { seg.removeAttribute('opacity'); tip.hide(); });
      a = b;
    });
    const t1 = el('text', { x: C, y: C - 2, 'text-anchor': 'middle', 'font-size': 22, 'font-weight': 700,
                            fill: 'var(--viz-ink)' }, svg);
    t1.textContent = total;
    const t2 = el('text', { x: C, y: C + 15, 'text-anchor': 'middle', 'font-size': 10,
                            fill: 'var(--viz-muted)' }, svg);
    t2.textContent = 'estimates';

    const ul = document.createElement('ul');
    ul.className = 'chart-legend w-full';
    ul.innerHTML = rows.map((r, i) =>
      `<li><span class="swatch" style="background:var(${CAT[i]})"></span>${r.label}` +
      `<span class="val">${r.value} · ${Math.round(r.value / total * 100)}%</span></li>`).join('');
    box.appendChild(ul);
  }

  /* ------------------------------------------------------------ table view */
  function table(body, rows, { format, head }) {
    const t = document.createElement('table');
    t.className = 'data-table';
    t.hidden = true;
    t.innerHTML = `<thead><tr><th>${head[0]}</th><th>${head[1]}</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.label}</td><td class="tabular-nums">${fmt[format](r.value)}</td></tr>`).join('') +
      '</tbody>';
    body.parentElement.appendChild(t);
    return t;
  }

  const CHARTS = {
    division: {
      rows: S.division_bars, format: 'lkr', head: ['GN division', 'Average price / perch'],
      draw: (b, rows) => hbar(b, rows, { format: 'lkr', colors: () => '--viz-1',
                                         sub: (r) => `${r.count} estimate${r.count === 1 ? '' : 's'}` }),
    },
    landtype: {
      rows: S.land_types, format: 'count', head: ['Land type', 'Estimates'],
      draw: (b, rows) => donut(b, rows),
    },
    activity: {
      rows: S.activity, format: 'count', head: ['Month', 'Estimates'],
      draw: (b, rows) => columns(b, rows, { format: 'count' }),
    },
    flood: {
      rows: S.flood, format: 'count', head: ['Past flooding record', 'Estimates'],
      draw: (b, rows) => hbar(b, rows, { format: 'count', colors: (i) => SEQ[i] }),
    },
  };

  document.querySelectorAll('[data-chart]').forEach((body) => {
    const spec = CHARTS[body.dataset.chart];
    if (!spec || !spec.rows || !spec.rows.length) {
      body.innerHTML = '<p class="m-0 text-xs text-muted">Not enough estimates yet.</p>';
      return;
    }
    spec.draw(body, spec.rows);
    const t = table(body, spec.rows, spec);
    const btn = body.closest('.chart-card').querySelector('.chart-toggle');
    if (btn) {
      btn.addEventListener('click', () => {
        const showTable = t.hidden;
        t.hidden = !showTable;
        body.hidden = showTable;
        btn.textContent = showTable ? 'Chart' : 'Table';
        btn.setAttribute('aria-pressed', String(showTable));
      });
    }
  });
})();
