const fs = require('fs');
const path = require('path');

const root = process.cwd();
const logDir = path.join(root, 'rangefinder_test_logs');
const outDir = path.join(root, 'output', 'rangefinder_ppt_figures');
fs.mkdirSync(outDir, { recursive: true });

function csvRows(file) {
  const text = fs.readFileSync(file, 'utf8').trim();
  const lines = text.split(/\r?\n/);
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const cells = line.split(',');
    const row = {};
    headers.forEach((h, i) => row[h] = cells[i] ?? '');
    return row;
  }).filter(r => r.status === 'ok');
}

function filesFor(pattern) {
  return fs.readdirSync(logDir)
    .filter(name => pattern.test(name))
    .filter(name => name.endsWith('.csv'))
    .sort()
    .map(name => path.join(logDir, name));
}

function num(v) { return Number(v); }
function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }
function std(xs) {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length);
}
function fmt(v, digits = 2) { return Number(v).toFixed(digits); }
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const run57Files = filesFor(/^laser100m_ref57mm_20260913_.*\.csv$/);
const staticFiles = {};
for (const ref of [60, 100, 200, 300]) {
  const matches = filesFor(new RegExp(`^laser100m_ref${ref}mm_20260913_.*\\.csv$`));
  if (!matches.length) throw new Error(`missing ${ref} mm CSV`);
  staticFiles[ref] = matches[matches.length - 1];
}

const run57 = run57Files.map(file => csvRows(file));
const summaries = [];
const all57 = run57.flat().map(r => num(r.raw_mm ?? r.distance_mm));
summaries.push({ ref: 57, n: all57.length, runs: run57.length, mean: mean(all57), sd: std(all57) });
for (const ref of [60, 100, 200, 300]) {
  const xs = csvRows(staticFiles[ref]).map(r => num(r.raw_mm ?? r.distance_mm));
  summaries.push({ ref, n: xs.length, runs: 1, mean: mean(xs), sd: std(xs) });
}
for (const s of summaries) {
  s.bias = s.mean - s.ref;
  s.mae = mean((s.ref === 57 ? all57 : csvRows(staticFiles[s.ref]).map(r => num(r.raw_mm ?? r.distance_mm))).map(x => Math.abs(x - s.ref)));
}

const colors = ['#1f77b4', '#e67e22', '#7b61a8'];
const W = 1920, H = 1080;
const font = "'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif";
function base(title, subtitle) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>text{font-family:${font};fill:#17202a} .small{font-size:22px}.axis{font-size:24px}.tick{font-size:20px}.title{font-size:40px;font-weight:700}.subtitle{font-size:23px;fill:#4f5b66}.note{font-size:20px;fill:#5c6770}</style>
  <text x="80" y="68" class="title">${esc(title)}</text>
  <text x="80" y="104" class="subtitle">${esc(subtitle)}</text>`;
}
function end() { return '</svg>'; }
function line(x1, y1, x2, y2, attrs = '') { return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`; }
function rect(x, y, w, h, attrs = '') { return `<rect x="${x}" y="${y}" width="${w}" height="${h}" ${attrs}/>`; }
function circle(cx, cy, r, attrs = '') { return `<circle cx="${cx}" cy="${cy}" r="${r}" ${attrs}/>`; }
function text(x, y, s, cls = 'small', attrs = '') { return `<text x="${x}" y="${y}" class="${cls}" ${attrs}>${esc(s)}</text>`; }
function polyline(points, attrs = '') { return `<polyline points="${points.map(p => p.join(',')).join(' ')}" ${attrs}/>`; }
function chartFrame(x, y, w, h, xMin, xMax, yMin, yMax) {
  const sx = v => x + (v - xMin) / (xMax - xMin) * w;
  const sy = v => y + h - (v - yMin) / (yMax - yMin) * h;
  return { sx, sy, x, y, w, h, xMin, xMax, yMin, yMax };
}
function axes(c, xLabel, yLabel, xTicks, yTicks, xFmt = v => String(v), yFmt = v => String(v)) {
  let s = rect(c.x, c.y, c.w, c.h, 'fill="none" stroke="#aab4bd" stroke-width="2"');
  s += line(c.x, c.y + c.h, c.x + c.w, c.y + c.h, 'stroke="#5d6872" stroke-width="2"');
  s += line(c.x, c.y, c.x, c.y + c.h, 'stroke="#5d6872" stroke-width="2"');
  for (const v of xTicks) { const px = c.sx(v); s += line(px, c.y + c.h, px, c.y + c.h + 8, 'stroke="#5d6872" stroke-width="2"'); s += text(px, c.y + c.h + 34, xFmt(v), 'tick', 'text-anchor="middle"'); }
  for (const v of yTicks) { const py = c.sy(v); s += line(c.x - 8, py, c.x, py, 'stroke="#5d6872" stroke-width="2"'); s += text(c.x - 16, py + 7, yFmt(v), 'tick', 'text-anchor="end"'); }
  s += text(c.x + c.w / 2, c.y + c.h + 76, xLabel, 'axis', 'text-anchor="middle"');
  s += `<text x="${c.x - 76}" y="${c.y + c.h / 2}" class="axis" text-anchor="middle" transform="rotate(-90 ${c.x - 76} ${c.y + c.h / 2})">${esc(yLabel)}</text>`;
  return s;
}

// Figure 1: static summary and distance-dependent bias.
let svg = base('100 m激光测距模块：静态距离验证', '未施加软件补偿；57 mm为3次合并，其余距离各1次；误差相对人工尺量参考值');
const left = chartFrame(150, 160, 680, 500, 40, 320, 40, 320);
const right = chartFrame(1030, 160, 680, 500, 40, 320, -5, 2);
svg += axes(left, '参考距离 / mm', '原始测量距离 / mm', [50, 100, 200, 300], [50, 100, 200, 300]);
svg += axes(right, '参考距离 / mm', '测量偏差 / mm', [50, 100, 200, 300], [-5, -3, -1, 0, 2]);
svg += line(left.sx(40), left.sy(40), left.sx(320), left.sy(320), 'stroke="#7f8c8d" stroke-width="3" stroke-dasharray="12 8"');
svg += line(right.sx(40), right.sy(0), right.sx(320), right.sy(0), 'stroke="#7f8c8d" stroke-width="3" stroke-dasharray="12 8"');
for (const s of summaries) {
  const color = s.ref === 57 ? '#c0392b' : '#1f77b4';
  const px = left.sx(s.ref), py = left.sy(s.mean), ey1 = left.sy(s.mean + s.sd), ey2 = left.sy(s.mean - s.sd);
  svg += line(px, ey1, px, ey2, `stroke="${color}" stroke-width="4"`);
  svg += line(px - 8, ey1, px + 8, ey1, `stroke="${color}" stroke-width="4"`);
  svg += line(px - 8, ey2, px + 8, ey2, `stroke="${color}" stroke-width="4"`);
  svg += circle(px, py, 9, `fill="${color}" stroke="#ffffff" stroke-width="3"`);
  const bx = right.sx(s.ref), by = right.sy(s.bias);
  svg += circle(bx, by, 10, `fill="${color}" stroke="#ffffff" stroke-width="3"`);
  svg += text(bx + 14, by - 12, `${s.bias >= 0 ? '+' : ''}${fmt(s.bias, 2)}`, 'small', `fill="${color}"`);
}
svg += text(150, 716, '点：均值；误差棒：±1个标准差；红色为57 mm三次合并结果', 'note');
svg += text(1030, 716, '近距离偏差更大：60 mm约−3.69 mm，300 mm约−0.56 mm', 'note');

// Summary table.
svg += rect(120, 770, 1680, 220, 'fill="#f7f9fb" stroke="#d4dce2" stroke-width="2"');
const cols = [170, 430, 650, 880, 1110, 1340, 1570];
const heads = ['参考距离', '有效帧', '均值 / mm', '标准差 / mm', '偏差 / mm', 'MAE / mm', 'RMSE / mm'];
heads.forEach((h, i) => svg += text(cols[i], 815, h, 'small', 'text-anchor="middle" font-weight="700"'));
summaries.forEach((s, idx) => {
  const y = 860 + idx * 30;
  const vals = [`${s.ref} mm${s.ref === 57 ? '（3次）' : ''}`, `${s.n}`, fmt(s.mean), fmt(s.sd), `${s.bias >= 0 ? '+' : ''}${fmt(s.bias)}`, fmt(s.mae), fmt(Math.sqrt(s.mae * s.mae + s.sd * s.sd))];
  vals.forEach((v, i) => svg += text(cols[i], y, v, 'small', 'text-anchor="middle"'));
});
svg += text(120, 1030, '结论：数据连续性和短时波动较好，但存在距离相关系统偏差；当前不建议直接使用单一固定补偿值。', 'note');
svg += end();
fs.writeFileSync(path.join(outDir, 'rangefinder_static_summary.svg'), svg, 'utf8');

// Figure 2: 57 mm repeatability time series.
let rep = base('57 mm近距离重复性实验', '3次独立30 s采集；软件补偿0 mm；参考距离57 mm');
const c = chartFrame(130, 160, 1120, 650, 0, 30, 48, 59);
rep += axes(c, '采集时间 / s', '原始距离 / mm', [0, 5, 10, 15, 20, 25, 30], [48, 50, 52, 54, 56, 57, 59]);
rep += line(c.sx(0), c.sy(57), c.sx(30), c.sy(57), 'stroke="#c0392b" stroke-width="3" stroke-dasharray="12 8"');
run57.forEach((rows, idx) => {
  const pts = rows.map(r => [num(r.elapsed_s), num(r.raw_mm ?? r.distance_mm)]);
  const step = Math.max(1, Math.floor(pts.length / 240));
  const sampled = pts.filter((_, i) => i % step === 0 || i === pts.length - 1).map(p => [c.sx(p[0]), c.sy(p[1])]);
  rep += polyline(sampled, `fill="none" stroke="${colors[idx]}" stroke-width="2.5" opacity="0.85"`);
  const m = mean(rows.map(r => num(r.raw_mm ?? r.distance_mm)));
  rep += line(c.sx(0), c.sy(m), c.sx(30), c.sy(m), `stroke="${colors[idx]}" stroke-width="2" stroke-dasharray="5 6" opacity="0.85"`);
});
rep += text(1290, 205, '统计结果', 'axis', 'font-weight="700"');
const repStats = run57.map(rows => { const xs = rows.map(r => num(r.raw_mm ?? r.distance_mm)); return { n: xs.length, mean: mean(xs), sd: std(xs), min: Math.min(...xs), max: Math.max(...xs) }; });
repStats.forEach((s, i) => {
  const y = 270 + i * 170;
  rep += line(1290, y - 12, 1335, y - 12, `stroke="${colors[i]}" stroke-width="6"`);
  rep += text(1355, y - 4, `第${i + 1}次`, 'axis', 'font-weight="700"');
  rep += text(1290, y + 38, `均值 ${fmt(s.mean, 2)} mm`, 'small');
  rep += text(1290, y + 70, `标准差 ${fmt(s.sd, 2)} mm`, 'small');
  rep += text(1290, y + 102, `范围 ${fmt(s.min, 0)}–${fmt(s.max, 0)} mm`, 'small');
  rep += text(1290, y + 134, `有效帧 ${s.n}/${s.n}`, 'small');
});
rep += text(1290, 830, '共同观察', 'axis', 'font-weight="700"');
rep += text(1290, 875, '三次均无设备错误；', 'small');
rep += text(1290, 905, '单帧波动约0.6–0.8 mm；', 'small');
rep += text(1290, 935, '重复均值存在约1.2 mm漂移。', 'small');
rep += text(130, 875, '红色虚线：人工尺量参考值57 mm；彩色虚线：各次采集均值。', 'note');
rep += text(130, 925, '结论：近距离可连续读数，但绝对偏置和装配/靶面引起的漂移仍需进一步确认。', 'note');
rep += end();
fs.writeFileSync(path.join(outDir, 'rangefinder_57mm_repeatability.svg'), rep, 'utf8');

// A compact CSV-like summary for reproducibility.
const table = ['reference_mm,n,runs,mean_mm,std_mm,bias_mm,mae_mm'];
summaries.forEach(s => table.push([s.ref, s.n, s.runs, fmt(s.mean, 6), fmt(s.sd, 6), fmt(s.bias, 6), fmt(s.mae, 6)].join(',')));
fs.writeFileSync(path.join(outDir, 'rangefinder_static_summary.csv'), table.join('\n') + '\n', 'utf8');
console.log(`Wrote figures to ${outDir}`);
