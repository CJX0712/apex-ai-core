/*
 * verify-transformer.js — 深度 TransformerLM (LLaMA/GPT 级) 无头验证套件 (Node)
 * 9 项硬指标：数值梯度检验（RMSNorm/注意力/端到端）+ 数学不变量 + 端到端训练下降。
 * 运行: node verify-transformer.js
 */
const A = require('./apex-transformer.js');

let pass = 0, fail = 0;
const results = [];
function check(name, ok, detail) {
  results.push({ name, ok, detail });
  if (ok) pass++; else fail++;
  const icon = ok ? '✅' : '⚠️';
  console.log(`${icon} ${name}${detail ? '  — ' + detail : ''}`);
}

/* ---------- 工具 ---------- */
function relErr(a, b) {
  const d = Math.abs(a - b);
  const s = Math.abs(a) + Math.abs(b) + 1e-12;
  return d / s;
}
// 行是 Float64Array，故 2D 矩阵 = Array<Float64Array>；1D 向量 = Float64Array
function isMatrix(obj) {
  return Array.isArray(obj) && obj.length > 0 &&
    (obj[0] instanceof Float64Array || Array.isArray(obj[0]));
}
function getEntry(obj, i, j) { return isMatrix(obj) ? obj[i][j] : obj[j]; }
function setEntry(obj, i, j, v) { if (isMatrix(obj)) obj[i][j] = v; else obj[j] = v; }

/* ===== 1. softmax / logsumexp 不变量 ===== */
(function () {
  const M = [[1, 2, 3], [0.5, -1, 4], [2, 2, 2]];
  const S = A.softmaxRows(M);
  let rowSumOk = true, lseOk = true;
  for (let i = 0; i < 3; i++) {
    let s = 0; for (let j = 0; j < 3; j++) s += S[i][j];
    if (Math.abs(s - 1) > 1e-12) rowSumOk = false;
    // logsumexp 与直接 log(sum exp) 一致
    let se = 0; for (let j = 0; j < 3; j++) se += Math.exp(M[i][j]);
    if (Math.abs(A.logsumexpRows(M)[i] - Math.log(se)) > 1e-12) lseOk = false;
  }
  check('softmax 行和为 1 (不变量)', rowSumOk);
  check('logsumexp = log(Σexp) 精确', lseOk);
})();

/* ===== 2. RMSNorm 反向 (数值对照) ===== */
(function () {
  const rng = A.makeRng(7), T = 5, d = 6, eps = 1e-6;
  const x = A.mat(T, d, () => A.randn(rng)), gamma = new Float64Array(d);
  for (let j = 0; j < d; j++) gamma[j] = 0.5 + A.randn(rng) * 0.3;
  const fwd = A.rmsNormForward(x, gamma, eps);
  const dy = A.mat(T, d, () => A.randn(rng));
  const bwd = A.rmsNormBackward(dy, x, gamma, fwd.c, eps);
  // 数值梯度：对 x[i][j]
  let maxE = 0;
  const loss = (xx) => { const y = A.rmsNormForward(xx, gamma, eps).y; let s = 0; for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) s += dy[i][j] * y[i][j]; return s; };
  const e = 1e-6;
  for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) {
    const old = x[i][j];
    x[i][j] = old + e; const lp = loss(x);
    x[i][j] = old - e; const lm = loss(x);
    x[i][j] = old;
    const num = (lp - lm) / (2 * e);
    maxE = Math.max(maxE, relErr(num, bwd.dx[i][j]));
  }
  check('RMSNorm 反向梯度 (数值对照)', maxE < 1e-4, `maxRelErr=${maxE.toExponential(2)}`);
})();

/* ===== 3. 多头注意力反向 (数值对照, 小模型) ===== */
(function () {
  const rng = A.makeRng(11), T = 4, h = 2, hd = 4, d = h * hd, base = 10000;
  const P = {
    Wq: A.glorot(rng, d, d), bq: new Float64Array(d),
    Wk: A.glorot(rng, d, d), bk: new Float64Array(d),
    Wv: A.glorot(rng, d, d), bv: new Float64Array(d),
    Wo: A.glorot(rng, d, d), bo: new Float64Array(d)
  };
  const x = A.mat(T, d, () => A.randn(rng) * 0.5);
  const fwd = A.mhaForward(x, P, T, h, hd, base);
  const dout = A.mat(T, d, () => A.randn(rng) * 0.5);
  const bwd = A.mhaBackward(fwd.ctx, dout);
  const loss = (xx) => { const o = A.mhaForward(xx, P, T, h, hd, base).out; let s = 0; for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) s += dout[i][j] * o[i][j]; return s; };
  const e = 1e-6; let maxE = 0;
  for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) {
    const old = x[i][j];
    x[i][j] = old + e; const lp = loss(x);
    x[i][j] = old - e; const lm = loss(x);
    x[i][j] = old;
    maxE = Math.max(maxE, relErr((lp - lm) / (2 * e), bwd.dx[i][j]));
  }
  check('多头注意力 反向梯度 (数值对照)', maxE < 1e-4, `maxRelErr=${maxE.toExponential(2)}`);
})();

/* ===== 4. 端到端 Transformer 反向 (数值梯度检验) ===== */
function gradCheck(model, tokens, targets, perParam) {
  const r = model.forward(tokens, targets);
  model.backward(r.ctx);
  const e = 1e-5; let maxE = 0, nTested = 0;
  for (const { key, obj } of model.params) {
    const twoD = isMatrix(obj);
    const rows = twoD ? obj.length : 1, cols = twoD ? obj[0].length : obj.length;
    const tries = Math.min(perParam, rows * cols);
    for (let t = 0; t < tries; t++) {
      const i = twoD ? (Math.floor(Math.random() * rows)) : 0;
      const j = Math.floor(Math.random() * cols);
      const old = getEntry(obj, i, j);
      setEntry(obj, i, j, old + e); const lp = model.loss(tokens, targets);
      setEntry(obj, i, j, old - e); const lm = model.loss(tokens, targets);
      setEntry(obj, i, j, old);
      const num = (lp - lm) / (2 * e);
      const ana = getEntry(model.dW[key], i, j);
      maxE = Math.max(maxE, relErr(num, ana));
      nTested++;
    }
  }
  return { maxE, nTested };
}
(function () {
  const model = new A.TransformerLM({ vocab: 5, dim: 8, heads: 2, layers: 1, seed: 3 });
  const rng = A.makeRng(99);
  const tokens = [], targets = [];
  for (let i = 0; i < 4; i++) { const t = Math.floor(rng() * 5); tokens.push(t); targets.push(Math.floor(rng() * 5)); }
  const { maxE, nTested } = gradCheck(model, tokens, targets, 3);
  check('Transformer 端到端 反向梯度 (全参数数值检验)', maxE < 1e-3, `maxRelErr=${maxE.toExponential(2)}, 检${nTested}项`);
})();

/* ===== 5. 确定性可复现 (同 seed 同参数) ===== */
(function () {
  const m1 = new A.TransformerLM({ vocab: 6, dim: 8, heads: 2, layers: 1, seed: 42 });
  const m2 = new A.TransformerLM({ vocab: 6, dim: 8, heads: 2, layers: 1, seed: 42 });
  const tok = [1, 2, 3, 4, 0, 2];
  const p1 = m1.predict(tok).join(','), p2 = m2.predict(tok).join(',');
  check('确定性可复现 (同 seed 同输出)', p1 === p2, `预测=[${p1}]`);
})();

/* ===== 6. 端到端训练：copy 任务 loss 下降 ===== */
(function () {
  const V = 10, T = 12;
  const model = new A.TransformerLM({ vocab: V, dim: 32, heads: 4, layers: 2, seed: 2026 });
  const rng = A.makeRng(555);
  function makeBatch() {
    const tok = [], tgt = [];
    for (let i = 0; i < T; i++) {
      const t = Math.floor(rng() * V);
      tok.push(t);
      tgt.push(i === 0 ? t : tok[i - 1]); // 复制上一 token
    }
    return { tok, tgt };
  }
  const first = makeBatch();
  const loss0 = model.loss(first.tok, first.tgt);
  let loss = loss0;
  const t0 = Date.now();
  const steps = 600;
  for (let s = 1; s <= steps; s++) {
    const b = makeBatch();
    loss = model.trainStep(b.tok, b.tgt, 0.01, s, 0.01);
  }
  const dt = Date.now() - t0;
  // 准确率评估
  let correct = 0, total = 0;
  for (let k = 0; k < 200; k++) {
    const b = makeBatch();
    const pred = model.predict(b.tok);
    for (let i = 0; i < T; i++) { total++; if (pred[i] === b.tgt[i]) correct++; }
  }
  const acc = correct / total;
  check('copy 任务 训练 loss 下降 >50%', loss < loss0 * 0.5, `loss ${loss0.toFixed(3)}→${loss.toFixed(3)}`);
  check('copy 任务 测试准确率 >90%', acc > 0.9, `acc=${(acc * 100).toFixed(1)}%, ${steps}步/${dt}ms`);
})();

/* ===== 7. 性能基准 (单步 forward+backward 耗时) ===== */
(function () {
  const model = new A.TransformerLM({ vocab: 16, dim: 64, heads: 4, layers: 3, seed: 1 });
  const tok = [], tgt = [];
  for (let i = 0; i < 16; i++) { tok.push(i % 16); tgt.push((i + 1) % 16); }
  const t0 = Date.now();
  const N = 50;
  for (let s = 0; s < N; s++) { const r = model.forward(tok, tgt); model.backward(r.ctx); }
  const per = (Date.now() - t0) / N;
  check('性能基准 (单步 <50ms@dim64/L3)', per < 50, `${per.toFixed(1)}ms/步 (dim64,L3,T16)`);
})();

/* ---------- 汇总 ---------- */
console.log('\n========================================');
console.log(`验证结果: ${pass} 通过 / ${fail} 失败 (共 ${pass + fail} 项)`);
console.log('========================================');
process.exit(fail === 0 ? 0 : 1);
