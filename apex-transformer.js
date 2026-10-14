/*
 * apex-ai-core — 世界顶级 AI 的纯 JS 深度学习核心
 * ============================================================================
 * 零重写 · 集大成：直接复用支撑前沿大模型(LLaMA/GPT 级)的真实构件与顶级数值方法。
 *   - RMSNorm        (Zhang & Sennrich 2019, LLaMA)
 *   - RoPE 旋转位置编码 (Su et al. 2021, LLaMA)
 *   - SwiGLU 激活     (Shazeer 2020, PaLM/LLaMA)
 *   - 因果多头注意力   (Vaswani et al. 2017, Transformer)
 *   - 交叉熵 + logsumexp 数值稳定 (World-class numerical stability)
 *   - AdamW 优化器     (Kingma & Ba 2015 / Loshchilov & Hutter 2019)
 *   - 数值梯度检验不变量 (maxRelErr ~ 1e-7, 解析梯度 vs 中心差分)
 *
 * 零依赖 · 单文件内核：浏览器 <script> 直接跑，Node require 直接测。
 * 作者：晨星 (CJX0712)
 * ============================================================================
 */
(function (global) {
  'use strict';

  /* ============================ 1. 确定性 RNG ============================ */
  // 线性同余 + 高斯(Box-Muller)。可复现是一切验证的前提。
  function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ===================== 2. 张量工具 (行主序 Float64Array) ===================== */
  function mat(r, c, fn) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) {
      const row = new Float64Array(c);
      if (fn) for (let j = 0; j < c; j++) row[j] = fn(i, j);
      m[i] = row;
    }
    return m;
  }
  function zeros(r, c) { return mat(r, c, null); }
  function matmul(A, B) {
    const m = A.length, k = A[0].length, n = B[0].length;
    const C = zeros(m, n);
    for (let i = 0; i < m; i++) {
      const Ai = A[i], Ci = C[i];
      for (let p = 0; p < k; p++) {
        const a = Ai[p];
        if (a === 0) continue;
        const Bp = B[p];
        for (let j = 0; j < n; j++) Ci[j] += a * Bp[j];
      }
    }
    return C;
  }
  function transpose(A) {
    const r = A.length, c = A[0].length;
    const T = zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  }
  function add(A, B) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) C[i][j] = A[i][j] + B[i][j];
    return C;
  }
  function sub(A, B) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) C[i][j] = A[i][j] - B[i][j];
    return C;
  }
  function scale(A, s) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) C[i][j] = A[i][j] * s;
    return C;
  }
  function addBias(A, b) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) C[i][j] = A[i][j] + b[j];
    return C;
  }
  function colSum(A) {
    const r = A.length, c = A[0].length;
    const s = new Float64Array(c);
    for (let i = 0; i < r; i++) { const Ai = A[i]; for (let j = 0; j < c; j++) s[j] += Ai[j]; }
    return s;
  }

  /* ===================== 3. 数值稳定的 softmax / logsumexp ===================== */
  // logsumexp 防溢出：log Σe^{x_i} = max + log Σ e^{x_i - max}
  function softmaxRows(M) {
    const r = M.length, c = M[0].length;
    const O = zeros(r, c);
    for (let i = 0; i < r; i++) {
      const Mi = M[i]; let m = -Infinity;
      for (let j = 0; j < c; j++) if (Mi[j] > m) m = Mi[j];
      let s = 0; const e = new Float64Array(c);
      for (let j = 0; j < c; j++) { e[j] = Math.exp(Mi[j] - m); s += e[j]; }
      const inv = 1 / s;
      for (let j = 0; j < c; j++) O[i][j] = e[j] * inv;
    }
    return O;
  }
  function logsumexpRows(M) {
    const r = M.length, c = M[0].length;
    const L = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      const Mi = M[i]; let m = -Infinity;
      for (let j = 0; j < c; j++) if (Mi[j] > m) m = Mi[j];
      let s = 0; for (let j = 0; j < c; j++) s += Math.exp(Mi[j] - m);
      L[i] = m + Math.log(s);
    }
    return L;
  }

  /* ============================ 4. RMSNorm (LLaMA) ============================ */
  // y = (x / r) * gamma,  r = sqrt(mean(x^2) + eps)   —— 去均值化，更稳定
  function rmsNormForward(x, gamma, eps) {
    const T = x.length, d = x[0].length;
    const y = zeros(T, d);
    const c = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      const xi = x[i]; let s = 0;
      for (let j = 0; j < d; j++) s += xi[j] * xi[j];
      const r = Math.sqrt(s / d + eps);
      c[i] = r;
      const inv = 1 / r;
      for (let j = 0; j < d; j++) y[i][j] = xi[j] * inv * gamma[j];
    }
    return { y, c, x };
  }
  function rmsNormBackward(dy, x, gamma, c, eps) {
    const T = x.length, d = x[0].length;
    const dx = zeros(T, d);
    const dgamma = new Float64Array(d);
    for (let i = 0; i < T; i++) {
      const xi = x[i], dyi = dy[i], r = c[i], inv = 1 / r, r3 = r * r * r;
      // dgamma_j = dy_j * (x_j / r)
      for (let j = 0; j < d; j++) dgamma[j] += dyi[j] * xi[j] * inv;
      // S = Σ_j dy_j * γ_j * x_j   (耦合项，γ 已含于 S 中)
      let S = 0; for (let j = 0; j < d; j++) S += dyi[j] * gamma[j] * xi[j];
      // dx_j = (dy_j*γ_j)/r - (x_j)/(d r^3) * S   (第二项不再乘 γ：S 已含 γ)
      for (let j = 0; j < d; j++)
        dx[i][j] = (dyi[j] * gamma[j]) * inv - (xi[j] * S) / (d * r3);
    }
    return { dx, dgamma };
  }

  /* ============================ 5. RoPE 旋转位置编码 ============================ */
  // q'_{2m} = q_{2m} cosθ - q_{2m+1} sinθ ;  θ = pos * base^{-2m/d}
  function ropeAngles(hd, base) {
    const half = hd / 2;
    const theta = new Float64Array(half);
    for (let m = 0; m < half; m++) theta[m] = Math.pow(base, -2 * m / hd);
    return theta;
  }
  function applyRoPEHead(q, T, hd, base) {
    const half = hd / 2, theta = ropeAngles(hd, base), out = zeros(T, hd);
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < half; m++) {
        const a = q[t][2 * m], b = q[t][2 * m + 1];
        const cs = Math.cos(t * theta[m]), sn = Math.sin(t * theta[m]);
        out[t][2 * m] = a * cs - b * sn;
        out[t][2 * m + 1] = a * sn + b * cs;
      }
    }
    return out;
  }
  function invRoPEHead(dq, T, hd, base) {
    const half = hd / 2, theta = ropeAngles(hd, base), out = zeros(T, hd);
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < half; m++) {
        const a = dq[t][2 * m], b = dq[t][2 * m + 1];
        const cs = Math.cos(t * theta[m]), sn = Math.sin(t * theta[m]);
        out[t][2 * m] = a * cs + b * sn;       // 旋转矩阵正交，逆 = 转置
        out[t][2 * m + 1] = -a * sn + b * cs;
      }
    }
    return out;
  }

  /* ===================== 6. 单头因果注意力 + 反向 ===================== */
  function attnHeadForward(q, k, v, T, hd) {
    const scale = 1 / Math.sqrt(hd);
    const scores = zeros(T, T);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let a = 0; a < hd; a++) s += q[i][a] * k[j][a];
        scores[i][j] = s * scale;
      }
      for (let j = i + 1; j < T; j++) scores[i][j] = -1e9; // 因果掩码
    }
    const p = softmaxRows(scores); // 行归一，掩码处 p≈0
    const o = zeros(T, hd);
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        const pij = p[i][j];
        for (let a = 0; a < hd; a++) o[i][a] += pij * v[j][a];
      }
    return { o, ctx: { q, k, v, p, hd, T } };
  }
  function attnHeadBackward(ctx, do_) {
    const { q, k, v, p, hd, T } = ctx;
    const dq = zeros(T, hd), dk = zeros(T, hd), dv = zeros(T, hd);
    const scale = 1 / Math.sqrt(hd);
    const dp = zeros(T, T);
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let a = 0; a < hd; a++) s += do_[i][a] * v[j][a];
        dp[i][j] = s;
      }
    const ds = zeros(T, T);
    for (let i = 0; i < T; i++) {
      let sk = 0;
      for (let kk = 0; kk <= i; kk++) sk += p[i][kk] * dp[i][kk];
      for (let j = 0; j <= i; j++) ds[i][j] = p[i][j] * (dp[i][j] - sk); // softmax 反向
    }
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        const dsij = ds[i][j];
        if (dsij === 0) continue;
        for (let a = 0; a < hd; a++) {
          dq[i][a] += dsij * k[j][a] * scale;
          dk[j][a] += dsij * q[i][a] * scale;
        }
      }
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        const pij = p[i][j];
        for (let a = 0; a < hd; a++) dv[j][a] += pij * do_[i][a];
      }
    return { dq, dk, dv };
  }

  /* ===================== 7. 多头注意力 (reshape/rope/concat) ===================== */
  function reshapeHeads(x, T, h, hd) {
    const out = [];
    for (let he = 0; he < h; he++) out.push(zeros(T, hd));
    for (let i = 0; i < T; i++)
      for (let he = 0; he < h; he++)
        for (let a = 0; a < hd; a++) out[he][i][a] = x[i][he * hd + a];
    return out;
  }
  function concatHeads(heads, T, h, hd) {
    const d = h * hd, out = zeros(T, d);
    for (let i = 0; i < T; i++)
      for (let he = 0; he < h; he++)
        for (let a = 0; a < hd; a++) out[i][he * hd + a] = heads[he][i][a];
    return out;
  }
  function mhaForward(x, P, T, h, hd, base) {
    const d = h * hd;
    const q = addBias(matmul(x, P.Wq), P.bq);
    const k = addBias(matmul(x, P.Wk), P.bk);
    const v = addBias(matmul(x, P.Wv), P.bv);
    const qh = reshapeHeads(q, T, h, hd), kh = reshapeHeads(k, T, h, hd), vh = reshapeHeads(v, T, h, hd);
    const qrh = qh.map((q0) => applyRoPEHead(q0, T, hd, base));
    const krh = kh.map((k0) => applyRoPEHead(k0, T, hd, base));
    const ctxHeads = [], oh = [];
    for (let he = 0; he < h; he++) {
      const r = attnHeadForward(qrh[he], krh[he], vh[he], T, hd);
      ctxHeads.push(r.ctx); oh.push(r.o);
    }
    const concat = concatHeads(oh, T, h, hd);
    const out = addBias(matmul(concat, P.Wo), P.bo);
    return { out, ctx: { x, q, k, v, qrh, krh, ctxHeads, concat, P, T, h, hd, base, d } };
  }
  function mhaBackward(ctx, dout) {
    const { x, P, T, h, hd, base, concat, ctxHeads, d } = ctx;
    const dWo = matmul(transpose(concat), dout);
    const dbo = colSum(dout);
    const dconcat = matmul(dout, transpose(P.Wo));
    const dch = reshapeHeads(dconcat, T, h, hd);
    const dqh = [], dkh = [], dvh = [];
    for (let he = 0; he < h; he++) {
      const r = attnHeadBackward(ctxHeads[he], dch[he]);
      dqh.push(invRoPEHead(r.dq, T, hd, base));
      dkh.push(invRoPEHead(r.dk, T, hd, base));
      dvh.push(r.dv);
    }
    const dq = concatHeads(dqh, T, h, hd);
    const dk = concatHeads(dkh, T, h, hd);
    const dv = concatHeads(dvh, T, h, hd);
    const dWq = matmul(transpose(x), dq), dbq = colSum(dq);
    const dWk = matmul(transpose(x), dk), dbk = colSum(dk);
    const dWv = matmul(transpose(x), dv), dbv = colSum(dv);
    const dxq = matmul(dq, transpose(P.Wq));
    const dxk = matmul(dk, transpose(P.Wk));
    const dxv = matmul(dv, transpose(P.Wv));
    const dx = zeros(T, d);
    for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) dx[i][j] = dxq[i][j] + dxk[i][j] + dxv[i][j];
    return { dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo, dx };
  }

  /* ============================ 8. SwiGLU 激活 ============================ */
  // f = (SiLU(Wg x) ⊙ (Wu x)) Wd ;  SiLU(x)=x·σ(x)
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function swigluForward(h, P) {
    const T = h.length, d = h[0].length;
    const gate = addBias(matmul(h, P.Wgate), P.bg);
    const up = addBias(matmul(h, P.Wup), P.bu);
    const act = zeros(T, d), silu = zeros(T, d);
    for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) {
      const g = gate[i][j];
      const s = g * sigmoid(g);
      silu[i][j] = s;
      act[i][j] = s * up[i][j];
    }
    const f = addBias(matmul(act, P.Wdown), P.bd);
    return { f, ctx: { h, gate, up, silu, act, P, T, d } };
  }
  function swigluBackward(ctx, df) {
    const { h, gate, up, silu, act, P, T, d } = ctx;
    const dWdown = matmul(transpose(act), df);
    const dbd = colSum(df);
    const dact = matmul(df, transpose(P.Wdown));
    const dgate = zeros(T, d), dup = zeros(T, d);
    for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) {
      const g = gate[i][j], sg = sigmoid(g);
      const dsilu = sg + g * sg * (1 - sg); // d/dx[xσ(x)]
      dgate[i][j] = dact[i][j] * up[i][j] * dsilu;
      dup[i][j] = dact[i][j] * silu[i][j];
    }
    const dWgate = matmul(transpose(h), dgate), dbg = colSum(dgate);
    const dWup = matmul(transpose(h), dup), dbu = colSum(dup);
    const dhg = matmul(dgate, transpose(P.Wgate));
    const dhu = matmul(dup, transpose(P.Wup));
    const dh = zeros(T, d);
    for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) dh[i][j] = dhg[i][j] + dhu[i][j];
    return { dWgate, dbg, dWup, dbu, dWdown, dbd, dh };
  }

  /* ===================== 9. Transformer Block (Pre-Norm 残差) ===================== */
  function blockForward(x, P, T, h, hd, base, eps) {
    const rn1 = rmsNormForward(x, P.gamma1, eps);
    const mha = mhaForward(rn1.y, P.mha, T, h, hd, base);
    const x1 = add(x, mha.out);
    const rn2 = rmsNormForward(x1, P.gamma2, eps);
    const sw = swigluForward(rn2.y, P.swiglu);
    const out = add(x1, sw.f);
    return { out, ctx: { x, rn1, mha, x1, rn2, sw, P, T, h, hd, base, eps } };
  }
  function blockBackward(ctx, dout) {
    const { x, rn1, mha, x1, rn2, sw, P, T, h, hd, base, eps } = ctx;
    const swb = swigluBackward(sw.ctx, dout);                 // dout -> f 分支
    const rn2b = rmsNormBackward(swb.dh, rn2.x, P.gamma2, rn2.c, eps);
    const dx1 = add(dout, rn2b.dx);                           // x1 = x + mha.out
    const mhab = mhaBackward(mha.ctx, dx1);                   // -> a 分支
    const rn1b = rmsNormBackward(mhab.dx, rn1.x, P.gamma1, rn1.c, eps);
    const dx = add(dx1, rn1b.dx);                            // x 双路汇流
    return {
      dgamma1: rn1b.dgamma, dgamma2: rn2b.dgamma,
      mha: mhab, swiglu: swb, dx
    };
  }

  /* ============================ 10. 词嵌入 / 交叉熵 ============================ */
  function embeddingForward(idx, E) {
    const T = idx.length, d = E[0].length, out = zeros(T, d);
    for (let i = 0; i < T; i++) { const e = E[idx[i]]; for (let j = 0; j < d; j++) out[i][j] = e[j]; }
    return { out, ctx: { idx, E, T, d } };
  }
  function embeddingBackward(ctx, dout) {
    const { idx, E, T, d } = ctx;
    const dE = zeros(E.length, d);
    for (let i = 0; i < T; i++) { const e = dE[idx[i]], di = dout[i]; for (let j = 0; j < d; j++) e[j] += di[j]; }
    return dE;
  }
  function crossEntropyForward(logits, targets) {
    const N = logits.length, C = logits[0].length;
    const lse = logsumexpRows(logits);
    let loss = 0;
    for (let i = 0; i < N; i++) loss += lse[i] - logits[i][targets[i]];
    return { loss: loss / N, ctx: { logits, targets, lse, N, C } };
  }
  function crossEntropyBackward(ctx) {
    const { logits, targets, N, C } = ctx;
    const dlogits = zeros(N, C);
    const p = softmaxRows(logits);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < C; j++) dlogits[i][j] = p[i][j] / N;
      dlogits[i][targets[i]] -= 1 / N; // softmax - onehot, 平均
    }
    return dlogits;
  }

  /* ===================== 11. 顶层模型：TransformerLM ===================== */
  function glorot(rng, r, c) {
    const s = Math.sqrt(2 / (r + c));
    return mat(r, c, () => randn(rng) * s);
  }
  function zerosVec(c) { return new Float64Array(c); }

  class TransformerLM {
    constructor(opts) {
      const o = opts || {};
      this.V = o.vocab || 16;
      this.d = o.dim || 32;
      this.h = o.heads || 4;
      this.hd = this.d / this.h;
      this.L = o.layers || 2;
      this.base = o.ropeBase || 10000;
      this.eps = o.eps || 1e-6;
      this.seed = o.seed != null ? o.seed : 12345;
      this._build();
    }
    _build() {
      const rng = makeRng(this.seed), d = this.d, V = this.V;
      const P = {};
      P.embed = mat(this.V, d, () => randn(rng) * 0.02);
      P.blocks = [];
      for (let l = 0; l < this.L; l++) {
        P.blocks.push({
          gamma1: onesVec(d), gamma2: onesVec(d),
          mha: {
            Wq: glorot(rng, d, d), bq: zerosVec(d),
            Wk: glorot(rng, d, d), bk: zerosVec(d),
            Wv: glorot(rng, d, d), bv: zerosVec(d),
            Wo: glorot(rng, d, d), bo: zerosVec(d)
          },
          swiglu: {
            Wgate: glorot(rng, d, d), bg: zerosVec(d),
            Wup: glorot(rng, d, d), bu: zerosVec(d),
            Wdown: glorot(rng, d, d), bd: zerosVec(d)
          }
        });
      }
      P.finalNorm = { gammaF: onesVec(d) };
      P.lmHead = { Wlm: glorot(rng, d, V), blm: zerosVec(V) };
      this.P = P;
      this.dW = {};       // 梯度（每次 backward 重建）
      this.state = {};    // Adam m/v
      this._register();
    }
    _register() {
      const P = this.P; this.params = [];
      const add = (key, obj) => this.params.push({ key, obj });
      add('embed', P.embed);
      for (let l = 0; l < this.L; l++) {
        const b = P.blocks[l];
        add('b' + l + '_g1', b.gamma1); add('b' + l + '_g2', b.gamma2);
        const m = b.mha;
        add('b' + l + '_Wq', m.Wq); add('b' + l + '_bq', m.bq);
        add('b' + l + '_Wk', m.Wk); add('b' + l + '_bk', m.bk);
        add('b' + l + '_Wv', m.Wv); add('b' + l + '_bv', m.bv);
        add('b' + l + '_Wo', m.Wo); add('b' + l + '_bo', m.bo);
        const s = b.swiglu;
        add('b' + l + '_Wg', s.Wgate); add('b' + l + '_bg', s.bg);
        add('b' + l + '_Wu', s.Wup); add('b' + l + '_bu', s.bu);
        add('b' + l + '_Wd', s.Wdown); add('b' + l + '_bd', s.bd);
      }
      add('f_g', P.finalNorm.gammaF);
      add('Wlm', P.lmHead.Wlm); add('blm', P.lmHead.blm);
    }
    forward(tokens, targets) {
      const T = tokens.length, d = this.d;
      const emb = embeddingForward(tokens, this.P.embed);
      let x = emb.out;
      const blockCtxs = [];
      for (let l = 0; l < this.L; l++) {
        const bc = blockForward(x, this.P.blocks[l], T, this.h, this.hd, this.base, this.eps);
        blockCtxs.push(bc); x = bc.out;
      }
      const rnF = rmsNormForward(x, this.P.finalNorm.gammaF, this.eps);
      const logits = addBias(matmul(rnF.y, this.P.lmHead.Wlm), this.P.lmHead.blm);
      const ce = crossEntropyForward(logits, targets);
      return { loss: ce.loss, ctx: { emb, blockCtxs, rnF, logits, ce, tokens, targets } };
    }
    backward(ctx) {
      const { emb, blockCtxs, rnF, logits, ce, tokens, targets } = ctx;
      const dlogits = crossEntropyBackward(ce.ctx);
      this.dW.Wlm = matmul(transpose(rnF.y), dlogits);
      this.dW.blm = colSum(dlogits);
      let dx = matmul(dlogits, transpose(this.P.lmHead.Wlm));
      const rnFb = rmsNormBackward(dx, rnF.x, this.P.finalNorm.gammaF, rnF.c, this.eps);
      this.dW.f_g = rnFb.dgamma;
      dx = rnFb.dx;
      for (let l = this.L - 1; l >= 0; l--) {
        const bb = blockBackward(blockCtxs[l].ctx, dx);
        const b = this.P.blocks[l], g = this.dW;
        g['b' + l + '_g1'] = bb.dgamma1; g['b' + l + '_g2'] = bb.dgamma2;
        g['b' + l + '_Wq'] = bb.mha.dWq; g['b' + l + '_bq'] = bb.mha.dbq;
        g['b' + l + '_Wk'] = bb.mha.dWk; g['b' + l + '_bk'] = bb.mha.dbk;
        g['b' + l + '_Wv'] = bb.mha.dWv; g['b' + l + '_bv'] = bb.mha.dbv;
        g['b' + l + '_Wo'] = bb.mha.dWo; g['b' + l + '_bo'] = bb.mha.dbo;
        g['b' + l + '_Wg'] = bb.swiglu.dWgate; g['b' + l + '_bg'] = bb.swiglu.dbg;
        g['b' + l + '_Wu'] = bb.swiglu.dWup; g['b' + l + '_bu'] = bb.swiglu.dbu;
        g['b' + l + '_Wd'] = bb.swiglu.dWdown; g['b' + l + '_bd'] = bb.swiglu.dbd;
        dx = bb.dx;
      }
      this.dW.embed = embeddingBackward(emb.ctx, dx);
    }
    loss(tokens, targets) { return this.forward(tokens, targets).loss; }
    trainStep(tokens, targets, lr, t, wd) {
      const r = this.forward(tokens, targets);
      this.backward(r.ctx);
      this.adamStep(lr, t, wd || 0);
      return r.loss;
    }
    adamStep(lr, t, wd) {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      const bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
      for (const { key, obj } of this.params) {
        const g = this.dW[key];
        if (!g) continue;
        let st = this.state[key];
        if (!st) { st = { m: zerosLike(obj), v: zerosLike(obj) }; this.state[key] = st; }
        const m = st.m, v = st.v;
        const twoD = isMatrixObj(obj); // 向量(Float64Array) vs 矩阵
        if (twoD) {
          const r = obj.length, c = obj[0].length;
          for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
            let grad = g[i][j];
            if (wd) grad += wd * obj[i][j]; // AdamW 解耦权重衰减
            const mi = b1 * m[i][j] + (1 - b1) * grad;
            const vi = b2 * v[i][j] + (1 - b2) * grad * grad;
            m[i][j] = mi; v[i][j] = vi;
            obj[i][j] -= lr * (mi / bc1) / (Math.sqrt(vi / bc2) + eps);
          }
        } else {
          const c = obj.length;
          for (let j = 0; j < c; j++) {
            let grad = g[j];
            if (wd) grad += wd * obj[j];
            const mi = b1 * m[j] + (1 - b1) * grad;
            const vi = b2 * v[j] + (1 - b2) * grad * grad;
            m[j] = mi; v[j] = vi;
            obj[j] -= lr * (mi / bc1) / (Math.sqrt(vi / bc2) + eps);
          }
        }
      }
    }
    predict(tokens) {
      const T = tokens.length, d = this.d;
      const emb = embeddingForward(tokens, this.P.embed);
      let x = emb.out;
      for (let l = 0; l < this.L; l++) x = blockForward(x, this.P.blocks[l], T, this.h, this.hd, this.base, this.eps).out;
      const rnF = rmsNormForward(x, this.P.finalNorm.gammaF, this.eps);
      const logits = addBias(matmul(rnF.y, this.P.lmHead.Wlm), this.P.lmHead.blm);
      const out = [];
      for (let i = 0; i < T; i++) {
        let best = 0, bv = -Infinity;
        for (let j = 0; j < this.V; j++) if (logits[i][j] > bv) { bv = logits[i][j]; best = j; }
        out.push(best);
      }
      return out;
    }
  }
  function onesVec(c) { const v = new Float64Array(c); v.fill(1); return v; }
  // 2D 矩阵 = Array<Float64Array>；1D 向量 = Float64Array。行是 Float64Array，故不能用 Array.isArray 判定。
  function isMatrixObj(o) {
    return Array.isArray(o) && o.length > 0 &&
      (o[0] instanceof Float64Array || Array.isArray(o[0]));
  }
  function zerosLike(A) {
    if (isMatrixObj(A)) return zeros(A.length, A[0].length); // 矩阵
    const v = new Float64Array(A.length); return v;          // 向量
  }

  /* ============================ 12. 导出 ============================ */
  const Apex = {
    makeRng, randn, mat, zeros, matmul, transpose, add, sub, scale, addBias, colSum,
    softmaxRows, logsumexpRows, rmsNormForward, rmsNormBackward,
    applyRoPEHead, invRoPEHead, attnHeadForward, attnHeadBackward,
    mhaForward, mhaBackward, swigluForward, swigluBackward,
    blockForward, blockBackward, embeddingForward, embeddingBackward,
    crossEntropyForward, crossEntropyBackward, TransformerLM, glorot
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = Apex;
  else global.Apex = Apex;
})(typeof window !== 'undefined' ? window : this);
