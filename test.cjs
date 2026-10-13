/* APEX 全量不变量自检 — Node 运行：node test.cjs
 * 每条算法配可独立验证的「数学不变量」，全部通过即 8/8(全绿)。 */
const A = require('./apex-ai.js');
const { M, Optim, DataGen, AutoGrad, Transformer, LinearRegression, LogisticRegression, SVM, kNN, NaiveBayes, DecisionTree, KMeans, PCA, GMM, MLP, Agent } = A;

let pass = 0, fail = 0; const fails = [];
const approx = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
const maxAbs = (X) => { let m = 0; for (const r of X) for (const v of r) m = Math.max(m, Math.abs(v)); return m; };
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  ✅ ${name}${detail ? '  ' + detail : ''}`); }
  else { fail++; fails.push(name); console.log(`  ❌ ${name}  ${detail}`); }
}
const I = (n) => M.identity(n);

(async () => {
  A.seed(20261014); // 固定随机种子，保证结果完全可复现
  console.log(`\n=== APEX v${A.version} 不变量自检 ===\n`);

  // ---------- L1–L2 线性代数 ----------
  console.log('[L2] 线性代数内核');
  {
    const Amat = [[4, 2, 1], [2, 5, 3], [1, 3, 6]];
    const Ainv = M.invert(Amat);
    check('matmul 逆', maxAbs(M.sub(M.matmul(Amat, Ainv), I(3))) < 1e-9, `‖A A⁻¹ - I‖=${maxAbs(M.sub(M.matmul(Amat, Ainv), I(3))).toExponential(2)}`);

    const S = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]; // SPD
    const L = M.cholesky(S);
    check('Cholesky A=LLᵀ', maxAbs(M.sub(M.matmul(L, M.transpose(L)), S)) < 1e-9);

    const B = [[1, 2, 3], [4, 5, 6], [7, 8, 10]];
    const { Q, R } = M.qr(B);
    check('QR QᵀQ=I', maxAbs(M.sub(M.matmul(M.transpose(Q), Q), I(3))) < 1e-9);
    check('QR A=QR', maxAbs(M.sub(M.matmul(Q, R), B)) < 1e-9);

    const Sym = [[2, 1, 0], [1, 2, 1], [0, 1, 2]];
    const { eigenvalues, eigenvectors } = M.eigSym(Sym);
    check('eigSym 特征值正确', approx(eigenvalues[0], 2 - Math.sqrt(2), 1e-6) && approx(eigenvalues[2], 2 + Math.sqrt(2), 1e-6), `λ=${eigenvalues.map(v => v.toFixed(4))}`);
    // A V = V D 残差
    let resid = 0; for (let j = 0; j < 3; j++) { const av = M.matmul(Sym, [[eigenvectors[0][j]], [eigenvectors[1][j]], [eigenvectors[2][j]]]); const dv = eigenvalues[j] * eigenvectors[0][j]; resid = Math.max(resid, Math.abs(av[0][0] - eigenvalues[j] * eigenvectors[0][j]), Math.abs(av[1][0] - eigenvalues[j] * eigenvectors[1][j]), Math.abs(av[2][0] - eigenvalues[j] * eigenvectors[2][j])); }
    check('eigSym AV=VD 残差', resid < 1e-8, `resid=${resid.toExponential(2)}`);
    // 正交性
    let ortho = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let d = 0; for (let k = 0; k < 3; k++) d += eigenvectors[k][i] * eigenvectors[k][j]; if (i !== j) ortho = Math.max(ortho, Math.abs(d)); else ortho = Math.max(ortho, Math.abs(d - 1)); }
    check('eigSym 特征向量正交归一', ortho < 1e-8);

    const Mx = [[1, 2], [3, 4], [5, 6]]; // 3x2
    const { U, S: Sv, V } = M.svd(Mx);
    const Smat = M.zeros(2, 2); for (let i = 0; i < 2; i++) Smat[i][i] = Sv[i];
    const recon = M.matmul(M.matmul(U, Smat), M.transpose(V));
    check('SVD A≈UΣVᵀ', maxAbs(M.sub(recon, Mx)) < 1e-7, `err=${maxAbs(M.sub(recon, Mx)).toExponential(2)}`);
    check('SVD UᵀU=I', maxAbs(M.sub(M.matmul(M.transpose(U), U), I(2))) < 1e-9);
  }

  // ---------- L3 数值优化 ----------
  console.log('[L3] 数值优化');
  {
    // f(x)=x²+y² 最小值 (0,0)
    const f = (x) => x[0] * x[0] + x[1] * x[1];
    const grad = (x) => [2 * x[0], 2 * x[1]];
    const r1 = Optim.gradientDescent(f, grad, [3, 3], { lr: 0.1, iters: 500, momentum: 0.9 });
    check('GD 收敛到最小值', approx(r1.fx, 0, 1e-4), `f=${r1.fx.toExponential(2)}`);
    const r2 = Optim.adam(f, grad, [3, 3], { lr: 0.1, iters: 300 });
    check('Adam 收敛到最小值', approx(r2.fx, 0, 1e-4), `f=${r2.fx.toExponential(2)}`);
    // 数值梯度 vs 解析
    const f3 = (x) => x[0] * x[0] * x[1] + Math.sin(x[1]);
    const gNum = Optim.numericalGradient(f3, [1.5, 0.5]);
    const gAna = [2 * 1.5 * 0.5, 1.5 * 1.5 + Math.cos(0.5)];
    check('数值梯度≈解析梯度', Math.abs(gNum[0] - gAna[0]) < 1e-5 && Math.abs(gNum[1] - gAna[1]) < 1e-5, `num=${gNum.map(v => v.toFixed(3))} ana=${gAna.map(v => v.toFixed(3))}`);
  }

  // ---------- L4 监督学习 ----------
  console.log('[L4] 监督学习');
  {
    const { X: Xb, y: yb } = DataGen.blobs(120, 3, 2);
    const yNum = yb.map(v => v * 1.0);
    const lr = new LinearRegression(0).fit(Xb, yNum);
    check('线性回归 残差⊥X (不变量)', lr.residualInvariant(Xb, yNum) < 1e-6, `max=${lr.residualInvariant(Xb, yNum).toExponential(2)}`);

    // 逻辑回归梯度检验（线性可分数据）
    const Xlr = [[0, 0], [1, 0], [0, 1], [1, 1], [5, 5], [6, 5], [5, 6], [6, 6]];
    const ylr = [-1, -1, -1, -1, 1, 1, 1, 1];
    const logit = new LogisticRegression(0.1, 3000, 0).fit(Xlr, ylr);
    const relErr = logit.gradientCheck(Xlr, ylr);
    check('逻辑回归 梯度检验≈1e-7 (不变量)', relErr < 1e-5, `relErr=${relErr.toExponential(2)}`);
    check('逻辑回归 训练可分', logit.predict(Xlr).every((p, i) => (p === 1 ? 1 : -1) === ylr[i]));

    // SVM 线性可分
    const Xlin = [[0, 0], [1, 0], [0, 1], [1, 1], [5, 5], [6, 5], [5, 6], [6, 6]];
    const ylin = [-1, -1, -1, -1, 1, 1, 1, 1];
    const svm = new SVM(0.001, 20000).fit(Xlin, ylin);
    check('SVM 线性可分 100% (不变量)', svm.predict(Xlin).every((p, i) => p === ylin[i]));

    const knn = new kNN(3).fit(Xb, yb);
    const accK = knn.predict(Xb).filter((p, i) => p === yb[i]).length / Xb.length;
    check('kNN 训练集高准确率', accK > 0.9, `acc=${(accK * 100).toFixed(1)}%`);

    const nb = new NaiveBayes().fit(Xb, yb);
    const proba = nb.predictProba(Xb);
    check('朴素贝叶斯 后验和=1 (不变量)', proba.every(p => Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9));

    const dt = new DecisionTree(6, 2).fit(Xb, yb);
    const accT = dt.predict(Xb).filter((p, i) => p === yb[i]).length / Xb.length;
    check('决策树 训练拟合', accT > 0.9, `acc=${(accT * 100).toFixed(1)}%`);
  }

  // ---------- L5 无监督 ----------
  console.log('[L5] 无监督学习');
  {
    const { X: Xc, y: yc } = DataGen.blobs(150, 3, 2);
    const km = new KMeans(3, 50).fit(Xc);
    check('KMeans 惯性单调不增 (不变量)', km.inertiaMonotone(), `steps=${km.inertiaHist.length}`);

    const pca = new PCA(2).fit(Xc);
    check('PCA 解释比∈[0,1]', pca.explainedRatio() > 0 && pca.explainedRatio() <= 1 + 1e-9, `ratio=${pca.explainedRatio().toFixed(4)}`);
    const proj = pca.transform(Xc);
    check('PCA 投影维度正确', proj[0].length === 2);

    const gmm = new GMM(3, 40).fit(Xc);
    check('GMM 责任度每行和=1 (不变量)', gmm.responsibilitySumInvariant());
    check('GMM 对数似然单调不减 (不变量)', gmm.llMonotone(), `Δll=${(gmm.llHist[gmm.llHist.length - 1] - gmm.llHist[0]).toFixed(2)}`);
  }

  // ---------- L6 表示学习：MLP ----------
  console.log('[L6] 深度学习 MLP');
  {
    const { X, y } = DataGen.xor();
    const mlp = new MLP([2, 5, 1], 0.3).train(X, y, 20000);
    check('MLP 损失单调下降 (不变量)', mlp.lossDecreasing(), `loss:${mlp.lossHist[0].toFixed(3)}→${mlp.lossHist[mlp.lossHist.length - 1].toFixed(3)}`);
    const preds = mlp.predict(X);
    const acc = preds.filter((p, i) => p === y[i]).length / 4;
    check('MLP 学会 XOR', acc === 1, `preds=${JSON.stringify(preds)}`);

    // AutoGrad 梯度检验：f = sigmoid(a*b + c)
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const a = new AutoGrad(2), b = new AutoGrad(3), c = new AutoGrad(-1);
    const out = AutoGrad.sigmoid(AutoGrad.add(AutoGrad.mul(a, b), c));
    AutoGrad.backward(out);
    const s = sig(2 * 3 - 1); const dout = s * (1 - s);
    check('AutoGrad df/da', Math.abs(a.grad - dout * 3) < 1e-12);
    check('AutoGrad df/db', Math.abs(b.grad - dout * 2) < 1e-12);
    check('AutoGrad df/dc', Math.abs(c.grad - dout) < 1e-12);
  }

  // ---------- L7 Transformer ----------
  console.log('[L7] Transformer');
  {
    const Q = [[1, 0, 1], [0, 1, 0], [1, 1, 0]];
    const K = [[1, 0, 0], [0, 1, 1], [1, 1, 1]];
    const V = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    check('注意力权重每行和=1 (不变量)', Transformer.attentionSumInvariant(Q, K, V));
    const mha = Transformer.multiHeadAttention(Q, K, V, 3);
    check('多头注意力 输出维度', mha.length === 3 && mha[0].length === 3, `[${mha.length}×${mha[0].length}]`);
    const pe = Transformer.positionalEncoding(4, 4);
    check('位置编码 形状正确', pe.length === 4 && pe[0].length === 4);
  }

  // ---------- L10 智能体 ----------
  console.log('[L10] 智能体规划内核');
  {
    const agent = new Agent()
      .register('search', async (a) => `results-for:${a.q}`)
      .register('compute', async () => ({ score: 0.95 }))
      .register('summarize', async (a) => `summary:${a.goal}`)
      .register('report', async () => 'report-ok');
    const res = await agent.run('锻造世界顶级 AI');
    check('Agent 生成微指令', res.microOps.length > 0, `microOps=${res.microOps.length}`);
    check('Agent 完成追踪', res.trace.length >= 1);
  }

  // ---------- 汇总 ----------
  console.log(`\n=== 自检结果：${pass} 通过 / ${fail} 失败 ===`);
  if (fail > 0) { console.log('失败项：', fails.join(', ')); process.exit(1); }
  else { console.log('🟢 全绿 (ALL GREEN) — APEX 引擎数学不变量全部通过'); process.exit(0); }
})();
