"""
Generate the final defense web deck with the Guizang Swiss PPT template.

The generated artifact is a single-file HTML slide deck. It intentionally keeps
the runtime from guizang-ppt-skill and replaces only the slide section.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_TITLE = "工业轴承设备剩余寿命预测系统的实现"
TOTAL_SLIDES = 11
SKILL_TEMPLATE = Path(".agents/skills/guizang-ppt-skill/assets/template-swiss.html")
OUTPUT_PATH = Path("docx/final/web-ppt/index.html")


CUSTOM_CSS = """
  /* Project-specific Swiss additions for a course-defense deck. */
  .course-cover{
    flex:1;display:grid;grid-template-columns:1.25fr .75fr;gap:4vw;align-items:stretch
  }
  .cover-title{
    display:flex;flex-direction:column;justify-content:center;border-top:2px solid var(--accent);
    padding-top:4vh
  }
  .cover-info{
    background:var(--grey-1);padding:3vh 2vw;display:flex;flex-direction:column;
    justify-content:space-between;border-top:2px solid var(--ink)
  }
  .info-list{display:grid;gap:1.2vh;border-top:1px solid var(--border-subtle);padding-top:2vh}
  .info-list div{display:grid;grid-template-columns:6em 1fr;gap:1vw;font-family:var(--sans),var(--sans-zh);font-size:max(15px,.94vw);line-height:1.45}
  .info-list b{font-family:var(--mono);font-size:14px;letter-spacing:.12em;color:var(--text-helper);font-weight:600}
  .evidence-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1vw;margin-top:3vh}
  .evidence-cell{background:var(--grey-1);padding:1.6vh 1.1vw;border-top:2px solid var(--ink)}
  .evidence-cell.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .evidence-cell .nb{font-family:var(--sans);font-size:min(3.7vw,6.5vh);font-weight:200;line-height:.9;letter-spacing:-.035em}
  .evidence-cell .label{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.9vw);line-height:1.4;margin-top:1vh;opacity:.82}
  .dataset-facts{display:grid;grid-template-columns:repeat(2,1fr);gap:1vw;margin-top:2.4vh}
  .fact-card{background:var(--paper);border-top:2px solid currentColor;padding:1.5vh 1vw}
  .fact-card .num{font-family:var(--sans);font-size:min(3.2vw,5.4vh);font-weight:200;line-height:.95;letter-spacing:-.035em}
  .fact-card .txt{font-family:var(--sans),var(--sans-zh);font-size:max(13px,.82vw);line-height:1.45;color:var(--text-secondary);margin-top:.8vh}
  .feature-board{display:grid;grid-template-columns:1.05fr 1fr;gap:2vw;flex:1;margin-top:4vh}
  .feature-list{display:grid;grid-template-columns:repeat(2,1fr);gap:1vw}
  .feature-card{background:var(--grey-1);padding:1.8vh 1.2vw;border-top:2px solid var(--ink)}
  .feature-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .feature-card .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(17px,1.25vw);font-weight:500;line-height:1.2}
  .feature-card .desc{font-family:var(--sans),var(--sans-zh);font-size:max(13px,.82vw);line-height:1.45;opacity:.78;margin-top:1vh}
  .feature-compact{background:var(--grey-1);padding:2.2vh 1.5vw;border-top:2px solid var(--accent);display:flex;flex-direction:column}
  .feature-compact .mono{font-family:var(--mono);font-size:max(12px,.72vw);line-height:1.65;color:var(--text-secondary);word-break:break-word}
  .rul-flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1vw;align-items:stretch;margin-top:4vh;flex:1}
  .rul-step{background:var(--grey-1);padding:2vh 1.1vw;display:flex;flex-direction:column;min-height:0;border-top:2px solid var(--ink)}
  .rul-step.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .rul-step .num{font-family:var(--mono);font-size:14px;letter-spacing:.12em;opacity:.65;margin-bottom:auto}
  .rul-step .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(17px,1.22vw);line-height:1.15;font-weight:500;letter-spacing:-.015em;margin-top:3vh}
  .rul-step .txt{font-family:var(--sans),var(--sans-zh);font-size:max(13px,.8vw);line-height:1.45;opacity:.75;margin-top:1vh}
  .spec-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.4vw;align-items:stretch;margin-top:4vh;flex:1}
  .spec-card{background:var(--grey-1);padding:2.4vh 1.5vw;display:flex;flex-direction:column;border-top:2px solid var(--ink)}
  .spec-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .spec-card .big{font-family:var(--sans);font-size:min(4.7vw,8vh);font-weight:200;line-height:.95;letter-spacing:-.04em}
  .spec-card .label{font-family:var(--mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase;opacity:.65;margin-bottom:2vh}
  .spec-card .desc{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.88vw);line-height:1.55;opacity:.78;margin-top:auto}
  .mini-ledger{display:grid;gap:0;margin-top:2vh;border-top:1px solid var(--border-subtle)}
  .mini-ledger div{display:grid;grid-template-columns:8em 1fr;gap:1vw;padding:1.05vh 0;border-bottom:1px solid var(--border-subtle)}
  .mini-ledger b{font-family:var(--mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase}
  .mini-ledger span{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.86vw);line-height:1.45;color:var(--text-secondary)}
  .closing-list{display:flex;flex-direction:column;gap:0}
  .closing-item{display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.2vh 0;border-top:1px solid var(--border-subtle)}
  .closing-item:last-child{border-bottom:2px solid var(--accent)}
  .closing-item .n{font-family:var(--sans);font-weight:200;font-size:min(3.8vw,6.8vh);line-height:.9;color:var(--text-primary)}
  .closing-item.accent .n,.closing-item.accent h3{color:var(--accent)}
  .closing-item h3{font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(18px,1.45vw);line-height:1.2;letter-spacing:-.015em;margin-bottom:.8vh}
  .closing-item p{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.88vw);line-height:1.5;color:var(--text-secondary)}
"""


SLIDES = f"""
<section class="slide" data-layout="S01" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">中国科学技术大学软件学院 · 软件工程课程</div>
      <div class="r">结题答辩 · 01 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="course-cover" data-anim="up">
      <div class="cover-title">
        <div class="t-meta" style="color:var(--accent);margin-bottom:2vh">项目名称</div>
        <h1 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.6vw,10vh);line-height:1.06;letter-spacing:-.025em">工业轴承设备<br/>剩余寿命预测系统<br/>的实现</h1>
        <p class="lead" style="font-weight:300;max-width:52ch;margin-top:3vh;color:var(--text-secondary)">围绕真实轴承退化数据，完成数据接入、特征分析、RUL 建模、论文复现和课程交付。</p>
      </div>
      <div class="cover-info">
        <div>
          <div class="t-meta">答辩信息</div>
          <div class="info-list">
            <div><b>课程</b><span>软件工程</span></div>
            <div><b>指导</b><span>zjf</span></div>
            <div><b>成员</b><span>zyj、cyj、zdh、zy</span></div>
            <div><b>日期</b><span>2026 年 6 月</span></div>
          </div>
        </div>
        <p class="body-sm" style="color:var(--text-secondary)">主线始终是剩余寿命预测，也就是估计轴承还能运行多久。</p>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">汇报主线</div>
      <div class="r">02 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">先说明问题和数据，再说明实现、复现和验收</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.8vw,10vh);line-height:.98;letter-spacing:-.035em">每一部分都对应仓库中的实现或输出文件。</h2>
    </div>
    <div class="sub-grid-3-2" data-anim="up">
      <article class="sub-card accent"><div class="nb-corner">01</div><div class="ttl">问题定义</div><p class="desc">为什么预测性维护更关注 RUL，而不是只看当前状态。</p></article>
      <article class="sub-card"><div class="nb-corner">02</div><div class="ttl">两个数据集</div><p class="desc">XJTU-SY 与 PHM2012 的采样组织、工况和 RUL 语义差异。</p></article>
      <article class="sub-card"><div class="nb-corner">03</div><div class="ttl">19 维特征</div><p class="desc">从振动快照提取时域、频域特征，再组织成特征序列。</p></article>
      <article class="sub-card"><div class="nb-corner">04</div><div class="ttl">工程实现</div><p class="desc">loader、feature、labeling、model、training、evaluation 分层，核心类可追溯。</p></article>
      <article class="sub-card"><div class="nb-corner">05</div><div class="ttl">两篇复现</div><p class="desc">CNN-LSTM-AM 与 xLSTM-Transformer 均完成真实数据小规模训练。</p></article>
      <article class="sub-card ink"><div class="nb-corner">06</div><div class="ttl">验收材料</div><p class="desc">31 个测试、8 个 notebook、输出 CSV 和课程文档可追溯。</p></article>
    </div>
  </div>
</section>

<section class="slide split" data-layout="S03" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="justify-content:space-between;position:relative;overflow:hidden">
        <div class="chrome-min" style="position:relative;z-index:1">
          <div class="l">问题定义</div>
          <div class="r">03 / {TOTAL_SLIDES:02d}</div>
        </div>
        <h2 data-anim="manifesto" style="position:relative;z-index:1;font-family:var(--sans),var(--sans-zh);font-size:min(7.2vw,13vh);line-height:1;letter-spacing:-.025em;font-weight:200;color:#fff">为什么是 RUL？</h2>
        <div class="t-meta" style="position:relative;z-index:1;color:rgba(255,255,255,.72)">Remaining Useful Life</div>
      </div>
      <div class="half b-grey" style="justify-content:center">
        <div data-anim="rules" style="display:flex;flex-direction:column;gap:3vh">
          <p class="lead" style="font-weight:300;color:var(--text-primary);max-width:38ch">预测性维护真正关心的是“还能运行多久”，这样才能提前安排维修窗口并降低停机风险。</p>
          <div class="mini-ledger">
            <div><b>输入</b><span>轴承运行过程中的水平、垂直振动快照。</span></div>
            <div><b>目标</b><span>按寿命顺序生成剩余寿命标签，输出连续 RUL 数值。</span></div>
            <div><b>边界</b><span>本轮答辩主线是 RUL 回归和预测性维护，不展开离散类别任务。</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">数据集理解</div>
      <div class="r">04 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">两个数据集都是真实 run-to-failure 数据，但时间组织并不相同</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.7vw,10vh);line-height:.98;letter-spacing:-.035em">不能只把它们当成 CSV 文件。</h2>
    </div>
    <div class="duo-compare" data-anim="up" style="margin-top:5vh">
      <div class="col accent">
        <div class="col-tag"><span class="num">A</span> XJTU-SY</div>
        <div class="col-ttl">三工况十五轴承</div>
        <p class="col-desc">每个轴承沿寿命周期持续采样，水平、垂直两个振动通道用于构建退化特征序列。</p>
        <div class="dataset-facts">
          <div class="fact-card"><div class="num">25.6 kHz</div><div class="txt">采样频率</div></div>
          <div class="fact-card"><div class="num">32768</div><div class="txt">每个快照点数</div></div>
          <div class="fact-card"><div class="num">1.28 s</div><div class="txt">单个快照覆盖时长</div></div>
          <div class="fact-card"><div class="num">1 min</div><div class="txt">相邻快照间隔</div></div>
        </div>
      </div>
      <div class="vrule"></div>
      <div class="col">
        <div class="col-tag"><span class="num">B</span> PHM2012 / FEMTO</div>
        <div class="col-ttl">三工况竞赛基准</div>
        <p class="col-desc">文件结构、快照长度和时间间隔不同；温度文件保留对齐，本次复现主线使用振动通道。</p>
        <div class="dataset-facts">
          <div class="fact-card"><div class="num">25.6 kHz</div><div class="txt">采样频率</div></div>
          <div class="fact-card"><div class="num">2560</div><div class="txt">每个加速度文件点数</div></div>
          <div class="fact-card"><div class="num">0.1 s</div><div class="txt">单个快照覆盖时长</div></div>
          <div class="fact-card"><div class="num">约 10 s</div><div class="txt">相邻快照间隔；Test_set 终止 RUL 只对官方条目叠加</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S16" data-animate="field-notes">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">特征分析</div>
      <div class="r">05 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">14 个时域特征 + 5 个频域特征</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.9vw,10.2vh);line-height:.98;letter-spacing:-.035em">19 维特征用于描述退化过程。</h2>
    </div>
    <div class="feature-board" data-anim="up">
      <div class="feature-list">
        <article class="feature-card accent"><div class="ttl">RMS / 谱能量</div><p class="desc">反映振动强度，退化加剧时通常更容易形成上升趋势。</p></article>
        <article class="feature-card"><div class="ttl">峰值 / 峭度</div><p class="desc">对冲击性变化敏感，适合捕捉退化后期的波动增强。</p></article>
        <article class="feature-card"><div class="ttl">均值 / 方差</div><p class="desc">描述总体稳定性和波动强度，便于不同轴承横向对齐。</p></article>
        <article class="feature-card"><div class="ttl">主频 / 谱质心</div><p class="desc">观察能量集中频段和频谱重心变化，补充时域信息。</p></article>
      </div>
      <div class="feature-compact">
        <div class="t-meta" style="color:var(--accent);margin-bottom:2vh">实现位置</div>
        <div class="mono">强度: rms, peak, spectrum_energy<br>冲击: kurtosis, crest_factor, impulse_factor, margin_factor<br>稳定性: mean, variance, standard_deviation<br>频谱: dominant_frequency, spectral_centroid, spectral_rms_frequency, spectral_entropy</div>
        <p class="body-sm" style="margin-top:auto;color:var(--text-secondary)">这些快照级特征再按时间顺序拼成长度 5 或 10 的 feature sequence，作为深度模型输入。</p>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S17" data-animate="system-diagram">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">系统架构</div>
      <div class="r">06 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;justify-content:space-between;gap:4vw;align-items:flex-start">
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5vw,8.8vh);line-height:.98;letter-spacing:-.035em">模块边界<br/>能追溯到代码。</h2>
      <p class="lead" style="font-weight:300;color:var(--text-secondary);max-width:40ch">核心代码位于 USTC.SSE.BearingPrediction。同一个 FeatureSequenceRulLabeler 同时服务两篇论文复现。</p>
    </div>
    <div class="rul-flow" data-anim="up">
      <div class="rul-step"><div class="num">01</div><div class="ttl">XJTULoader / PHM2012Loader</div><div class="txt">解析目录、采样率、工况和通道。</div></div>
      <div class="rul-step"><div class="num">02</div><div class="ttl">SignalFeature<br/>Extractor</div><div class="txt">生成 19 维时频域特征。</div></div>
      <div class="rul-step"><div class="num">03</div><div class="ttl">FeatureSequence<br/>RulLabeler</div><div class="txt">构造特征序列和 RUL 标签。</div></div>
      <div class="rul-step accent"><div class="num">04</div><div class="ttl">Models</div><div class="txt">CNN-LSTM-AM、XLSTM-Transformer 与 baseline。</div></div>
      <div class="rul-step"><div class="num">05</div><div class="ttl">ExperimentTracker</div><div class="txt">保存 config、history、metrics、predictions。</div></div>
      <div class="rul-step"><div class="num">06</div><div class="ttl">Examples / Docs</div><div class="txt">notebook、复现说明和课程交付文档。</div></div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RUL 建模流程</div>
      <div class="r">07 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:grid;grid-template-columns:auto 1fr;gap:3vw;align-items:start">
      <div style="font-family:var(--sans);font-weight:200;font-size:min(7vw,12vh);line-height:.9;letter-spacing:-.04em;color:var(--accent)">RUL</div>
      <div>
        <div class="t-meta">从振动快照到剩余寿命曲线</div>
        <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(4.9vw,8.5vh);line-height:.98;letter-spacing:-.035em;margin-top:1.2vh">同一组特征和标签可以替换不同模型。</h2>
      </div>
    </div>
    <div class="timeline-h" data-anim="up">
      <div class="tl-row">
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">01</span><span class="name">读取快照</span><span class="desc">水平/垂直振动</span></div></div>
        <div class="th-node down"><span class="dot"></span><div class="label"><span class="yr">02</span><span class="name">抽取特征</span><span class="desc">19 维时频域统计</span></div></div>
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">03</span><span class="name">生成标签</span><span class="desc">按寿命顺序计算 RUL</span></div></div>
        <div class="th-node down accent"><span class="dot"></span><div class="label"><span class="yr">04</span><span class="name">训练模型</span><span class="desc">baseline 与论文结构</span></div></div>
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">05</span><span class="name">输出结果</span><span class="desc">metrics / predictions / history</span></div></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S21" data-animate="tech-spec">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">论文复现一</div>
      <div class="r">08 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">Huang 等 2024：CNN-LSTM-AM</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.4vw,9.4vh);line-height:.98;letter-spacing:-.035em">复现重点是结构、数据链路和论文 Score。</h2>
    </div>
    <div class="spec-grid" data-anim="up">
      <article class="spec-card accent"><div class="label">实现</div><div class="big">CNN<br>LSTM<br>AM</div><p class="desc">`CNNLSTMAttention(use_attention=True)`；不带 attention 的同类模型作为 CNN-LSTM baseline。</p></article>
      <article class="spec-card"><div class="label">真实训练</div><div class="big">8<br>epoch</div><p class="desc">XJTU-SY 与 PHM2012 各抽样 48 个真实快照；输出 `history.csv`、`predictions.csv`、`attention_weights.csv`。</p></article>
      <article class="spec-card"><div class="label">验收结果</div><div class="big">RMSE</div><p class="desc">XJTU-SY AM: 406.30；PHM2012 AM: 651.04。课程验收优先证明真实数据链路、结构和指标可复跑。</p></article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S21" data-animate="tech-spec">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">论文复现二</div>
      <div class="r">09 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">Jiang 等 2026：xLSTM-Transformer</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.4vw,9.4vh);line-height:.98;letter-spacing:-.035em">复现重点是跨工况划分和统一特征管线。</h2>
    </div>
    <div class="spec-grid" data-anim="up">
      <article class="spec-card"><div class="label">数据划分</div><div class="big">6</div><p class="desc">XJTU-SY 三工况按 1/2/4/5 训练、3 测试；PHM2012 按 1/2 训练、3 测试。</p></article>
      <article class="spec-card accent"><div class="label">模型对比</div><div class="big">18</div><p class="desc">XLSTM-Transformer、Feature-Transformer、LSTM-Transformer 在两个数据集六个工况输出 18 行对比结果。</p></article>
      <article class="spec-card"><div class="label">复现边界</div><div class="big">8<br>epoch</div><p class="desc">论文未开源作者代码；本项目是结构复现 + 项目特征管线适配，完整论文级训练受时间和算力限制。</p></article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S20" data-animate="stacked-ledger">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">测试验收</div>
      <div class="r">10 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="ledger" style="display:flex;flex-direction:column;flex:1;justify-content:center">
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,17vw) 1fr 8vw;gap:2vw;align-items:center;padding:2.2vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(7.8vw,12vh);line-height:.92;letter-spacing:-.04em">31</div>
        <div class="ledger-label"><div class="t-meta">自动化测试</div><div class="lead" style="font-weight:300">`uv run --extra dev pytest -q` 通过，覆盖 loader、指标、训练流程、论文 workflow 和 notebook smoke。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">pytest</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,17vw) 1fr 8vw;gap:2vw;align-items:center;padding:2.2vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(7.8vw,12vh);line-height:.92;letter-spacing:-.04em;color:var(--accent)">8</div>
        <div class="ledger-label"><div class="t-meta">notebook 示例</div><div class="lead" style="font-weight:300">examples 下 notebook 统一执行 smoke test，不只是检查文件存在。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">ipynb</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,17vw) 1fr 8vw;gap:2vw;align-items:center;padding:2.2vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(7.8vw,12vh);line-height:.92;letter-spacing:-.04em">2</div>
        <div class="ledger-label"><div class="t-meta">论文复现</div><div class="lead" style="font-weight:300">两篇复现均读取 data/external 真实数据，并落盘 `comparison_metrics.csv`、`metrics.json`、`history.csv`、`predictions.csv`。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">CSV</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,17vw) 1fr 8vw;gap:2vw;align-items:center;padding:2.2vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(7.8vw,12vh);line-height:.92;letter-spacing:-.04em">3</div>
        <div class="ledger-label"><div class="t-meta">指标口径</div><div class="lead" style="font-weight:300">普通误差、Huang 原版 Score、PHM/NASA 惩罚 Score 分开输出，避免复现指标混用。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">score</div>
      </div>
    </div>
  </div>
</section>

<section class="slide split" data-layout="S10" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min" style="margin-bottom:0">
          <div class="l">总结</div>
          <div class="r">11 / {TOTAL_SLIDES:02d}</div>
        </div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh">
          <div class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.18em;margin-bottom:1.6vh">结题结论</div>
          <h2 style="font-family:var(--sans),var(--sans-zh);font-size:min(6.6vw,11.5vh);line-height:1;letter-spacing:-.025em;font-weight:200;color:#fff">完成了 RUL 预测的工程闭环。</h2>
          <div style="font-family:var(--sans),var(--sans-zh);font-size:max(14px,1vw);line-height:1.6;color:rgba(255,255,255,.84);font-weight:300;max-width:38ch;margin-top:1.4vh">真实数据接入、特征分析、模型训练、论文复现、自动化测试和课程文档均可追溯。</div>
        </div>
        <div data-anim="signature" style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh">
          <div class="t-meta" style="color:rgba(255,255,255,.62)">谢谢老师和同学</div>
          <div class="t-meta" style="color:rgba(255,255,255,.62)">欢迎提问</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min">
          <div class="l">不足与展望</div>
          <div class="r">后续工作</div>
        </div>
        <div data-anim="rules" class="closing-list">
          <div class="closing-item"><div class="n">01</div><div><h3>复现规模</h3><p>当前是小样本真实训练验收，后续可扩大样本、epoch 和多随机种子统计。</p></div></div>
          <div class="closing-item"><div class="n">02</div><div><h3>泛化评估</h3><p>后续可补充更严格的跨工况留一验证和长时间训练预算。</p></div></div>
          <div class="closing-item accent"><div class="n">03</div><div><h3>扩展能力</h3><p>后续可把相关概率分析和可视化报告做得更完整，本次答辩不作为主线展开。</p></div></div>
        </div>
        <div data-anim="foot" class="t-meta" style="color:var(--text-helper);text-align:right">主线：剩余寿命预测与预测性维护</div>
      </div>
    </div>
  </div>
</section>
"""


def build_deck() -> str:
    """
    Build the final web PPT by replacing the template slide sample block.

    Returns
    -------
    str
        Complete HTML slide deck.
    """
    template = SKILL_TEMPLATE.read_text(encoding="utf-8")
    start = template.index("<!-- ============================================================\n     SLIDES 插入区")
    end = template.index("</div>\n\n<div id=\"nav\"></div>", start)
    replacement = f"<!-- Generated final defense slides. -->\n{SLIDES.strip()}\n"
    html = template[:start] + replacement + template[end:]
    template_title = "<title>" + "[\u5fc5\u586b] 替换为 PPT 标题 · Deck Title</title>"
    html = html.replace(template_title, f"<title>{PROJECT_TITLE} · 结题答辩</title>")
    html = html.replace("</style>", CUSTOM_CSS.rstrip() + "\n</style>", 1)
    return html


def main() -> None:
    """
    Generate the final deck file.
    """
    if not SKILL_TEMPLATE.exists():
        raise FileNotFoundError(
            "Guizang PPT skill template was not found. Expected "
            f"{SKILL_TEMPLATE}. Clone op7418/guizang-ppt-skill.git into .agents/skills first."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_deck(), encoding="utf-8")
    print(f"generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
