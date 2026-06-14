"""
Generate the final defense web deck with the Guizang Swiss PPT template.

The generated artifact is a single-file HTML slide deck. It intentionally keeps
the runtime from guizang-ppt-skill and replaces only the slide section.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_TITLE = "工业轴承设备剩余寿命预测系统的实现"
TOTAL_SLIDES = 12
SKILL_TEMPLATE = Path(".agents/skills/guizang-ppt-skill/assets/template-swiss.html")
OUTPUT_PATH = Path("docx/final/web-ppt/index.html")


CUSTOM_CSS = """
  /* Project-specific Swiss additions. Keep them small and template-compatible. */
  .rul-flow{
    display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1vw;
    align-items:stretch;margin-top:4vh;flex:1
  }
  .rul-step{
    background:var(--grey-1);padding:2.2vh 1.2vw;display:flex;flex-direction:column;
    min-height:0;border-top:2px solid var(--ink)
  }
  .rul-step.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .rul-step .num{font-family:var(--mono);font-size:14px;letter-spacing:.16em;opacity:.65;margin-bottom:auto}
  .rul-step .ttl{font-family:var(--sans),var(--sans-zh);font-size:max(18px,1.4vw);line-height:1.15;font-weight:500;letter-spacing:-.015em;margin-top:3vh}
  .rul-step .txt{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.86vw);line-height:1.55;opacity:.75;margin-top:1.2vh}
  .compact-table{display:grid;gap:0;margin-top:3vh;border-top:1px solid var(--border-subtle)}
  .compact-row{display:grid;grid-template-columns:1.1fr 1fr 1.2fr 1.25fr;gap:1.2vw;align-items:start;padding:1.55vh 0;border-bottom:1px solid var(--border-subtle)}
  .compact-row.head{font-family:var(--mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase;color:var(--text-helper)}
  .compact-row:not(.head){font-family:var(--sans),var(--sans-zh);font-size:max(15px,.95vw);line-height:1.45;color:var(--text-primary)}
  .compact-row .muted{color:var(--text-secondary)}
  .spec-grid{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:2vw;align-items:stretch;margin-top:4vh;flex:1}
  .spec-card{background:var(--grey-1);padding:2.4vh 1.6vw;display:flex;flex-direction:column;border-top:2px solid var(--ink)}
  .spec-card.accent{background:var(--accent);color:var(--accent-on);border-color:var(--accent)}
  .spec-card .big{font-family:var(--sans);font-size:min(5.2vw,9vh);font-weight:200;line-height:.95;letter-spacing:-.04em}
  .spec-card .label{font-family:var(--mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;opacity:.65;margin-bottom:2vh}
  .spec-card .desc{font-family:var(--sans),var(--sans-zh);font-size:max(15px,.94vw);line-height:1.55;opacity:.78;margin-top:auto}
  .mini-ledger{display:grid;gap:0;margin-top:2vh;border-top:1px solid var(--border-subtle)}
  .mini-ledger div{display:grid;grid-template-columns:8em 1fr;gap:1vw;padding:1.2vh 0;border-bottom:1px solid var(--border-subtle)}
  .mini-ledger b{font-family:var(--mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase}
  .mini-ledger span{font-family:var(--sans),var(--sans-zh);font-size:max(14px,.9vw);line-height:1.5;color:var(--text-secondary)}
  .closing-list{display:flex;flex-direction:column;gap:0}
  .closing-item{display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.4vh 0;border-top:1px solid var(--border-subtle)}
  .closing-item:last-child{border-bottom:2px solid var(--accent)}
  .closing-item .n{font-family:var(--sans);font-weight:200;font-size:min(4vw,7.2vh);line-height:.9;color:var(--text-primary)}
  .closing-item.accent .n,.closing-item.accent h3{color:var(--accent)}
  .closing-item h3{font-family:var(--sans),var(--sans-zh);font-weight:400;font-size:max(18px,1.55vw);line-height:1.2;letter-spacing:-.015em;margin-bottom:1vh}
  .closing-item p{font-family:var(--sans),var(--sans-zh);font-size:max(15px,.9vw);line-height:1.55;color:var(--text-secondary)}
"""


SLIDES = f"""
<section class="slide accent" data-layout="SWISS-COVER-ASCII" data-animate="hero">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    <div class="chrome-min">
      <div class="l">USTC SSE · SOFTWARE ENGINEERING FINAL DEFENSE</div>
      <div class="r">26.06 · 01 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.6vh">
      <div data-anim="kicker" class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em">REMAINING USEFUL LIFE · PREDICTIVE MAINTENANCE</div>
      <h1 data-anim="title" style="align-self:center;font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(8.2vw,16vh);line-height:1.02;letter-spacing:-.025em;color:#fff">{PROJECT_TITLE}</h1>
      <div data-anim="bottom" style="display:grid;grid-template-rows:auto auto;gap:1.6vh;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh">
        <div data-anim="lead" class="lead" style="max-width:58ch;color:rgba(255,255,255,.86);font-weight:300">从轴承振动退化数据出发，完成数据接入、特征提取、RUL 建模、论文复现和课程交付闭环。</div>
        <div style="display:flex;justify-content:space-between;align-items:end">
          <div class="t-meta" style="color:rgba(255,255,255,.6)">中国科学技术大学 软件学院 · 软件工程课程</div>
          <div class="t-meta" style="color:rgba(255,255,255,.6)">→ arrow keys / swipe</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S01" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">AGENDA · WHAT WILL BE PROVED</div>
      <div class="r">02 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div class="sub-grid-3-2" data-anim="up" style="margin-top:0">
      <article class="sub-card accent"><div class="nb-corner">01</div><div class="ttl">项目定位</div><p class="desc">围绕剩余寿命预测展开，服务预测性维护场景。</p></article>
      <article class="sub-card"><div class="nb-corner">02</div><div class="ttl">数据认识</div><p class="desc">XJTU-SY 与 PHM2012 的工况、通道、采样节奏与退化过程。</p></article>
      <article class="sub-card"><div class="nb-corner">03</div><div class="ttl">特征体系</div><p class="desc">时域、频域、序列窗口和健康指标的工程化组织。</p></article>
      <article class="sub-card"><div class="nb-corner">04</div><div class="ttl">系统实现</div><p class="desc">loader、preprocess、feature、labeling、model、evaluation 分层。</p></article>
      <article class="sub-card"><div class="nb-corner">05</div><div class="ttl">论文复现</div><p class="desc">CNN-LSTM-AM 与 xLSTM-Transformer 两条真实训练工作流。</p></article>
      <article class="sub-card ink"><div class="nb-corner">06</div><div class="ttl">验收结论</div><p class="desc">notebook、单元测试、指标落盘和文档交付可追溯。</p></article>
    </div>
  </div>
</section>

<section class="slide split" data-layout="S03" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="position:relative;z-index:1">
          <div class="l">POSITIONING</div>
          <div class="r">03 / {TOTAL_SLIDES:02d}</div>
        </div>
        <h2 data-anim="manifesto" style="position:relative;z-index:1;font-family:var(--sans),var(--sans-zh);font-size:min(8vw,14vh);line-height:.96;letter-spacing:-.025em;font-weight:200;color:#fff">不是只跑一个模型。</h2>
        <div class="t-meta" style="position:relative;z-index:1;color:rgba(255,255,255,.62)">RUL AS ENGINEERING SYSTEM</div>
      </div>
      <div class="half b-grey" style="justify-content:center">
        <div data-anim="rules" style="display:flex;flex-direction:column;gap:3vh">
          <p class="lead" style="font-weight:300;color:var(--text-primary);max-width:38ch">本项目把真实轴承振动数据转化为可训练、可评估、可展示的剩余寿命预测流程。</p>
          <div class="mini-ledger">
            <div><b>OBJECT</b><span>轴承退化过程中的 RUL 曲线，而不是离散类别判断。</span></div>
            <div><b>OUTPUT</b><span>预测剩余寿命、误差指标、趋势曲线、论文复现实验表。</span></div>
            <div><b>VALUE</b><span>让预测性维护从数据读取到答辩展示都有可复查证据。</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">DATASETS · TWO RUN-TO-FAILURE BENCHMARKS</div>
      <div class="r">04 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">数据不是背景材料，而是模型设计的边界条件</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(6.4vw,11vh);line-height:.96;letter-spacing:-.035em">XJTU-SY / PHM2012</h2>
    </div>
    <div class="duo-compare" data-anim="up">
      <div class="col accent">
        <div class="col-tag"><span class="num">A</span> XJTU-SY</div>
        <div class="col-ttl">三工况 · 长序列</div>
        <p class="col-desc">每个轴承都有水平、垂直两个振动通道，采样文件沿寿命周期递增，适合观察退化阶段和跨工况泛化。</p>
        <ul class="col-list">
          <li>论文复现按同工况内 1/2/4/5 训练，3 测试。</li>
          <li>高采样频率片段转成特征序列，避免直接把原始长信号塞入模型。</li>
          <li>更适合展示健康指标随时间上升、RUL 随时间下降的关系。</li>
        </ul>
      </div>
      <div class="vrule"></div>
      <div class="col">
        <div class="col-tag"><span class="num">B</span> PHM2012 / FEMTO</div>
        <div class="col-ttl">三工况 · 竞赛基准</div>
        <p class="col-desc">同样是 run-to-failure 振动数据，但工况、命名、寿命长度和采样节奏不同，适合检验 loader 与评估指标的泛化性。</p>
        <ul class="col-list">
          <li>论文复现按每工况 1/2 训练，3 测试。</li>
          <li>保留水平和垂直通道，统一抽取时域与频域统计特征。</li>
          <li>PHM/NASA 类惩罚分数用于补充衡量预测偏早或偏晚的风险。</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">FEATURES · FROM VIBRATION TO HEALTH STATE</div>
      <div class="r">05 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">特征分析贯穿数据理解、建模输入和结果解释</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(6vw,10.5vh);line-height:.96;letter-spacing:-.035em">振动信号如何变成 RUL 输入</h2>
    </div>
    <div class="sub-grid-3-2" data-anim="up">
      <article class="sub-card accent"><div class="nb-corner">01</div><div class="ttl">RMS / 能量</div><p class="desc">对振动幅值增强最敏感，常作为退化上升趋势的主线索。</p></article>
      <article class="sub-card"><div class="nb-corner">02</div><div class="ttl">峰值 / 峭度</div><p class="desc">反映冲击性增强，对局部异常和后期加速退化更敏感。</p></article>
      <article class="sub-card"><div class="nb-corner">03</div><div class="ttl">均值 / 方差</div><p class="desc">描述信号总体稳定性和波动强度，便于不同轴承横向对齐。</p></article>
      <article class="sub-card"><div class="nb-corner">04</div><div class="ttl">频域能量</div><p class="desc">通过 FFT 捕捉频谱结构变化，弥补纯时域指标的盲区。</p></article>
      <article class="sub-card"><div class="nb-corner">05</div><div class="ttl">序列窗口</div><p class="desc">将连续时间片组织为 feature sequence，让模型学习退化路径而不是孤立点。</p></article>
      <article class="sub-card ink"><div class="nb-corner">06</div><div class="ttl">RUL 标签</div><p class="desc">按寿命文件顺序生成剩余寿命目标，确保时间方向与预测任务一致。</p></article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S17" data-animate="system-diagram">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">ARCHITECTURE · PIPELINE AS THE PRODUCT</div>
      <div class="r">06 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;justify-content:space-between;gap:4vw;align-items:flex-start">
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.4vw,9.4vh);line-height:.98;letter-spacing:-.035em">项目不是 notebook 拼接。</h2>
      <p class="lead" style="font-weight:300;color:var(--text-secondary);max-width:40ch">核心代码放在 USTC.SSE.BearingPrediction 命名空间下，训练、评估、可视化和示例都通过统一 API 进入。</p>
    </div>
    <div class="rul-flow" data-anim="up">
      <div class="rul-step"><div class="num">01</div><div class="ttl">Dataset Loader</div><div class="txt">XJTU-SY 与 PHM2012 文件解析、元数据组织、通道读取。</div></div>
      <div class="rul-step"><div class="num">02</div><div class="ttl">Preprocess</div><div class="txt">异常处理、标准化、窗口化，避免模型层直接接触原始文件。</div></div>
      <div class="rul-step"><div class="num">03</div><div class="ttl">Feature</div><div class="txt">时域、频域、健康指标和 feature sequence 统一生成。</div></div>
      <div class="rul-step accent"><div class="num">04</div><div class="ttl">RUL Model</div><div class="txt">传统回归、CNN-LSTM-AM、xLSTM-Transformer 可替换训练。</div></div>
      <div class="rul-step"><div class="num">05</div><div class="ttl">Evaluation</div><div class="txt">RMSE、R2、Huang Score、PHM/NASA score 和方向性指标。</div></div>
      <div class="rul-step"><div class="num">06</div><div class="ttl">Visualization</div><div class="txt">RUL 曲线、健康趋势、复现表和课程交付图表。</div></div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">PAPER 01 · CNN-LSTM-AM</div>
      <div class="r">07 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:grid;grid-template-columns:auto 1fr;gap:3vw;align-items:start">
      <div style="font-family:var(--sans);font-weight:200;font-size:min(7vw,12vh);line-height:.9;letter-spacing:-.04em;color:var(--accent)">01</div>
      <div>
        <div class="t-meta">Huang et al. 2024 · attention based RUL workflow</div>
        <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5vw,8.6vh);line-height:.98;letter-spacing:-.035em;margin-top:1.2vh">用注意力把“哪些特征、哪些时刻更重要”显式化。</h2>
      </div>
    </div>
    <div class="timeline-h" data-anim="up">
      <div class="tl-row">
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">STEP 01</span><span class="name">特征序列</span><span class="desc">统计特征按时间排序</span></div></div>
        <div class="th-node down"><span class="dot"></span><div class="label"><span class="yr">STEP 02</span><span class="name">CNN</span><span class="desc">局部模式提取</span></div></div>
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">STEP 03</span><span class="name">LSTM</span><span class="desc">退化路径记忆</span></div></div>
        <div class="th-node down accent"><span class="dot"></span><div class="label"><span class="yr">STEP 04</span><span class="name">Attention</span><span class="desc">时序权重聚合</span></div></div>
        <div class="th-node up"><span class="dot"></span><div class="label"><span class="yr">STEP 05</span><span class="name">RUL</span><span class="desc">输出寿命回归值</span></div></div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S13" data-animate="three-forces">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">PAPER 02 · XLSTM-TRANSFORMER</div>
      <div class="r">08 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">Jiang et al. 2026 · sequence model reproduction</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(6vw,10.5vh);line-height:.96;letter-spacing:-.035em">第二篇复现验证框架可承载更复杂结构。</h2>
    </div>
    <div class="stack-row" data-anim="up" style="margin-top:5vh">
      <article class="stack-block b-ink"><div class="layer-nb">BASELINE</div><div class="layer-ttl">Feature Sequence</div><p class="layer-desc">统一使用 sequence length = 10 的特征序列输入，保证模型比较落在同一数据口径。</p><div class="layer-tag">same data contract</div></article>
      <article class="stack-block b-accent"><div class="layer-nb">MODEL</div><div class="layer-ttl">xLSTM + Transformer</div><p class="layer-desc">xLSTM 强化长期记忆，Transformer 负责全局依赖建模，输出 RUL 回归值。</p><div class="layer-tag">deep sequence learner</div></article>
      <article class="stack-block b-grey"><div class="layer-nb">CHECK</div><div class="layer-ttl">Two Dataset Split</div><p class="layer-desc">XJTU-SY 和 PHM2012 均按论文风格训练/测试轴承划分，指标统一落 CSV。</p><div class="layer-tag">paper-aligned workflow</div></article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S21" data-animate="tech-spec">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">METRICS · RUL SCORE SYSTEM</div>
      <div class="r">09 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="line" style="display:flex;flex-direction:column;gap:1.2vh">
      <div class="t-meta">指标体系区分普通误差、论文原版 Score 和非对称惩罚</div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.8vw,10vh);line-height:.96;letter-spacing:-.035em">答辩时不能把 Score 混成一个数。</h2>
    </div>
    <div class="spec-grid" data-anim="up">
      <article class="spec-card accent"><div class="label">ERROR</div><div class="big">RMSE</div><p class="desc">MAE、RMSE、Normalized RMSE、SMAPE、R2 用于衡量回归误差和论文表格对齐。</p></article>
      <article class="spec-card"><div class="label">ORIGINAL</div><div class="big">Huang</div><p class="desc">Huang RUL Score 严格按论文公式，以相对误差 Er_i 分段累计，不做额外缩放。</p></article>
      <article class="spec-card"><div class="label">RISK</div><div class="big">Penalty</div><p class="desc">PHM/NASA score、Over/Under Prediction Rate、Within Tolerance Rate 描述偏早或偏晚风险。</p></article>
    </div>
  </div>
</section>

<section class="slide" data-layout="S20" data-animate="stacked-ledger">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">VERIFICATION · WHAT REALLY RAN</div>
      <div class="r">10 / {TOTAL_SLIDES:02d}</div>
    </div>
    <div data-anim="ledger" style="display:flex;flex-direction:column;flex:1;justify-content:center">
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,18vw) 1fr 7vw;gap:2vw;align-items:center;padding:2.3vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(8.2vw,13vh);line-height:.92;letter-spacing:-.04em">31</div>
        <div class="ledger-label"><div class="t-meta">PYTEST</div><div class="lead" style="font-weight:300">全量单元与集成测试通过，覆盖 loader、指标、paper workflow 和 notebook smoke。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">PASS</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,18vw) 1fr 7vw;gap:2vw;align-items:center;padding:2.3vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(8.2vw,13vh);line-height:.92;letter-spacing:-.04em;color:var(--accent)">8</div>
        <div class="ledger-label"><div class="t-meta">NOTEBOOKS</div><div class="lead" style="font-weight:300">examples 统一使用 notebook，并纳入可执行测试，避免只摆展示文件。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">RUN</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,18vw) 1fr 7vw;gap:2vw;align-items:center;padding:2.3vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(8.2vw,13vh);line-height:.92;letter-spacing:-.04em">2</div>
        <div class="ledger-label"><div class="t-meta">PAPERS</div><div class="lead" style="font-weight:300">两篇 RUL 论文完成同规格复现，均读取 data/external 真实数据并输出对比表。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">CSV</div>
      </div>
      <div class="ledger-row" style="display:grid;grid-template-columns:minmax(12vw,18vw) 1fr 7vw;gap:2vw;align-items:center;padding:2.3vh 0;border-bottom:1px solid var(--border-subtle)">
        <div class="ledger-num" style="font-family:var(--sans);font-weight:200;font-size:min(8.2vw,13vh);line-height:.92;letter-spacing:-.04em">8</div>
        <div class="ledger-label"><div class="t-meta">EPOCHS</div><div class="lead" style="font-weight:300">真实训练验收使用 8 epoch；notebook 验收使用 1 epoch smoke 保持可运行性。</div></div>
        <div class="ledger-icon t-meta" style="text-align:right">REAL</div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div data-anim="line" style="display:flex;flex-direction:column;gap:2.2vh">
      <div style="height:2px;background:var(--accent);width:100%"></div>
      <div class="chrome-min tight" style="margin-bottom:0">
        <div class="l">CLOSURE · LIMITS AND NEXT WORK</div>
        <div class="r">11 / {TOTAL_SLIDES:02d}</div>
      </div>
      <h2 style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.8vw,10vh);line-height:.96;letter-spacing:-.035em">边界讲清楚，成果才可信。</h2>
    </div>
    <div data-anim="up" style="display:grid;grid-template-columns:repeat(4,1fr);gap:1.4vw;flex:1;margin-top:5vh">
      <article class="sub-card"><div class="t-meta">— 01 / DATA</div><div class="ttl" style="margin-top:3vh">小规模复现</div><p class="desc">当前强调框架可复现与真实训练，不承诺作者全量 epoch 的完全同表数值。</p></article>
      <article class="sub-card accent"><div class="t-meta" style="color:rgba(255,255,255,.75)">— 02 / MODEL</div><div class="ttl" style="margin-top:3vh">结构适配</div><p class="desc">论文结构按项目特征管线适配，保留训练、指标和对比逻辑。</p></article>
      <article class="sub-card"><div class="t-meta">— 03 / VALIDATION</div><div class="ttl" style="margin-top:3vh">随机性控制</div><p class="desc">后续可扩展多随机种子、多工况留一验证和更长训练预算。</p></article>
      <article class="sub-card ink"><div class="t-meta" style="color:rgba(255,255,255,.65)">— 04 / ROADMAP</div><div class="ttl" style="margin-top:3vh">系统增强</div><p class="desc">继续补全生存概率分析、可视化报告和更完整的线上推理入口。</p></article>
    </div>
  </div>
</section>

<section class="slide split" data-layout="SWISS-CLOSING-ASCII" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="margin-bottom:0;position:relative;z-index:1">
          <div class="l">12 / {TOTAL_SLIDES:02d}</div>
          <div class="r">CLOSING</div>
        </div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em;margin-bottom:1.6vh">FINAL CLAIM</div>
          <h2 style="font-family:var(--sans),var(--sans-zh);font-size:min(7.6vw,13vh);line-height:.96;letter-spacing:-.025em;font-weight:200;color:#fff">数据能读。<br/>模型能训。<br/>结果能验。</h2>
          <div style="font-family:var(--sans),var(--sans-zh);font-size:max(14px,1vw);line-height:1.6;color:rgba(255,255,255,.82);font-weight:300;max-width:36ch;margin-top:1.4vh">这就是本项目作为课程工程交付的核心价值。</div>
        </div>
        <div data-anim="signature" style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.62)">THANK YOU</div>
          <div class="t-meta" style="color:rgba(255,255,255,.62)">2026.06</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min">
          <div class="l">TAKEAWAYS</div>
          <div class="r">03 POINTS</div>
        </div>
        <div data-anim="rules" class="closing-list">
          <div class="closing-item"><div class="n">01</div><div><h3>围绕开题任务</h3><p>主线是工业轴承设备剩余寿命预测，所有演示围绕 RUL、特征、训练和评估展开。</p></div></div>
          <div class="closing-item"><div class="n">02</div><div><h3>理解真实数据</h3><p>区分两个数据集的工况、通道、寿命序列和论文划分，不把数据当成黑盒。</p></div></div>
          <div class="closing-item accent"><div class="n">03</div><div><h3>完成工程闭环</h3><p>源码、notebook、测试、论文复现、课程文档和汇报材料均可在仓库中追溯。</p></div></div>
        </div>
        <div data-anim="foot" class="t-meta" style="color:var(--text-helper);text-align:right">END OF FINAL DEFENSE</div>
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
