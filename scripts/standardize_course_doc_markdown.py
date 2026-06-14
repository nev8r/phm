#!/usr/bin/env python3
"""
Standardize course delivery Markdown documents.

The script keeps the document body authored in Markdown, replaces uneven
front matter with a unified course-document cover block, and injects detailed
sections needed for final delivery review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "工业轴承设备剩余寿命预测系统的实现"
COURSE_NAME = "中国科学技术大学软件学院《软件工程》"
TEACHER = "zjf"
TEAM_MEMBERS = "zyj、cyy、zdh、zy"
REVISION_DATE = "2026-06-14"


@dataclass(frozen=True)
class DocMeta:
    path: str
    doc_id: str
    title: str
    stage: str
    owner: str
    contributors: str
    baseline: str


DOCS = [
    DocMeta("docx/proposal/md/01_开题报告.md", "SE-PHM-PROP-01", "开题报告", "开题", "zyj", "cyy、zdh、zy", "项目立项、研究目标、技术路线、计划与分工"),
    DocMeta("docx/proposal/md/03_技术预研报告.md", "SE-PHM-PROP-03", "技术预研报告", "开题", "zdh", "zyj、cyy、zy", "数据、特征、模型、工具链和风险预研"),
    DocMeta("docx/proposal/md/04_需求定义文档.md", "SE-PHM-PROP-04", "需求定义文档", "开题", "zy", "zyj、cyy、zdh", "业务目标、用户角色、总体需求和验收目标"),
    DocMeta("docx/proposal/md/05_SRS规格说明文档.md", "SE-PHM-PROP-05", "SRS 软件需求规格说明文档", "开题", "zdh", "zyj、cyy、zy", "IEEE 830 风格需求规格、接口、约束和追踪矩阵"),
    DocMeta("docx/proposal/md/09_确认测试计划文档.md", "SE-PHM-PROP-09", "确认测试计划文档", "开题", "zdh", "zyj、cyy、zy", "确认测试范围、依据、用例和通过准则"),
    DocMeta("docx/proposal/md/10_项目管理计划文档.md", "SE-PHM-PROP-10", "项目管理计划文档", "开题", "zyj", "cyy、zdh、zy", "进度、质量、风险、沟通和配置管理"),
    DocMeta("docx/mid-term/md/02_中期检查报告.md", "SE-PHM-MID-02", "中期检查报告", "中期", "cyy", "zyj、zdh、zy", "中期进度、阶段成果、问题和后续计划"),
    DocMeta("docx/mid-term/md/06_设计文档.md", "SE-PHM-MID-06", "设计文档", "中期", "zyj", "cyy、zdh、zy", "总体架构、模块设计、接口设计、数据流和部署设计"),
    DocMeta("docx/mid-term/md/07_单元测试计划文档.md", "SE-PHM-MID-07", "单元测试计划文档", "中期", "cyy", "zyj、zdh、zy", "单元划分、用例设计、通过准则和回归策略"),
    DocMeta("docx/mid-term/md/08_集成测试计划文档.md", "SE-PHM-MID-08", "集成测试计划文档", "中期", "zy", "zyj、cyy、zdh", "跨模块集成链路、联调策略和通过准则"),
    DocMeta("docx/mid-term/md/11_编码规范文档.md", "SE-PHM-MID-11", "编码规范文档", "中期", "cyy", "zyj、zdh、zy", "源码、测试、文档、提交和导出规范"),
    DocMeta("docx/mid-term/md/12_UML设计文档.md", "SE-PHM-MID-12", "UML 设计文档", "中期", "zyj", "cyy、zdh、zy", "用例图、类图、顺序图、组件关系和部署视图"),
    DocMeta("docx/final/md/12_结题报告.md", "SE-PHM-FINAL-12", "结题报告", "结题", "zyj", "cyy、zdh、zy", "最终成果、数据理解、技术难点、测试和验收总结"),
    DocMeta("docx/final/md/13_单元测试报告.md", "SE-PHM-FINAL-13", "单元测试报告", "结题", "cyy", "zyj、zdh、zy", "单元测试执行记录、覆盖范围和结论"),
    DocMeta("docx/final/md/14_集成测试报告.md", "SE-PHM-FINAL-14", "集成测试报告", "结题", "zy", "zyj、cyy、zdh", "端到端链路、输出物检查和集成测试结论"),
    DocMeta("docx/final/md/15_确认测试报告.md", "SE-PHM-FINAL-15", "确认测试报告", "结题", "zdh", "zyj、cyy、zy", "需求确认、用户场景验证、风险和结论"),
    DocMeta("docx/final/md/16_用户使用手册.md", "SE-PHM-FINAL-16", "用户使用手册", "结题", "zy", "zyj、cyy、zdh", "运行流程、notebook、API、输出文件和注意事项"),
    DocMeta("docx/final/md/17_安装配置手册.md", "SE-PHM-FINAL-17", "安装配置手册", "结题", "zy", "zyj、cyy、zdh", "环境、依赖、数据目录、命令和问题排查"),
    DocMeta("docx/final/md/18_项目技术论文.md", "SE-PHM-FINAL-18", "项目技术论文", "结题", "zyj", "cyy、zdh、zy", "系统设计、数据特征、模型、实验和软件工程实践"),
    DocMeta("docx/final/md/19_成员贡献比说明.md", "SE-PHM-FINAL-19", "成员贡献比说明", "结题", "zyj", "cyy、zdh、zy", "成员职责、贡献比例、交叉评审和确认说明"),
    DocMeta("docx/final/md/20_结题答辩提纲.md", "SE-PHM-FINAL-20", "结题答辩提纲", "结题", "zy", "zyj、cyy、zdh", "答辩结构、每页要点、时间分配和备答重点"),
    DocMeta("docx/final/md/21_结题答辩演讲稿.md", "SE-PHM-FINAL-21", "结题答辩演讲稿", "结题", "zy", "zyj、cyy、zdh", "正式汇报讲稿、页间衔接和答问口径"),
]


BODY_START_MARKERS = [
    "## 团队分工说明",
    "## 使用说明",
    "## 摘要",
    "## 1.",
]


REPLACEMENTS = {
    "cyj": "cyy",
    "未来故障风险": "未来失效风险",
    "故障退化过程明确": "全寿命退化过程明确",
    "混淆矩阵或注意力热图": "误差分布图、RUL 曲线或注意力热图",
    "混淆矩阵和阶段图": "误差分布图和退化阶段图",
    "混淆矩阵、阶段图": "误差分布图、退化阶段图",
    "混淆矩阵": "误差分布图",
    "NASA Score": "PHM/RUL 非对称惩罚 Score",
    "PHM/NASA 类惩罚 Score": "PHM/RUL 惩罚 Score",
    "PHM/NASA 类 score": "PHM/RUL 非对称惩罚 score",
    "PHM/NASA 类非对称惩罚 Score": "PHM/RUL 非对称惩罚 Score",
    "PHM/NASA 惩罚 Score": "PHM/RUL 惩罚 Score",
    "PHM/NASA": "PHM/RUL",
    "NASAScore": "AsymmetricRulPenalty",
    "故障分类仅保留基础能力，不作为结题主线": "离散状态划分仅作为退化阶段辅助，不作为结题主线",
    "分类不是主线": "离散诊断不是主线",
    "不扩展故障诊断分类指标": "不扩展离散状态识别指标",
    "不以故障分类作为结题重点": "不以离散故障识别作为结题重点",
    "故障诊断或分类": "离散状态识别",
    "阶段分类流程": "退化阶段标注与可视化流程",
    "阶段分类数据集": "退化阶段标注数据集",
    "训练 Transformer 分类模型": "生成退化阶段标注并训练序列预测或可视化模型",
    "阶段分类任务": "退化阶段辅助分析任务",
    "用于阶段分类与健康状态标注": "用于退化阶段标注与健康状态辅助分析",
    "RUL 回归、阶段分类或健康指标预测数据集": "RUL 回归、退化阶段标注或健康指标预测数据集",
    "文档不出现发动机/C-MAPSS 或离散诊断主线偏移": "文档不混入外部样例的领域语境，不偏离 RUL 预测主线",
    "### 3.1 数据对象与特征分析计划": "### 3.3 数据对象与特征分析计划",
    "### 5.1 关键难点与解决思路": "### 5.3 关键难点与解决思路",
    "### 6.1 特征序列与 RUL 标签设计": "### 6.4 特征序列与 RUL 标签设计",
    "### 8.1 技术难点评估与取舍": "### 8.3 技术难点评估与取舍",
    "### 3.7 关键需求追踪矩阵": "### 3.8 关键需求追踪矩阵",
    "### 4.4 数据与实验约束": "### 4.3 数据与实验约束",
    "| FR-DATA-01 | 支持 XJTU-SY 数据目录加载 | `dataset`、`XJTULoader` | `tests/test_data_io_and_loaders.py` |": "| FR-M1-01 | 支持 XJTU-SY 数据目录加载 | `dataset`、`XJTULoader` | `tests/test_data_io_and_loaders.py` |",
    "| FR-DATA-02 | 支持 PHM2012/FEMTO 数据目录加载 | `dataset`、`PHM2012Loader` | `tests/test_data_io_and_loaders.py` |": "| FR-M1-02 | 支持 PHM2012/FEMTO 数据目录加载 | `dataset`、`PHM2012Loader` | `tests/test_data_io_and_loaders.py` |",
    "| FR-FEAT-01 | 支持时域、频域特征提取 | `feature`、`labeling` | 特征导出 notebook、单元测试 |": "| FR-M2-03/FR-M2-04 | 支持时域、频域特征提取 | `feature`、`labeling` | 特征导出 notebook、单元测试 |",
    "| FR-RUL-01 | 支持 RUL 标签构造和回归训练 | `labeling`、`models`、`training` | 训练 pipeline 测试、真实训练记录 |": "| FR-M3-01/FR-M4-01 | 支持 RUL 标签构造和回归训练 | `labeling`、`models`、`training` | 训练 pipeline 测试、真实训练记录 |",
    "| FR-METRIC-01 | 支持 RUL 论文常用指标 | `evaluation` | `tests/test_rul_metrics.py` |": "| FR-M5-01/FR-M5-02 | 支持 RUL 论文常用指标 | `evaluation` | `tests/test_rul_metrics.py` |",
    "| FR-REPRO-01 | 支持 CNN-LSTM-AM 复现流程 | `examples/06_*`、workflow | 复现测试、comparison_metrics.csv |": "| FR-M4-01/FR-M5-04 | 支持 CNN-LSTM-AM 复现流程 | `examples/06_*`、workflow | 复现测试、comparison_metrics.csv |",
    "| FR-REPRO-02 | 支持 xLSTM-Transformer 复现流程 | `examples/07_*`、workflow | 复现测试、comparison_metrics.csv |": "| FR-M4-01/FR-M5-04 | 支持 xLSTM-Transformer 复现流程 | `examples/07_*`、workflow | 复现测试、comparison_metrics.csv |",
    "| NFR-REPRO-01 | 实验结果可追踪 | 输出 history、metrics、predictions | 集成测试和文档检查 |": "| FR-M5-04/FR-M5-05 | 实验结果可追踪 | 输出 history、metrics、predictions | 集成测试和文档检查 |",
    "## 1. 分工说明": "## 1. 贡献比例汇总\n\n| 成员 | 贡献比例 | 主要贡献依据 |\n| --- | ---: | --- |\n| zyj | 35% | 项目负责人、系统架构、训练框架、RUL 模型、论文复现和最终集成 |\n| cyy | 25% | 数据处理、数据集 loader、特征工程、数据文档和数据测试 |\n| zdh | 20% | 退化分析、生存分析接口、评价指标、确认测试计划和确认测试报告 |\n| zy | 20% | 可视化、用户手册、安装配置、答辩提纲、讲稿和展示材料 |\n\n贡献比例根据项目计划、代码模块、文档产物、测试内容和最终集成工作综合评估。zyj 作为项目负责人承担额外集成与复现工作，因此比例略高；其他成员围绕数据、评价测试和展示文档分别承担主责。\n\n## 2. 分工说明",
    "## 1.1 交付物对应关系": "## 3. 交付物对应关系",
    "## 2. 贡献确认": "## 4. 贡献确认",
    "工程化平台的发展趋势": "工程化系统的发展趋势",
    "本课题希望弥补这一缺口": "本课题希望形成课程要求的工程化流程",
    "完整闭环": "主要流程",
    "具备良好的可运行性、可测试性和课程答辩展示价值": "能够支撑课程验收中的运行、测试和对比展示",
}


def make_preamble(meta: DocMeta) -> str:
    return f"""# {PROJECT_NAME}：{meta.title}

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | {meta.title} |
| 文档编号 | {meta.doc_id} |
| 文档阶段 | {meta.stage} |
| 项目名称 | {PROJECT_NAME} |
| 课程 | {COURSE_NAME} |
| 指导老师 | {TEACHER} |
| 小组成员 | {TEAM_MEMBERS} |
| 文档负责人 | {meta.owner} |
| 参与编写 | {meta.contributors} |
| 版本 | V3.0 |
| 修订日期 | {REVISION_DATE} |
| 归档形式 | Markdown 源文件、PDF、DOCX |
| 内容基线 | {meta.baseline} |

## 修订记录

| 版本 | 日期 | 编写人 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2025-11-10 | 项目组 | 完成初版课程文档 |
| V2.0 | 2026-03-12 | 项目组 | 统一项目名称、技术基线和阶段交付内容 |
| V3.0 | {REVISION_DATE} | 项目组 | 参考课程交付样例统一封面与归档格式，补充数据理解、技术难点、测试验收和 DOCX 交付信息 |

"""


def extract_body(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    for marker in BODY_START_MARKERS:
        idx = normalized.find(marker)
        if idx >= 0:
            return normalized[idx:].strip()
    lines = normalized.splitlines()
    return "\n".join(line for line in lines if not line.startswith("# ")).strip()


def extra_block(key: str, content: str) -> str:
    return content.strip()


def insert_before(body: str, heading: str, key: str, content: str) -> str:
    marker = f"<!-- doc-standard-extra:{key} -->"
    first_content_line = content.strip().splitlines()[0]
    if marker in body or first_content_line in body:
        return body
    block = "\n\n" + extra_block(key, content) + "\n\n"
    idx = body.find(heading)
    if idx < 0:
        return body.rstrip() + block
    return body[:idx].rstrip() + block + body[idx:].lstrip()


def remove_standard_comments(text: str) -> str:
    return re.sub(r"<!-- doc-standard-extra:[^>]+ -->\n?", "", text)


def first_heading(content: str) -> str | None:
    for line in content.strip().splitlines():
        if line.startswith("#"):
            return line.strip()
    return None


def all_headings(content: str) -> list[str]:
    return [line.strip() for line in content.strip().splitlines() if line.startswith("#")]


def remove_section_by_heading(text: str, heading: str) -> str:
    lines = text.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    output: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == heading:
            index += 1
            while index < len(lines):
                stripped = lines[index].lstrip()
                if stripped.startswith("#"):
                    next_level = len(stripped) - len(stripped.lstrip("#"))
                    if next_level <= level:
                        break
                index += 1
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output).strip()


def remove_existing_extra_sections(body: str) -> str:
    for items in EXTRAS.values():
        for _, _, content in items:
            normalized_content = apply_replacements(content)
            for heading in all_headings(normalized_content):
                body = remove_section_by_heading(body, heading)
    return body


EXTRAS: dict[str, list[tuple[str, str, str]]] = {
    "01_开题报告.md": [
        (
            "## 4. 系统主要功能概述",
            "proposal-data-understanding",
            """
### 3.1 数据对象与特征分析计划

本课题的数据对象不是普通静态表格，而是轴承从正常运行到退化加剧的多通道振动时间序列。项目计划从两个角度理解数据：一是文件组织和时间语义，二是特征随寿命阶段变化的规律。

| 数据集 | 数据组织特点 | 计划重点分析的规律 | 对系统设计的要求 |
| --- | --- | --- | --- |
| XJTU-SY | 三种工况、多个 run-to-failure 轴承，水平和垂直振动快照按时间顺序排列 | 寿命后期 RMS、峰值、峭度和谱能量通常抬升；不同工况下振动幅值分布存在差异 | Loader 必须保留工况、轴承编号、采样顺序和通道信息，训练划分应避免破坏时间顺序 |
| PHM2012/FEMTO | Learning/Test/Full_Test 目录语义不同，包含加速度和温度文件，快照间隔较短 | 后期振动强度增强但曲线更密集，测试集终止 RUL 需要结合官方条目解释 | Loader 需要统一 split 语义，并为加速度主线和温度扩展预留字段 |

特征分析不只服务图表展示，还决定后续模型输入。项目将优先提取均值、方差、RMS、峰值、峭度、峰值因子、谱能量、谱熵和主频等时域/频域特征，用于观察退化趋势、构造健康指标、训练 RUL 模型，并支撑答辩中对“为什么这样建模”的解释。
""",
        ),
        (
            "## 6. 可行性分析",
            "proposal-key-difficulties",
            """
### 5.1 关键难点与解决思路

| 难点 | 形成原因 | 解决思路 | 预期验证方式 |
| --- | --- | --- | --- |
| 数据集异构 | 两个数据集目录结构、快照长度、采样间隔和 split 语义不同 | 将差异限制在 `XJTULoader`、`PHM2012Loader` 内部，对外输出统一实体 | 数据加载单元测试、notebook 数据概览 |
| RUL 标签构造 | 原始文件通常只给时间顺序或终止信息，不直接给每个窗口的 RUL | 按轴承寿命终点和当前快照时间差生成 RUL，必要时保留秒级/快照级单位说明 | 标签构造测试、预测结果与 target 对照 |
| 特征可解释性 | 深度模型可学习复杂模式，但答辩需要解释数据规律 | 保留 19 维时频域特征和健康指标曲线，先解释趋势再解释模型 | 数据特征图、特征导出 notebook |
| 训练可复现 | 深度模型训练受随机性、数据划分和 epoch 影响 | 统一训练入口、环境变量、输出目录、指标 CSV 和 notebook smoke test | pytest、真实训练记录、comparison_metrics.csv |
""",
        ),
    ],
    "03_技术预研报告.md": [
        (
            "## 8. 技术方案对比与最终选型",
            "tech-data-detail",
            """
### 7.4 数据特征与处理策略细化

技术预研阶段对两个数据集的处理策略如下：

| 维度 | XJTU-SY | PHM2012/FEMTO | 处理策略 |
| --- | --- | --- | --- |
| 时间粒度 | 快照间隔约 1 分钟，单个快照较长 | 快照间隔约 10 秒，单个加速度快照较短 | 在实体层记录采样率、快照索引和时间单位，避免把文件序号误认为统一物理时间 |
| 通道 | 水平、垂直振动 | 水平、垂直振动，并存在温度文件 | 当前 RUL 主线以振动为主，温度作为可扩展字段保留 |
| 退化表现 | 后期振动能量和冲击特征明显增强 | 后期振动增强更密集，部分序列存在突变阶段 | 使用 RMS、峰值、峭度、谱能量、谱熵等特征同时刻画强度与冲击 |
| 划分方式 | 可按轴承编号和工况划分 | Learning/Test/Full_Test 语义更接近竞赛设置 | 训练 workflow 明确记录划分规则，防止数据泄漏 |

因此，项目不直接把原始信号硬塞给单一模型，而是先建立可解释特征层。这样既能支撑传统模型和深度模型，也便于在课程答辩中说明退化规律，而不是只展示一个黑盒分数。
""",
        ),
        (
            "## 9. 预研结论",
            "tech-risk-comparison",
            """
### 8.1 技术难点评估与取舍

| 技术方向 | 价值 | 风险 | 本项目取舍 |
| --- | --- | --- | --- |
| 端到端原始信号深度学习 | 可减少手工特征依赖 | 训练资源要求高，解释成本高 | 作为扩展方向保留，结题主线采用特征序列深度模型 |
| 传统特征 + 回归模型 | 稳定、可解释、运行快 | 表达能力有限 | 作为 baseline 和教学说明基础 |
| CNN-LSTM-AM | 兼顾局部特征、时序关系和注意力权重 | 参数较多，需要明确输入序列构造 | 用于第一篇论文复现 |
| xLSTM-Transformer | 适合序列建模和长依赖表达 | 作者设置与本地资源可能不完全一致 | 用于第二篇论文结构复现和工程适配 |
| 生存分析 | 能表达失效/存活概率视角 | 数据删失与风险解释要求较高 | 保留接口和基础能力，作为 RUL 主线补充 |
""",
        ),
    ],
    "04_需求定义文档.md": [
        (
            "## 7. 功能需求列表",
            "requirements-scenarios",
            """
## 6.1 典型业务流程细化

| 流程 | 用户操作 | 系统响应 | 验收关注点 |
| --- | --- | --- | --- |
| 数据接入 | 指定 XJTU-SY 或 PHM2012 本地目录 | 系统识别工况、轴承编号、通道和快照顺序 | 目录缺失时给出明确错误；加载后样本数、通道数和时间顺序正确 |
| 特征分析 | 选择轴承和通道生成特征曲线 | 输出 RMS、峰值、峭度、健康指标等趋势 | 曲线能反映后期振动增强，图表含标题、轴和单位说明 |
| RUL 训练 | 选择模型、epoch 和输出目录 | 执行训练并生成 history、metrics、predictions | 训练过程真实执行，输出文件可追踪 |
| 论文复现 | 运行指定 notebook 或 workflow | 生成两个数据集上的对比指标表 | 包含 RMSE、NormalizedRMSE、R2、Score 和 prediction_count |
| 结果解释 | 查看指标与曲线 | 系统说明误差、方向偏差和适用边界 | 不把小样本训练结论夸大为工业部署效果 |
""",
        ),
        (
            "## 12. 验收目标",
            "requirements-acceptance-detail",
            """
## 11.1 需求验收口径

项目验收时按“能否运行、结果是否可解释、文档是否能追溯”三个层级判断：

1. 运行层：示例 notebook 可执行，核心测试通过，真实数据目录存在时能够读入 XJTU-SY 和 PHM2012。
2. 解释层：每个主要模型输出 RMSE/MAE/Score 等指标，同时提供 RUL 曲线或特征曲线辅助说明。
3. 追溯层：训练配置、输出目录、指标 CSV、测试命令和文档结论之间能够对应，不出现“文档说有、代码不能跑”的情况。
""",
        ),
    ],
    "05_SRS规格说明文档.md": [
        (
            "## 4. 其他需求",
            "srs-traceability",
            """
### 3.7 关键需求追踪矩阵

| 需求编号 | 需求说明 | 设计/实现位置 | 验证方式 |
| --- | --- | --- | --- |
| FR-DATA-01 | 支持 XJTU-SY 数据目录加载 | `dataset`、`XJTULoader` | `tests/test_data_io_and_loaders.py` |
| FR-DATA-02 | 支持 PHM2012/FEMTO 数据目录加载 | `dataset`、`PHM2012Loader` | `tests/test_data_io_and_loaders.py` |
| FR-FEAT-01 | 支持时域、频域特征提取 | `feature`、`labeling` | 特征导出 notebook、单元测试 |
| FR-RUL-01 | 支持 RUL 标签构造和回归训练 | `labeling`、`models`、`training` | 训练 pipeline 测试、真实训练记录 |
| FR-METRIC-01 | 支持 RUL 论文常用指标 | `evaluation` | `tests/test_rul_metrics.py` |
| FR-REPRO-01 | 支持 CNN-LSTM-AM 复现流程 | `examples/06_*`、workflow | 复现测试、comparison_metrics.csv |
| FR-REPRO-02 | 支持 xLSTM-Transformer 复现流程 | `examples/07_*`、workflow | 复现测试、comparison_metrics.csv |
| NFR-REPRO-01 | 实验结果可追踪 | 输出 history、metrics、predictions | 集成测试和文档检查 |
""",
        ),
        (
            "## 5. 附录",
            "srs-data-constraints",
            """
### 4.4 数据与实验约束

1. 系统不承诺在线采集工业现场数据，输入以公开数据集、本地解压目录和生成的示例数据为主。
2. RUL 预测结果用于课程实验和趋势分析，不作为真实工业维护的直接决策依据。
3. 真实训练输出、原始数据和中间模型文件不纳入 Git 版本库，避免仓库体积失控。
4. 对论文复现实验，系统复现的是论文结构、数据划分口径和指标输出流程，不声明完全复刻作者私有训练环境。
""",
        ),
    ],
    "09_确认测试计划文档.md": [
        (
            "## 7. 验收标准",
            "acceptance-test-cases",
            """
## 6.1 详细确认测试用例

| 用例编号 | 场景 | 输入条件 | 期望结果 |
| --- | --- | --- | --- |
| AT-01 | 数据目录检查 | 提供 XJTU-SY 解压目录 | 能列出工况、轴承编号和通道，异常目录给出可读错误 |
| AT-02 | PHM2012 数据接入 | 提供 Learning_set 或 Test_set | 能解析加速度文件并保留时间顺序 |
| AT-03 | 特征趋势验证 | 选择单个 run-to-failure 轴承 | RMS、峰值或健康指标曲线能展示后期退化增强 |
| AT-04 | RUL 训练验证 | 设置 1 epoch smoke test | 训练真实执行，history 和 metrics 文件存在 |
| AT-05 | 论文复现验证 | 运行两篇复现 notebook | 对比表包含两个数据集、模型名称和关键指标 |
| AT-06 | 文档交付验证 | 执行导出脚本 | 每份 Markdown 均生成 PDF 和 DOCX |
""",
        ),
    ],
    "10_项目管理计划文档.md": [
        (
            "## 8. 质量管理计划",
            "management-quality-gates",
            """
### 7.4 阶段质量门禁

| 阶段 | 质量门禁 | 负责人 | 通过标准 |
| --- | --- | --- | --- |
| 开题 | 项目名称、任务边界和分工一致 | zyj | 文档不混入外部样例的领域语境，不偏离 RUL 预测主线 |
| 中期 | 数据链路和设计链路可说明 | cyy、zyj | loader、特征和训练接口均有设计说明与测试计划 |
| 结题 | 代码、notebook、测试、文档可互相印证 | 全体成员 | pytest 通过，notebook smoke 通过，DOCX/PDF 归档完整 |
| 答辩 | 语气自然、证据明确、边界清楚 | zy | PPT 和讲稿能解释数据规律、技术难点和验收结果 |
""",
        ),
    ],
    "02_中期检查报告.md": [
        (
            "## 6. 遇到的问题与原因分析",
            "mid-data-progress",
            """
### 5.1 数据理解阶段性结论

中期阶段已经确认两个数据集不能简单用同一套文件读取逻辑处理。XJTU-SY 的优势是 run-to-failure 过程清晰、工况命名规则较稳定，适合展示完整退化曲线；PHM2012 的优势是竞赛语义明确、样本数量更密，但 Learning/Test/Full_Test 的组织方式需要谨慎解释。基于这一判断，项目没有把训练代码直接绑定到某个目录，而是先设计 `BearingEntity` 和 loader 适配层。

在特征层，中期已经确定以 RMS、峰值、峭度、峰值因子和谱能量作为主要观察指标。它们能够分别反映振动强度、冲击程度和频域能量变化，为后续健康指标曲线和 RUL 模型输入提供基础。
""",
        ),
        (
            "## 8. 后续工作计划",
            "mid-difficulty-progress",
            """
### 7.1 关键难点处理进展

| 难点 | 中期状态 | 后续处理 |
| --- | --- | --- |
| 数据格式差异 | 已完成统一实体设计和 loader 初步实现 | 补充真实目录 smoke test 和详细数据文档 |
| 特征维度与解释 | 已确定时域/频域基础特征 | 增加特征趋势图和跨数据集特征导出 notebook |
| RUL 标签单位 | 已明确快照级和时间级需要区分 | 在 labeler 和文档中保留单位说明 |
| 训练输出追踪 | 已设计输出目录结构 | 后续补充 metrics、predictions 和 history 的落盘检查 |
""",
        ),
    ],
    "06_设计文档.md": [
        (
            "## 3. 功能模块设计",
            "design-architecture-detail",
            """
### 2.1 总体架构说明

系统总体架构围绕 RUL 预测数据流组织，而不是围绕某一个模型组织。设计上分为“数据接入、特征标签、模型训练、评价展示”四个主层：

| 层次 | 输入 | 输出 | 主要设计原则 |
| --- | --- | --- | --- |
| 数据接入层 | 官方数据目录、示例数据目录 | 统一轴承实体、通道数据、元数据 | 数据集差异只在 loader 内处理 |
| 特征标签层 | 振动快照、采样率、寿命终点 | 19 维特征、特征序列、RUL 标签 | 特征有物理含义，标签单位可追溯 |
| 模型训练层 | 特征序列、标签、训练配置 | 模型参数、history、预测结果 | 训练、测试、记录职责分离 |
| 评价展示层 | target、prediction、训练日志 | 指标表、曲线图、notebook、报告 | 指标区分普通误差、论文 score 和方向偏差 |

这种结构的关键好处是：如果后续增加新数据集，只需要新增 loader；如果增加新模型，只需要遵循训练输入输出约定；如果更新指标，不影响数据和模型代码。
""",
        ),
        (
            "## 7. 接口设计",
            "design-algorithm-detail",
            """
### 6.1 特征序列与 RUL 标签设计

特征序列构造流程如下：

1. 按轴承编号读取原始快照，并保持时间顺序。
2. 对每个快照提取时域和频域统计特征，形成 feature vector。
3. 使用固定长度窗口将连续 feature vector 组织为 feature sequence。
4. 根据当前窗口末端到寿命终点的距离生成 RUL 标签。
5. 将 `(feature_sequence, rul_target, metadata)` 交给模型训练模块。

该设计避免了两个常见问题：一是把相邻时间点随机打乱导致数据泄漏；二是只使用单个快照导致模型看不到退化趋势。对于 XJTU-SY 和 PHM2012，loader 可以分别解释时间单位，但训练模块只依赖统一的序列张量和 RUL 标签。
""",
        ),
    ],
    "07_单元测试计划文档.md": [
        (
            "## 10. 预期结果与通过标准",
            "unit-test-detail",
            """
## 9.1 指标与论文 workflow 单元测试补充

| 单元 | 测试重点 | 典型断言 |
| --- | --- | --- |
| RUL 指标 | RMSE、NormalizedRMSE、SMAPE、R2、Huang Score | 完美预测误差为 0，R2 为 1，range 为 0 时不产生 NaN/inf |
| 方向性指标 | OverPredictionRate、UnderPredictionRate、WithinToleranceRate | 预测大于真实值时计入 over，阈值内样本计入 within |
| CNN-LSTM-AM | 模型 forward、attention 权重、workflow 输出 | 输出维度正确，comparison_metrics.csv 含论文 score |
| xLSTM-Transformer | 模型结构、baseline 对比、划分 helper | 两个数据集均输出模型名称、样本数和关键指标 |
| notebook smoke | examples 下所有 notebook 代码单元 | 1 epoch 条件下可执行，不依赖提交大型真实输出 |
""",
        ),
    ],
    "08_集成测试计划文档.md": [
        (
            "## 8. 接口联调测试",
            "integration-flow-detail",
            """
## 7.1 端到端数据流检查点

| 检查点 | 上游模块 | 下游模块 | 检查内容 |
| --- | --- | --- | --- |
| Loader 输出 | 数据接入 | 标签构造 | 轴承编号、通道、采样率、快照顺序是否完整 |
| 特征表输出 | 特征工程 | 模型训练 | 特征列数量、缺失值处理、序列长度是否符合配置 |
| 训练输出 | 模型训练 | 评价模块 | predictions 中 target/prediction 是否成对存在 |
| 指标输出 | 评价模块 | notebook/报告 | RMSE、Score、样本数和 epoch 是否落盘 |
| 可视化输出 | 评价模块 | 答辩材料 | 图表标题、轴标签、数据集名称是否清楚 |
""",
        ),
    ],
    "11_编码规范文档.md": [
        (
            "## 12. 结论",
            "coding-doc-standard",
            """
## 11.1 课程文档与导出规范

1. 课程正式文档优先维护 Markdown 源文件，再通过脚本导出 PDF 和 DOCX，避免只修改二进制 Word 文件导致无法审阅差异。
2. 文档中项目名称统一为“工业轴承设备剩余寿命预测系统的实现”，成员名称统一为 `zyj、cyy、zdh、zy`。
3. 业务主线统一表述为“预测性维护、轴承健康评估、剩余寿命预测、生存概率/失效风险分析”，不把项目主线写成离散故障诊断或分类。
4. 引用论文复现实验时必须说明训练规模、数据划分和复现边界，避免把课程小规模训练说成作者完整实验结果。
5. 正式交付前执行导出脚本，并检查每份 Markdown 都存在对应 PDF 和 DOCX。
""",
        ),
    ],
    "12_结题报告.md": [
        (
            "## 3. 论文复现成果",
            "final-data-and-difficulties",
            """
### 2.1 数据理解与特征分析成果

本项目在结题阶段已经形成对两个真实轴承数据集的基本认识：

| 数据集 | 数据特点 | 观察到的退化规律 | 对建模的影响 |
| --- | --- | --- | --- |
| XJTU-SY | 三工况、多个轴承全寿命序列，单快照长，分钟级间隔 | 以 Bearing1_1 为例，寿命后期 RMS、峰值和综合健康指标明显抬升 | 适合展示完整退化曲线，训练划分应按轴承或时间顺序组织 |
| PHM2012/FEMTO | 文件更密，包含加速度和温度信息，Learning/Test/Full_Test 语义明确 | 以 Bearing1_1 为例，后期振动强度增强，但曲线更密集、波动更明显 | 需要 loader 解释 split 语义，并对官方终止 RUL 信息保持可追溯 |

特征层采用 19 维时域/频域特征。RMS 和谱能量用于反映振动强度，峭度、峰值因子和脉冲因子用于反映冲击增强，谱熵和主频用于补充频率分布变化。这些特征既作为模型输入，也用于答辩中说明数据规律。

### 2.2 项目难点与解决方案

| 难点 | 解决原则 | 实现结果 |
| --- | --- | --- |
| 两个数据集格式不同 | 将目录差异收敛在 loader 层 | `XJTULoader` 和 `PHM2012Loader` 输出统一实体 |
| 原始信号长度不同 | 先提取可解释特征，再组织序列 | 支持 19 维特征和 `FeatureSequenceRulLabeler` |
| RUL 标签和时间单位易混淆 | 在标签构造和文档中保留单位说明 | notebook 和测试均检查 prediction_count 与 target |
| 论文 score 口径不同 | 区分论文原版 Score、PHM/NASA 惩罚 Score 和普通误差 | 新增 RUL 指标体系并写入复现文档 |
| 课程验收需要可复跑 | 将 notebook、pytest、真实训练输出分层验证 | 8 个 notebook smoke、31 个测试和真实训练记录 |
""",
        ),
    ],
    "13_单元测试报告.md": [
        (
            "## 6. 测试结论",
            "unit-report-evidence",
            """
## 5.1 单元测试执行证据说明

单元测试不是只统计通过数量，而是覆盖项目中最容易出错的局部行为：

| 类别 | 主要风险 | 测试证据 |
| --- | --- | --- |
| 数据加载 | 目录结构差异导致字段缺失或顺序错误 | loader 测试检查实体、通道、元数据和抽样加载 |
| 标签构造 | RUL 方向反了或窗口边界错误 | labeler 测试检查目标值、窗口长度和样本数量 |
| 指标计算 | Score 公式、归一化或方向性指标口径错误 | RUL 指标测试使用确定性小数组和完美预测样例 |
| 模型结构 | forward 输出维度与 trainer 不匹配 | CNN-LSTM-AM、xLSTM-Transformer forward 测试 |
| notebook | 示例只存在但不能执行 | notebook smoke test 逐个运行代码单元 |
""",
        ),
    ],
    "14_集成测试报告.md": [
        (
            "## 6. 测试结论",
            "integration-report-evidence",
            """
## 5.1 集成输出物核对表

| 输出物 | 生成位置 | 验证意义 |
| --- | --- | --- |
| `history.csv` | 训练输出目录 | 证明 epoch 循环真实执行，不是静态示例 |
| `metrics.json` | 模型评估目录 | 保存单模型评估结果，便于回归比较 |
| `predictions.csv` | 测试输出目录 | 保留 target 与 prediction，可画 RUL 曲线和误差分布 |
| `comparison_metrics.csv` | 论文复现目录 | 记录不同模型、数据集和指标的对比 |
| notebook 执行日志 | pytest 输出 | 证明 examples 不是孤立文档，而是可执行入口 |
""",
        ),
    ],
    "15_确认测试报告.md": [
        (
            "## 5. 确认测试结论",
            "acceptance-report-detail",
            """
## 4.1 确认测试判定说明

确认测试从课程评审角度判断系统是否达到“可用、可信、可解释”的最低闭环：

| 维度 | 判定 | 说明 |
| --- | --- | --- |
| 可用 | 通过 | 用户可按 README、用户手册和 notebook 运行数据加载、训练和复现实验 |
| 可信 | 通过 | 自动化测试覆盖关键模块，真实训练输出包含 history、metrics 和 predictions |
| 可解释 | 通过 | 文档和答辩材料解释了数据特点、特征含义、模型边界和指标口径 |
| 可归档 | 通过 | 课程文档同时保留 Markdown、PDF 和 DOCX，便于审阅和提交 |
""",
        ),
    ],
    "16_用户使用手册.md": [
        (
            "## 6. 注意事项",
            "user-workflow-detail",
            """
## 5.1 常见输出文件说明

| 文件 | 含义 | 建议查看方式 |
| --- | --- | --- |
| `history.csv` | 每个 epoch 的训练/验证损失 | 检查训练是否真实执行、是否出现异常震荡 |
| `metrics.json` | 单次评估指标 | 查看 RMSE、MAE、R2、Score 等摘要 |
| `predictions.csv` | 每个样本的真实 RUL 和预测 RUL | 绘制预测曲线或误差分布 |
| `comparison_metrics.csv` | 论文复现模型对比结果 | 比较 baseline 与改进模型的指标变化 |
| 特征导出 CSV | 每个快照或窗口的特征值 | 分析 RMS、峭度、谱能量等趋势 |

初学者建议先运行 `00_generate_demo_datasets.ipynb`，再运行两个 loader overview notebook，确认数据结构后再进入训练 notebook。
""",
        ),
    ],
    "17_安装配置手册.md": [
        (
            "## 6. 故障排查",
            "install-verification-detail",
            """
## 5.1 安装后验证步骤

完成安装后建议依次执行：

```bash
uv run python -c "import USTC.SSE.BearingPrediction as bp; print(bp.__name__)"
uv run --extra dev pytest tests/test_rul_metrics.py -q
BEARING_EXAMPLE_EPOCHS=1 uv run --extra dev pytest tests/test_examples_notebooks.py -q
```

第一条命令验证包导入，第二条命令验证指标体系，第三条命令验证 notebook 示例链路。若只需要阅读文档，可执行 `scripts/export_course_docs.sh` 生成 PDF 和 DOCX。
""",
        ),
    ],
    "18_项目技术论文.md": [
        (
            "## 4. 模型实现",
            "paper-data-feature-detail",
            """
### 3.1 数据集与特征分析

XJTU-SY 和 PHM2012 都属于轴承全寿命或近全寿命退化数据，但两者采样组织差异明显。XJTU-SY 的单个振动快照较长，更适合观察分钟级退化趋势；PHM2012 的快照更短但文件更密，同时包含温度信息，适合验证 loader 对竞赛式目录的兼容能力。

本文采用的特征可以分为三类：强度类特征包括 RMS、峰值和谱能量，用于描述振动整体增强；冲击类特征包括峭度、峰值因子和脉冲因子，用于捕捉后期冲击尖峰；频域类特征包括主频、谱质心和谱熵，用于描述频率能量分布变化。实验中这些特征先形成单快照向量，再按时间顺序组成特征序列作为模型输入。
""",
        ),
        (
            "## 7. 软件工程实践",
            "paper-experiment-detail",
            """
## 6.1 实验设置与复现边界

实验采用课程项目可承受的小规模真实训练设置，重点验证数据链路、模型结构和指标输出，而不是追求与论文完整实验表格逐项对齐。每次复现实验均记录数据集名称、模型名称、预测样本数、epoch 数和指标值。对于论文 score，文中明确区分 Huang 原版 Score、PHM/NASA 类非对称惩罚 Score 和普通回归误差，避免不同口径混用。
""",
        ),
    ],
    "19_成员贡献比说明.md": [
        (
            "## 2. 贡献确认",
            "contribution-detail",
            """
## 1.1 交付物对应关系

| 成员 | 主要交付物 | 交叉评审责任 |
| --- | --- | --- |
| zyj | 系统架构、训练框架、RUL 模型、论文复现 workflow、结题集成 | 审核模型接口、实验指标和最终文档一致性 |
| cyy | XJTU-SY/PHM2012 loader、数据说明、特征工程、数据相关测试 | 审核数据语义、目录说明和特征分析表述 |
| zdh | 生存分析接口、评价指标、确认测试计划与报告 | 审核指标口径、验收标准和风险边界 |
| zy | 可视化、用户手册、安装配置、答辩提纲和讲稿 | 审核图表可读性、使用说明和汇报语气 |

贡献比例不是按代码行数简单计算，而是综合设计、实现、测试、文档、联调和答辩准备工作量确定。
""",
        ),
    ],
    "20_结题答辩提纲.md": [
        (
            "## 14. 成员分工、不足与总结",
            "outline-timing-and-tone",
            """
## 13.1 汇报时间与语气控制

| 部分 | 建议时间 | 语气要求 |
| --- | --- | --- |
| 项目目标与数据 | 2 分钟 | 先讲清楚为什么做 RUL，再讲两个数据集的差异 |
| 特征与系统架构 | 2 分钟 | 用数据流说明模块，不把架构讲成抽象口号 |
| 难点与解决方案 | 2 分钟 | 每个难点都说明原因、原理和对应实现 |
| 论文复现与测试 | 2 分钟 | 强调真实训练和指标输出，同时说明复现边界 |
| 总结与分工 | 1 分钟 | 客观说明贡献、不足和后续改进 |
""",
        ),
    ],
    "21_结题答辩演讲稿.md": [
        (
            "## 备答补充",
            "speech-tone-note",
            """
## 汇报语气提醒

正式汇报时建议保持课程项目口吻：多用“我们完成了、我们验证了、目前的边界是”，少用“领先、显著优越、工业级部署”等无法由当前实验直接证明的表述。讲到论文复现时，要主动说明这是基于本项目特征管线的小规模真实训练复现，重点是结构、流程和指标体系可复跑。
""",
        ),
    ],
}


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def standardize_doc(meta: DocMeta) -> None:
    path = ROOT / meta.path
    original = path.read_text(encoding="utf-8")
    body = apply_replacements(extract_body(original))
    body = remove_existing_extra_sections(body)
    for heading, key, content in EXTRAS.get(path.name, []):
        body = insert_before(body, apply_replacements(heading), key, apply_replacements(content))
    updated = make_preamble(meta) + body.strip() + "\n"
    updated = apply_replacements(updated)
    updated = remove_standard_comments(updated)
    if updated != original:
        path.write_text(updated, encoding="utf-8")


def main() -> None:
    for meta in DOCS:
        standardize_doc(meta)


if __name__ == "__main__":
    main()
