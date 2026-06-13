from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUTPUT_PATH = Path("docx/final/工业轴承故障预测系统的实现-结题答辩.pptx")

NAVY = RGBColor(15, 36, 74)
BLUE = RGBColor(37, 99, 235)
GREEN = RGBColor(22, 163, 74)
ORANGE = RGBColor(249, 115, 22)
RED = RGBColor(220, 38, 38)
SLATE = RGBColor(71, 85, 105)
LIGHT_BG = RGBColor(248, 250, 252)
CARD_BG = RGBColor(255, 255, 255)
LINE = RGBColor(226, 232, 240)
TEXT = RGBColor(17, 24, 39)
MUTED = RGBColor(100, 116, 139)


def set_background(slide, color: RGBColor = LIGHT_BG) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title(slide, title: str, subtitle: str = "工业轴承故障预测系统的实现 | 结题答辩") -> None:
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.78))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()

    box = slide.shapes.add_textbox(Inches(0.52), Inches(0.15), Inches(8.8), Inches(0.34))
    run = box.text_frame.paragraphs[0].add_run()
    run.text = title
    run.font.name = "PingFang SC"
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

    sub = slide.shapes.add_textbox(Inches(8.3), Inches(0.24), Inches(4.5), Inches(0.24))
    paragraph = sub.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.RIGHT
    run = paragraph.add_run()
    run.text = subtitle
    run.font.name = "PingFang SC"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(226, 232, 240)


def add_footer(slide) -> None:
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.58), Inches(7.08), Inches(12.1), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = LINE
    line.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.62), Inches(7.13), Inches(11.8), Inches(0.18))
    run = box.text_frame.paragraphs[0].add_run()
    run.text = "中国科学技术大学 软件学院 | 软件工程课程结题答辩"
    run.font.name = "PingFang SC"
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


def add_bullets(slide, left: float, top: float, width: float, items: list[str], *, size: int = 20) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.8))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.bullet = True
        paragraph.space_after = Pt(8)
        paragraph.font.name = "PingFang SC"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = TEXT


def add_card(slide, left: float, top: float, width: float, height: float, title: str, body: str, color: RGBColor) -> None:
    card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = LINE

    accent = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(left), Inches(top), Inches(0.12), Inches(height))
    accent.fill.solid()
    accent.fill.fore_color.rgb = color
    accent.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(left + 0.28), Inches(top + 0.16), Inches(width - 0.4), Inches(0.3))
    run = title_box.text_frame.paragraphs[0].add_run()
    run.text = title
    run.font.name = "PingFang SC"
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = color

    if body and height > 0.75:
        body_box = slide.shapes.add_textbox(Inches(left + 0.28), Inches(top + 0.55), Inches(width - 0.42), Inches(height - 0.65))
        frame = body_box.text_frame
        frame.word_wrap = True
        run = frame.paragraphs[0].add_run()
        run.text = body
        run.font.name = "PingFang SC"
        run.font.size = Pt(13)
        run.font.color.rgb = TEXT


def add_table_like_rows(slide, left: float, top: float, rows: list[tuple[str, str, str]]) -> None:
    for index, (label, middle, right) in enumerate(rows):
        y = top + index * 0.62
        add_card(slide, left, y, 3.2, 0.48, label, "", BLUE)
        mid = slide.shapes.add_textbox(Inches(left + 3.55), Inches(y + 0.09), Inches(4.4), Inches(0.24))
        run = mid.text_frame.paragraphs[0].add_run()
        run.text = middle
        run.font.name = "PingFang SC"
        run.font.size = Pt(13)
        run.font.color.rgb = TEXT
        out = slide.shapes.add_textbox(Inches(left + 8.1), Inches(y + 0.09), Inches(3.6), Inches(0.24))
        run = out.text_frame.paragraphs[0].add_run()
        run.text = right
        run.font.name = "PingFang SC"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = GREEN


def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = 5765800
    prs.slide_height = 3244850

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, NAVY)
    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.22))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ORANGE
    bar.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.7), Inches(1.1), Inches(9.4), Inches(1.2))
    run = title.text_frame.paragraphs[0].add_run()
    run.text = "工业轴承故障预测系统的实现"
    run.font.name = "PingFang SC"
    run.font.size = Pt(38)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)
    subtitle = slide.shapes.add_textbox(Inches(0.76), Inches(2.35), Inches(8.2), Inches(0.4))
    run = subtitle.text_frame.paragraphs[0].add_run()
    run.text = "RUL 预测 | 真实数据接入 | 论文复现 | 软件工程结题答辩"
    run.font.name = "PingFang SC"
    run.font.size = Pt(19)
    run.font.color.rgb = RGBColor(226, 232, 240)
    info = slide.shapes.add_textbox(Inches(0.78), Inches(5.85), Inches(10.8), Inches(0.6))
    run = info.text_frame.paragraphs[0].add_run()
    run.text = "中国科学技术大学 软件学院《软件工程》 | 指导老师：zjf | zyj、cyj、zdh、zy"
    run.font.name = "PingFang SC"
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(226, 232, 240)

    slides = [
        ("项目目标", ["面向预测性维护场景，完成工业轴承 RUL 预测系统", "支持 XJTU-SY 与 PHM2012 两个真实数据集", "形成数据、特征、模型、训练、评估、文档的工程闭环"]),
        ("系统架构", ["dataset/data：统一 BearingEntity 与 loader", "preprocess/feature/labeling：完成信号处理、特征与 RUL 标签", "models/training/evaluation：统一模型训练、预测和指标输出", "examples/docs/docx：支撑实验复现和课程交付"]),
        ("核心实现", ["XJTULoader、PHM2012Loader 支持真实数据和读前抽样", "19 维时域/频域特征与 FeatureSequenceRulLabeler", "BaseTrainer、BaseTester、ExperimentTracker 统一训练记录", "RUL 指标体系覆盖论文 Score、普通误差和方向性偏差"]),
        ("论文复现一", ["Huang 等 CNN-LSTM-AM：CNN 局部编码 + LSTM 时序建模 + attention 聚合", "对比 CNN-LSTM baseline", "真实 XJTU-SY 和 PHM2012 上完成 8 epoch 小样本训练", "输出 huang_rul_score、normalized_rmse、smape 和相对变化列"]),
        ("论文复现二", ["Jiang 等 xLSTM-Transformer：xLSTM-inspired memory + Transformer encoder", "按论文划分 XJTU-SY 三工况与 PHM2012 三工况", "对比 Feature-Transformer 和 LSTM-Transformer baseline", "输出 RMSE、R2、PHM2012 Score、Huang RUL Score"]),
        ("测试与验收", ["pytest 覆盖 loader、labeler、metrics、model forward、prediction modes", "notebook 测试直接执行 examples/*.ipynb", "真实训练验收保存在 tmp/，指标摘要写入 PAPER_REPRODUCTION.md", "结题文档、测试报告、用户手册和安装手册已补齐"]),
        ("结论与展望", ["系统完成课程要求的可运行、可测试、可复现实验闭环", "当前复现为小样本真实训练，不冒充论文完整数值对齐", "后续可扩展全量训练、多随机种子统计和交互式实验看板"]),
    ]
    for title_text, items in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        set_background(slide)
        add_title(slide, title_text)
        add_bullets(slide, 0.95, 1.35, 11.3, items)
        add_footer(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    add_title(slide, "交付物清单")
    add_table_like_rows(
        slide,
        0.75,
        1.2,
        [
            ("源码", "src/USTC/SSE/BearingPrediction", "已完成"),
            ("示例", "examples/*.ipynb", "已跑通"),
            ("论文复现", "docs/PAPER_REPRODUCTION.md", "已完成"),
            ("测试", "tests + pytest", "已覆盖"),
            ("文档", "proposal / mid-term / final", "已补齐"),
            ("PPT", "docx/final/*.pptx", "已生成"),
        ],
    )
    add_footer(slide)

    return prs


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    presentation = build_presentation()
    presentation.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
