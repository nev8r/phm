# Final Defense Web PPT Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the final defense web PPT so it reads like a USTC software engineering course defense, not a technical product launch.

**Architecture:** Keep the existing Guizang Swiss HTML runtime and generator, but replace the slide story, wording, and evidence density. Synchronize the final outline and speech script with the 11-slide web deck.

**Tech Stack:** Python 3.11 script generation, Guizang Swiss HTML template, Node validator, pytest.

---

### Task 1: Rewrite Web Deck Story

**Files:**
- Modify: `scripts/generate_final_web_ppt.py`
- Generate: `docx/final/web-ppt/index.html`

- [ ] **Step 1: Change deck to 11 slides**

Use `TOTAL_SLIDES = 11`.

- [ ] **Step 2: Replace launch-style wording**

Replace English-heavy labels like `WHAT WILL BE PROVED`, `FINAL CLAIM`, `TAKEAWAYS`, and `PIPELINE AS THE PRODUCT` with Chinese course-defense labels such as `汇报主线`, `问题定义`, `系统架构`, `测试验收`.

- [ ] **Step 3: Add teacher-facing evidence**

Add XJTU-SY and PHM2012 sampling facts, 19-dimensional feature coverage, core class names, output files, and reproduction boundaries directly on slides.

- [ ] **Step 4: Regenerate HTML**

Run:

```bash
uv run python scripts/generate_final_web_ppt.py
```

Expected: `generated docx/final/web-ppt/index.html`.

### Task 2: Synchronize Final Speech Materials

**Files:**
- Modify: `docx/final/md/20_结题答辩提纲.md`
- Modify: `docx/final/md/21_结题答辩演讲稿.md`

- [ ] **Step 1: Align outline to 11 slides**

Use the 11-slide structure: title, RUL problem definition, scope, datasets, features, architecture, RUL workflow, CNN-LSTM-AM, xLSTM-Transformer, verification, limits.

- [ ] **Step 2: Rewrite script in student defense tone**

Use natural Chinese wording: specific, modest, evidence-backed. Avoid product-launch wording and avoid the old task framing.

### Task 3: Improve Offline Robustness

**Files:**
- Create: `docx/final/web-ppt/assets/motion.min.js`

- [ ] **Step 1: Copy Guizang local motion runtime**

Copy `.agents/skills/guizang-ppt-skill/assets/motion.min.js` to `docx/final/web-ppt/assets/motion.min.js` so the web deck does not rely on CDN animation fallback during defense.

### Task 4: Validate and Review

**Files:**
- Read-only review of generated artifacts.

- [ ] **Step 1: Run Guizang validator**

```bash
node .agents/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs docx/final/web-ppt/index.html
```

Expected: `Swiss deck validation passed: 11 slide(s).`

- [ ] **Step 2: Scan forbidden wording and placeholders**

```bash
tmp_scan_pattern="tmp/final_deck_forbidden_terms.txt"
python - <<'PY'
from pathlib import Path
terms = [
    "\u5de5\u4e1a\u8f74\u627f\u6545\u969c",
    "\u6545\u969c\u9884\u6d4b",
    "\u6545\u969c\u8bca\u65ad",
    "\u8bca\u65ad",
    "\u5206\u7c7b",
    "\u6df7\u6dc6\u77e9\u9635",
    "[\u5fc5\u586b]",
    "SLIDES_HERE",
    "P23",
    "P24",
    "<text",
]
Path("tmp/final_deck_forbidden_terms.txt").write_text("\\n".join(terms), encoding="utf-8")
PY
rg -n -F -f "$tmp_scan_pattern" docx/final/web-ppt scripts/generate_final_web_ppt.py docx/final/md/20_结题答辩提纲.md docx/final/md/21_结题答辩演讲稿.md
```

Expected: no matches.

- [ ] **Step 3: Re-run teacher/student review subagents**

Ask a teacher reviewer and a student rehearsal reviewer to inspect the redesigned deck and speech material. Fix concrete issues they find.

- [ ] **Step 4: Run tests**

```bash
uv run --extra dev pytest -q
```

Expected: all tests pass.

### Task 5: Commit and Push

**Files:**
- Stage only tracked final deliverables and the new web-ppt asset.
- Do not stage `.agents/`.

- [ ] **Step 1: Commit**

```bash
git add docs/superpowers/plans/2026-06-14-final-defense-web-ppt-redesign.md \
  scripts/generate_final_web_ppt.py \
  docx/final/web-ppt/index.html \
  docx/final/web-ppt/assets/motion.min.js \
  docx/final/md/20_结题答辩提纲.md \
  docx/final/md/21_结题答辩演讲稿.md
git commit -m "docs: redesign final defense web deck"
```

- [ ] **Step 2: Push**

```bash
git push origin main
```
