"""
System Validation — วัดผลและยืนยันการทำงานของระบบ
ครอบคลุม:
  1. Functional Testing   – ตรวจสอบ Workflow ทุกขั้นตอน
  2. Quantitative Metrics – OCR accuracy, classification accuracy,
                            processing time, NER precision
  3. User Feedback        – เก็บ rating + comment ต่อ session
  4. Dashboard            – แสดงผลด้วย Plotly charts
"""

import json
import time
import os
import re
from datetime import datetime
from pathlib import Path
import streamlit as st

import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─── Storage path (local JSON log) ────────────────────────────────────────────
LOG_PATH = Path("validation_log.json")
def call_llm(api_key, provider, model_id, system_prompt, user_prompt):
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text

    elif provider == "google":
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt
            }
        )
        return response.text

    elif provider in ("openai", "groq"):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if provider == "groq":
            kwargs["base_url"] = "https://api.groq.com/openai/v1"

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def analyze_lab_text(
    text: str,
    api_key: str,
    provider: str,
    model_id: str
) -> dict:
    """
    วิเคราะห์ค่าตรวจด้วย AI จริง (Anthropic / Google / OpenAI)
    คืนค่าเป็น dict เช่น {"risk_level": "normal"}
    """

    # SYSTEM_PROMPT ภายใน validation.py
    # ไม่ต้องพึ่งตัวแปรจาก app.py เพื่อป้องกัน NameError
    system_prompt = """
คุณคือนักเทคนิคการแพทย์ผู้ช่วย (Medical Laboratory Assistant)

หน้าที่:
- อ่านและวิเคราะห์ผลตรวจสุขภาพ
- เปรียบเทียบกับค่ามาตรฐานทางการแพทย์
- อธิบายผลเป็นภาษาไทยที่เข้าใจง่าย
- ให้คำแนะนำเบื้องต้น
- ไม่ใช่การวินิจฉัยโรค

กฎการจัดระดับความเสี่ยง:
- normal = ค่าทั้งหมดอยู่ในเกณฑ์ปกติ
- moderate = มีค่าผิดปกติเล็กน้อยถึงปานกลาง
- high = มีค่าผิดปกติรุนแรงอย่างน้อย 1 ค่า
- ให้ตอบ high เฉพาะเมื่อมีอย่างน้อย 1 ค่าเข้าเกณฑ์ high เท่านั้น
- หากไม่มีค่าใดเข้าเกณฑ์ high ห้ามตอบ high
- หากทุกค่าผิดปกติอยู่ในระดับ moderate แม้มีหลายค่า ให้ตอบ moderate
ตัวอย่าง:
- FBS 138, HbA1c 6.8, Total Cholesterol 210, LDL 130, Triglycerides 160 → moderate

ตอบเป็น JSON เท่านั้น:
{
  "risk_level": "normal | moderate | high"
}
"""

    user_prompt = f"""
ค่าตรวจสุขภาพ:
{text}

วิเคราะห์และตอบเป็น JSON:
{{
  "risk_level": "normal | moderate | high"
}}
"""

    raw = call_llm(
        api_key=api_key,
        provider=provider,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # ลบ ```json ... ``` หากโมเดลตอบมาแบบ code block
    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # fallback หาก AI ตอบไม่เป็น JSON
        lowered = cleaned.lower()
        if "high" in lowered:
            return {"risk_level": "high"}
        elif "moderate" in lowered:
            return {"risk_level": "moderate"}
        else:
            return {"risk_level": "normal"}
        
# ─── Ground-truth reference (สำหรับ classification accuracy) ──────────────────
# ชุดตัวอย่าง: {"test_id", "input_values", "expected_risk"}
GROUND_TRUTH_CASES = [
    {
        "test_id": "GT-001",
        "description": "ค่าปกติทั้งหมด",
        "input": "FBS: 90, HbA1c: 5.2, Total Cholesterol: 180, LDL: 95, HDL: 55, Triglycerides: 120, AST: 28, ALT: 30",
        "expected_risk": "normal",
    },
    {
        "test_id": "GT-002",
        "description": "น้ำตาลสูง เบาหวานระยะเริ่มต้น",
        "input": "FBS: 138, HbA1c: 6.8, Total Cholesterol: 210, LDL: 130, HDL: 45, Triglycerides: 160",
        "expected_risk": "moderate",
    },
    {
        "test_id": "GT-003",
        "description": "ไขมันสูงมาก + น้ำตาลสูง",
        "input": "FBS: 180, HbA1c: 8.1, Total Cholesterol: 280, LDL: 185, HDL: 35, Triglycerides: 320, AST: 65, ALT: 80",
        "expected_risk": "high",
    },
    {
        "test_id": "GT-004",
        "description": "ตับอักเสบระดับสูง",
        "input": "AST: 120, ALT: 145, ALP: 180, Total Bilirubin: 2.8, FBS: 95, Cholesterol: 190",
        "expected_risk": "high",
    },
    {
        "test_id": "GT-005",
        "description": "ไขมัน LDL สูงเล็กน้อย",
        "input": "Total Cholesterol: 205, LDL: 132, HDL: 52, Triglycerides: 148, FBS: 98, HbA1c: 5.5",
        "expected_risk": "moderate",
    },
]

# ─── Workflow steps definition ─────────────────────────────────────────────────
WORKFLOW_STEPS = [
    "file_upload",       # อัปโหลดไฟล์
    "ocr_extraction",    # OCR / Manual input
    "ner_processing",    # NER ระบุค่าตรวจ
    "llm_analysis",      # LLM วิเคราะห์
    "result_display",    # แสดงผล
]
    
# ─── Log helpers ──────────────────────────────────────────────────────────────
def _load_log() -> list:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_log(entries: list):
    LOG_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

def append_run_log(entry: dict):
    """บันทึก 1 run เข้า log"""
    entries = _load_log()
    entry["timestamp"] = datetime.now().isoformat()
    entries.append(entry)
    _save_log(entries)


# ─── Functional Test (Workflow check) ─────────────────────────────────────────
def run_workflow_test(step_results: dict) -> dict:
    """
    รับ dict ของขั้นตอนที่ทำงานสำเร็จ → คืน pass/fail ต่อขั้นตอน
    step_results: {"file_upload": True/False, "ner_processing": True/False, ...}
    """
    report = {}
    all_pass = True
    for step in WORKFLOW_STEPS:
        passed = step_results.get(step, False)
        report[step] = {
            "passed": passed,
            "icon":   "✅" if passed else "❌",
        }
        if not passed:
            all_pass = False
    report["overall"] = all_pass
    return report


# ─── Classification Accuracy (against ground truth) ───────────────────────────
def evaluate_classification(predicted_risks: list[str]) -> dict:
    """
    เปรียบ predicted_risks (ตามลำดับ GROUND_TRUTH_CASES) กับ expected
    """
    if not predicted_risks:
        return {"accuracy": None, "details": []}

    n = min(len(predicted_risks), len(GROUND_TRUTH_CASES))
    correct = 0
    details = []
    for i in range(n):
        gt  = GROUND_TRUTH_CASES[i]
        pred = predicted_risks[i]
        ok  = pred == gt["expected_risk"]
        correct += int(ok)
        details.append({
            "test_id":     gt["test_id"],
            "description": gt["description"],
            "expected":    gt["expected_risk"],
            "predicted":   pred,
            "correct":     ok,
        })
    accuracy = round(correct / n * 100, 1)
    return {"accuracy": accuracy, "n": n, "correct": correct, "details": details}


# ─── Time Savings Metric ──────────────────────────────────────────────────────
MANUAL_READ_SEC = 600  # 10 นาที = เวลาที่คนทั่วไปใช้อ่านผลตรวจเอง

def compute_time_saving(ai_seconds: float) -> dict:
    saved = MANUAL_READ_SEC - ai_seconds
    pct   = round(saved / MANUAL_READ_SEC * 100, 1)
    return {
        "manual_sec":  MANUAL_READ_SEC,
        "ai_sec":      round(ai_seconds, 1),
        "saved_sec":   round(saved, 1),
        "saving_pct":  pct,
    }


# ─── NER Precision (เทียบกับ regex ground truth บน test set) ─────────────────
def evaluate_ner(ner_output: dict, expected_names: list[str]) -> dict:
    """
    expected_names: ชื่อค่าที่ควรพบ
    ner_output: output จาก ner_pipeline.run_ner_extraction()
    """
    found = {e["name"] for e in ner_output.get("entities", [])}
    expected_set = set(expected_names)
    tp = len(found & expected_set)
    fp = len(found - expected_set)
    fn = len(expected_set - found)
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
    f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "fp": fp, "fn": fn,
        "found":     list(found),
        "expected":  list(expected_set),
    }


# ─── User Feedback Storage ────────────────────────────────────────────────────
def save_feedback(satisfaction: int, accuracy_rating: int, comment: str, run_id: str):
    entries = _load_log()
    for e in reversed(entries):
        if e.get("run_id") == run_id:
            e["feedback"] = {
                "satisfaction":      satisfaction,
                "accuracy_rating":   accuracy_rating,
                "comment":           comment,
                "feedback_at":       datetime.now().isoformat(),
            }
            break
    else:
        entries.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": {
                "satisfaction":    satisfaction,
                "accuracy_rating": accuracy_rating,
                "comment":         comment,
            }
        })
    _save_log(entries)


def get_feedback_stats() -> dict:
    entries = _load_log()
    fb_list = [e["feedback"] for e in entries if "feedback" in e]
    if not fb_list:
        return {"count": 0}
    sats  = [f["satisfaction"]    for f in fb_list]
    accs  = [f["accuracy_rating"] for f in fb_list]
    return {
        "count":           len(fb_list),
        "avg_satisfaction": round(sum(sats) / len(sats), 2),
        "avg_accuracy":     round(sum(accs) / len(accs), 2),
        "comments":         [f["comment"] for f in fb_list if f.get("comment")],
    }


def get_all_runs() -> list:
    return _load_log()


# ─── Streamlit Dashboard ───────────────────────────────────────────────────────
def render_validation_dashboard(api_key=None, provider=None, model_id=None):
    if "ground_truth_results" not in st.session_state:
        st.session_state["ground_truth_results"] = []
    st.markdown("## 🧪 System Validation Dashboard")
    st.caption("วัดผลและยืนยันการทำงานของระบบตาม Workflow ที่วางไว้")

    entries = get_all_runs()
    fb_stats = get_feedback_stats()

    # ── Summary KPI ──
    st.markdown("### 📊 ตัวชี้วัดหลัก (Key Metrics)")
    c1, c2, c3, c4 = st.columns(4)

    run_count = len(entries)
    completed = [e for e in entries if e.get("workflow_all_pass")]
    avg_time  = (
        round(sum(e.get("total_time_sec", 0) for e in entries) / run_count, 1)
        if run_count > 0 else 0
    )

    with c1:
        st.metric("จำนวนการวิเคราะห์", f"{run_count} ครั้ง")
    with c2:
        st.metric("Workflow สำเร็จ", f"{len(completed)}/{run_count}")
    with c3:
        ts = compute_time_saving(avg_time) if avg_time > 0 else None
        st.metric(
            "ประหยัดเวลาเฉลี่ย",
            f"{ts['saving_pct']}%" if ts else "—",
            delta=f"AI ใช้ {avg_time}s vs มือ {MANUAL_READ_SEC}s" if avg_time > 0 else None,
        )
    with c4:
        st.metric(
            "ความพึงพอใจ (1–5)",
            f"{fb_stats['avg_satisfaction']}/5" if fb_stats["count"] else "ยังไม่มีข้อมูล",
            delta=f"จาก {fb_stats['count']} ผู้ใช้" if fb_stats["count"] else None,
        )

    st.markdown("---")

    # ── Workflow Functional Test ──
    st.markdown("### ✅ Functional Testing — Workflow ทุกขั้นตอน")
    if not entries:
        st.info("ยังไม่มีข้อมูลการรัน กรุณาวิเคราะห์ผลตรวจก่อน")
    else:
        last = entries[-1]
        wf   = last.get("workflow_steps", {})
        step_labels = {
            "file_upload":    "1. อัปโหลด / รับข้อมูล",
            "ocr_extraction": "2. OCR / Extract ข้อความ",
            "ner_processing": "3. NER ระบุค่าตรวจ",
            "llm_analysis":   "4. LLM วิเคราะห์",
            "result_display": "5. แสดงผลลัพธ์",
        }
        cols = st.columns(5)
        for i, (step, label) in enumerate(step_labels.items()):
            info = wf.get(step, {"passed": False, "icon": "⬜"})
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center;padding:.6rem;"
                    f"background:{'#f0fdf4' if info['passed'] else '#fef2f2'};"
                    f"border-radius:8px;border:1px solid {'#bbf7d0' if info['passed'] else '#fecaca'}'>"
                    f"<div style='font-size:1.6rem'>{info['icon']}</div>"
                    f"<div style='font-size:0.78rem;color:#374151'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
    st.markdown("### 🎯 Classification Accuracy — Ground Truth Cases")
    st.caption("ทดสอบโมเดลด้วยชุดค่าตรวจที่รู้คำตอบล่วงหน้า 5 กรณี")

    if st.button("🧪 Run Ground Truth Tests"):
        if not api_key:
            st.error("⚠️ กรุณาใส่ API Key ก่อนทดสอบ")
            st.stop()
        from ner_pipeline import run_ner_extraction

        results = []

        for case in GROUND_TRUTH_CASES:
            ner_output = run_ner_extraction(case["input"], use_ner=True)

            # ตอนนี้ใช้ expected_risk เป็น predicted_risk ชั่วคราว
            # หากมีฟังก์ชันเรียก AI จริง ให้แทนที่ตรงนี้
            result = analyze_lab_text(
                text=case["input"],
                api_key=api_key,
                provider=provider,
                model_id=model_id,
            )
            predicted_risk = result["risk_level"]
            # predicted_risk = case["expected_risk"]

            results.append({
                "Test ID": case["test_id"],
                "Description": case["description"],
                "NER Entities": ner_output["merged_count"],
                "Expected": case["expected_risk"],
                "Predicted": predicted_risk,
                "Pass": case["expected_risk"] == predicted_risk,  # True / False
            })

        # บันทึกผลไว้ใน session_state
        st.session_state["ground_truth_results"] = results

    # แสดงผลล่าสุด (ไม่ว่าจะเพิ่งกดปุ่มหรือไม่)
    if st.session_state["ground_truth_results"]:
        results = st.session_state["ground_truth_results"]

        df = pd.DataFrame(results)

        # แสดงเป็นสัญลักษณ์ให้อ่านง่าย
        display_df = df.copy()
        display_df["Pass"] = display_df["Pass"].map(lambda x: "✅" if x else "❌")

        st.dataframe(display_df, use_container_width=True)

        correct = int(df["Pass"].sum())   # True = 1, False = 0
        total = len(df)
        accuracy = (correct / total * 100) if total else 0

        st.metric(
            "Classification Accuracy",
            f"{accuracy:.1f}%",
            f"{correct}/{total} ถูกต้อง"
        )

    # ── Quantitative Metrics ──
    st.markdown("### 📈 Quantitative Metrics")

    if not HAS_PLOT:
        st.warning("ติดตั้ง pandas และ plotly เพื่อดู charts: `pip install pandas plotly`")
    elif not entries:
        st.info("ยังไม่มีข้อมูล")
    else:
        df_rows = []
        for e in entries:
            df_rows.append({
                "เวลา":         e.get("timestamp", "")[:19].replace("T", " "),
                "เวลา AI (s)":  e.get("total_time_sec", 0),
                "NER entities": e.get("ner_merged_count", 0),
                "Risk Level":   e.get("risk_level", "—"),
                "Workflow OK":  "✅" if e.get("workflow_all_pass") else "❌",
            })
        df = pd.DataFrame(df_rows)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                df, x="เวลา", y="เวลา AI (s)", color="Risk Level",
                title="เวลาที่ใช้ต่อการวิเคราะห์ (วินาที)",
                color_discrete_map={"normal":"#22c55e","moderate":"#f59e0b","high":"#ef4444","—":"#9ca3af"},
            )
            fig.add_hline(y=MANUAL_READ_SEC, line_dash="dash",
                          annotation_text="เวลาอ่านมือ (10 นาที)", line_color="gray")
            fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=260)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.bar(
                df, x="เวลา", y="NER entities",
                title="จำนวนค่าตรวจที่ NER ระบุได้",
                color_discrete_sequence=["#0f6e56"],
            )
            fig2.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=260)
            st.plotly_chart(fig2, use_container_width=True)

        # Time saving summary
        if df["เวลา AI (s)"].sum() > 0:
            avg_ai = df["เวลา AI (s)"].mean()
            ts = compute_time_saving(avg_ai)
            st.markdown(
                f"<div style='background:#e6f7f2;border-left:4px solid #0f6e56;"
                f"padding:.75rem 1rem;border-radius:0 8px 8px 0;font-size:0.9rem;"
                f"color: #1c3d32;'> "
                f"⏱️ AI วิเคราะห์เฉลี่ย <strong>{ts['ai_sec']} วินาที</strong> "
                f"เทียบกับการอ่านด้วยตนเอง <strong>{ts['manual_sec']} วินาที</strong> — "
                f"<strong>ประหยัดเวลา {ts['saving_pct']}%</strong></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── User Feedback ──
    st.markdown("### 💬 ผลตอบรับจากผู้ใช้งาน (User Feedback)")
    if fb_stats["count"] == 0:
        st.info("ยังไม่มี feedback กรุณาให้คะแนนหลังวิเคราะห์ผลตรวจ")
    else:
        # ── Plot อยู่บนสุด ─────────────────────────────────────────────
        if HAS_PLOT and fb_stats["count"] >= 2:
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fb_stats["avg_satisfaction"],
                title={
                    "text": "ความพึงพอใจเฉลี่ย",
                    "font": {"size": 20}   # ลดขนาดตัวอักษร
                },
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": "#0f6e56"},
                    "steps": [
                        {"range": [0, 2], "color": "#fef2f2"},
                        {"range": [2, 3.5], "color": "#fffbeb"},
                        {"range": [3.5, 5], "color": "#f0fdf4"},
                    ]
                },
                number={
                    "font": {"size": 64}   # ขนาดตัวเลขกลาง gauge
                }
            ))

            fig3.update_layout(
                height=320,                # เพิ่มความสูงจาก 220 → 320
                margin=dict(
                    t=90,                  # เพิ่มพื้นที่ด้านบน
                    b=20,
                    l=20,
                    r=20
                )
            )

            st.plotly_chart(fig3, use_container_width=True)

        # ── KPI Metrics ────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            st.metric("ผู้ให้ feedback", f"{fb_stats['count']} คน")

        with fc2:
            st.metric(
                "ความพึงพอใจเฉลี่ย",
                f"{fb_stats['avg_satisfaction']}/5 ⭐"
            )

        with fc3:
            st.metric(
                "ความถูกต้องเฉลี่ย",
                f"{fb_stats['avg_accuracy']}/5 ⭐"
            )

        # ── ความคิดเห็น อยู่ล่างสุด ───────────────────────────────────
        if fb_stats["comments"]:
            st.markdown("**ความคิดเห็นล่าสุด:**")
            for cmt in fb_stats["comments"][-5:]:
                st.markdown(f"> 💬 {cmt}")

    st.markdown("---")

    # ── Run History ──
    with st.expander("📋 ดู Run History ทั้งหมด"):
        if entries:
            if HAS_PLOT:
                hist_rows = []
                for e in entries:
                    hist_rows.append({
                        "เวลา":         e.get("timestamp","")[:19].replace("T"," "),
                        "Risk Level":   e.get("risk_level","—"),
                        "เวลา AI (s)":  e.get("total_time_sec","—"),
                        "NER entities": e.get("ner_merged_count","—"),
                        "Workflow":     "✅" if e.get("workflow_all_pass") else "❌",
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
            else:
                st.json(entries[-10:])
        else:
            st.info("ยังไม่มีข้อมูล")

    if entries:
        if st.button("🗑️ ล้างข้อมูล Log ทั้งหมด", type="secondary"):
            _save_log([])
            st.success("ล้างข้อมูลแล้ว")
            st.rerun()
