"""
Health Screener & Lab Explainer
Main Streamlit Application
รวม: NER (bert-base-NER) + Claude AI + System Validation
"""

import streamlit as st
import anthropic
from openai import OpenAI
from google import genai
import base64
import json
import re
import time
import uuid
import io
from PIL import Image

from ner_pipeline import run_ner_extraction, format_for_prompt
from validation import (
    run_workflow_test, append_run_log, save_feedback,
    render_validation_dashboard, compute_time_saving,
    GROUND_TRUTH_CASES,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Screener & Lab Explainer",
    page_icon="🧬",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }

.main-header { text-align:center; padding:1.2rem 0 .8rem;
    border-bottom:1px solid #e5e7eb; margin-bottom:1.2rem; }
.main-header h1 { font-size:1.55rem; font-weight:600; color:#0f6e56; margin-bottom:.2rem; }
.main-header p  { font-size:.88rem; color:#6b7280; }

.risk-high     { background:#fef2f2; border-left:5px solid #ef4444;
    padding:.9rem 1.1rem; border-radius:8px; margin-bottom:.9rem; }
.risk-moderate { background:#fffbeb; border-left:5px solid #f59e0b;
    padding:.9rem 1.1rem; border-radius:8px; margin-bottom:.9rem; }
.risk-normal   { background:#f0fdf4; border-left:5px solid #22c55e;
    padding:.9rem 1.1rem; border-radius:8px; margin-bottom:.9rem; }

.badge-high   { background:#fef2f2; color:#b91c1c; padding:2px 10px;
    border-radius:999px; font-size:.77rem; font-weight:600; }
.badge-low    { background:#fffbeb; color:#b45309; padding:2px 10px;
    border-radius:999px; font-size:.77rem; font-weight:600; }
.badge-normal { background:#f0fdf4; color:#15803d; padding:2px 10px;
    border-radius:999px; font-size:.77rem; font-weight:600; }

.rec-box { background:#e6f7f2; border-left:4px solid #0f6e56;
    padding:.9rem 1.1rem; border-radius:0 8px 8px 0;
    font-size:.91rem; line-height:1.75; }
.disclaimer { background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;
    padding:.55rem 1rem; font-size:.79rem; color:#6b7280;
    text-align:center; margin-top:.9rem; }

.ner-chip { display:inline-block; background:#eff6ff; color:#1d4ed8;
    border:1px solid #bfdbfe; border-radius:999px; padding:2px 10px;
    font-size:.77rem; margin:2px; }
.step-box { text-align:center; padding:.55rem;
    border-radius:8px; border:1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Health Screener & Lab Explainer</h1>
    <p>อัปโหลดผลตรวจสุขภาพ → NER ระบุค่า → Claude AI วิเคราะห์ → คำอธิบายภาษาไทย</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ การตั้งค่า")

    # ── Model selector ────────────────────────────────────────────────────────
    MODEL_OPTIONS = {
        "Claude Opus 4.5  (Anthropic)":        ("anthropic", "claude-opus-4-5"),
        "Claude Sonnet 4.5  (Anthropic)":      ("anthropic", "claude-sonnet-4-5"),
        "Claude Haiku 4.5  (Anthropic)":       ("anthropic", "claude-haiku-4-5-20251001"),
        "Gemini 3 Flash Preview  (Google)":    ("google",    "gemini-3-flash-preview"),
        "Gemini 3.1 Pro Preview  (Google)":    ("google",    "gemini-3.1-pro-preview"),
        "GPT-5 (OpenAI)":                      ("openai",    "gpt-5"),
        "GPT-5 Mini (OpenAI)":                 ("openai",    "gpt-5-mini"),
        "GPT-4.1 (OpenAI)":                    ("openai",    "gpt-4.1"),

    }

    selected_label = st.selectbox(
        "🤖 เลือกโมเดล AI",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Anthropic → ใช้ Anthropic API Key\nGoogle → ใช้ Google AI API Key\nOpenAI → ใช้ OpenAI API Key",
    )
    provider, model_id = MODEL_OPTIONS[selected_label]

    # badge แสดง provider
    if provider == "google":
        badge_color = "#e0f2fe"
        badge_text_color = "#0369a1"
        provider_label = "Google AI"
    elif provider == "openai":
        badge_color = "#f3e8ff"
        badge_text_color = "#7c3aed"
        provider_label = "OpenAI"
    else:
        badge_color = "#e6f7f2"
        badge_text_color = "#0f6e56"
        provider_label = "Anthropic"
    st.markdown(
        f'<span style="background:{badge_color};color:{badge_text_color};'
        f'padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600">'
        f'🔑 ต้องการ: {provider_label} API Key</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── API Key (label เปลี่ยนตาม provider) ──────────────────────────────────
    if provider == "google":
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="ขอได้ที่ https://aistudio.google.com/apikey",
            placeholder="AIza...",
        )
    elif provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="ขอได้ที่ https://platform.openai.com/api-keys",
            placeholder="sk-...",
        )
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="ขอได้ที่ https://console.anthropic.com",
            placeholder="sk-ant-...",
        )
    st.session_state["api_key"] = api_key
    st.session_state["provider"] = provider
    st.session_state["model_id"] = model_id

    st.markdown("---")
    use_ner = st.toggle("🤖 เปิดใช้ NER (bert-base-NER)", value=True,
                        help="ใช้ HuggingFace NER ระบุค่าตรวจก่อนส่ง AI\n(โหลดครั้งแรกอาจช้า ~30s)")
    st.markdown("---")
    st.markdown("**Workflow**")
    st.markdown("1️⃣ อัปโหลด / พิมพ์ค่าตรวจ  \n"
                "2️⃣ NER ระบุชื่อ-ค่า-หน่วย  \n"
                "3️⃣ AI วิเคราะห์  \n"
                "4️⃣ แสดงผล + บันทึก Metrics")
    st.caption("⚠️ ไม่ใช่การวินิจฉัยโรค ควรปรึกษาแพทย์")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
main_tab, validation_tab = st.tabs(["🔬 วิเคราะห์ผลตรวจ", "📊 System Validation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — วิเคราะห์ผลตรวจ
# ══════════════════════════════════════════════════════════════════════════════
with main_tab:

    left_col, right_col = st.columns([1.1, 1], gap="large")

    # ── Input (left) ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown("#### 📥 ข้อมูลผลตรวจ")
        input_tab1, input_tab2 = st.tabs(["📷 อัปโหลดภาพ", "✏️ พิมพ์ค่าตรวจ"])

        uploaded_image   = None
        uploaded_mimetype = "image/png"
        manual_text      = ""

        with input_tab1:
            uploaded_file = st.file_uploader(
                "ภาพผลตรวจ (JPG/PNG)", type=["jpg","jpeg","png"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                manual_text = ""
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                uploaded_image    = base64.b64encode(buf.getvalue()).decode()
                uploaded_mimetype = "image/png"

        with input_tab2:
            manual_text = st.text_area(
                "ค่าตรวจ", height=180, label_visibility="collapsed",
                placeholder=(
                    "HbA1c: 6.8\nFBS: 130\nTotal Cholesterol: 220\n"
                    "LDL: 145\nHDL: 42\nTriglycerides: 180\n"
                    "AST: 45\nALT: 52\nWBC: 8.5"
                )
            )
        if manual_text.strip():
            uploaded_image = None
            uploaded_mimetype = "image/png"

        st.markdown("#### 🧾 ข้อมูลเพิ่มเติม")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("อายุ (ปี)", 0, 120, 0)
        with c2:
            gender = st.selectbox("เพศ", ["ไม่ระบุ","ชาย","หญิง"])
        with c3:
            bmi_v = st.number_input("BMI", 0.0, 60.0, 0.0, 0.1)

        symptoms = st.multiselect("อาการ / โรคประจำตัว", [
            "เหนื่อยง่าย","ปวดหัวบ่อย","กระหายน้ำมาก","ปัสสาวะบ่อย",
            "น้ำหนักลด","น้ำหนักเกิน","ไม่ออกกำลังกาย","สูบบุหรี่",
            "ดื่มแอลกอฮอล์","ความดันสูง","เบาหวาน","ไขมันสูง",
            "โรคหัวใจ","ไม่มีอาการผิดปกติ",
        ])

        analyze_btn = st.button("🔬 วิเคราะห์ผลตรวจ",
                                use_container_width=True, type="primary")

    # ── Result (right) ────────────────────────────────────────────────────────
    with right_col:
        st.markdown("#### 📊 ผลการวิเคราะห์")
        result_placeholder = st.empty()

        if "last_result" not in st.session_state:
            result_placeholder.info("กรอกข้อมูลและกด **วิเคราะห์ผลตรวจ** เพื่อดูผล")

    # ── Analyze logic ─────────────────────────────────────────────────────────
    if analyze_btn:
        for key in [
            "last_result",
            "last_run_id",
            "last_wf",
            "last_time",
            "last_saving",
            "last_ner",
        ]:
            st.session_state.pop(key, None)
        if not api_key:
            st.error("⚠️ กรุณาใส่ API Key ในแถบซ้าย")
            st.stop()
        if not uploaded_image and not manual_text.strip():
            st.error("⚠️ กรุณาอัปโหลดภาพ หรือพิมพ์ค่าตรวจก่อน")
            st.stop()

        run_id   = str(uuid.uuid4())[:8]
        t_start  = time.perf_counter()

        step_results = {s: False for s in
                        ["file_upload","ocr_extraction","ner_processing",
                         "llm_analysis","result_display"]}

        # Step 1 — file upload
        step_results["file_upload"] = bool(uploaded_image or manual_text.strip())

        # Step 2 — OCR / extract
        raw_text = manual_text.strip()
        if uploaded_image:
            raw_text = raw_text or "[image input — OCR via Vision AI]"
        step_results["ocr_extraction"] = bool(raw_text)

        # Step 3 — NER
        ner_output = {"entities": [], "merged_count": 0,
                      "ner_loaded": False, "processing_ms": 0}
        ner_structured = ""

        if raw_text and raw_text != "[image input — OCR via Vision AI]":
            with st.spinner("🤖 NER กำลังระบุค่าตรวจ (bert-base-NER)..."):
                ner_output    = run_ner_extraction(raw_text, use_ner=use_ner)
                ner_structured = format_for_prompt(ner_output)
            step_results["ner_processing"] = True
        else:
            # image path — NER ไม่มี text ให้ process แต่ถือว่า step ผ่าน
            step_results["ner_processing"] = True

        # Show NER result inline (under left col)
        if ner_output["entities"]:
            with left_col:
                with st.expander(
                    f"🔍 NER พบ {ner_output['merged_count']} ค่าตรวจ "
                    f"({ner_output['processing_ms']} ms)", expanded=False
                ):
                    chips = " ".join(
                        f'<span class="ner-chip">{e["name"]}: {e["value"]}'
                        f'{" " + e["unit"] if e.get("unit") else ""}'
                        f' <small style="color:#6b7280">({e["source"]})</small></span>'
                        for e in ner_output["entities"]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    if ner_output.get("ner_loaded"):
                        st.caption("✅ bert-base-NER โหลดสำเร็จ")
                    else:
                        st.caption("⚡ ใช้ Regex extraction (NER ปิดอยู่หรือโหลดไม่สำเร็จ)")

        # Step 4 — LLM
        SYSTEM_PROMPT = """คุณคือนักเทคนิคการแพทย์ผู้ช่วย (Medical Laboratory Assistant)
                        หน้าที่:
                        - อ่านและวิเคราะห์ผลตรวจสุขภาพ
                        - เปรียบเทียบกับค่ามาตรฐานทางการแพทย์ (Mayo Clinic / MedlinePlus)
                        - อธิบายผลเป็นภาษาไทยที่เข้าใจง่าย
                        - ให้คำแนะนำเบื้องต้น
                        - แจ้งเสมอว่าไม่ใช่การวินิจฉัยโรค

                        กฎการจัดระดับความเสี่ยง (risk_level):
                        1. normal
                        - ค่าตรวจทั้งหมดอยู่ในเกณฑ์ปกติ
                        - หากค่าอยู่ที่ขอบบนของค่าปกติ (เช่น LDL = 100) ให้ถือว่าเป็น normal

                        2. moderate
                        - มีค่าผิดปกติเล็กน้อยถึงปานกลาง
                        - เช่น LDL 130–159, Total Cholesterol 200–239, HbA1c 5.7–7.4, FBS 100–179
                        - ยังไม่มีค่าที่อยู่ในระดับรุนแรง

                        3. high
                        - มีค่าผิดปกติรุนแรงอย่างน้อย 1 ค่า
                        - เช่น LDL ≥ 160, Total Cholesterol ≥ 240, Triglycerides ≥ 300,
                            HbA1c ≥ 7.5, FBS ≥ 180, AST ≥ 100, ALT ≥ 100,
                            Total Bilirubin ≥ 2.0
                        - หรือมีค่าผิดปกติหลายค่าพร้อมกันอย่างชัดเจน

                        กฎเพิ่มเติม:
                        - LDL = 100 ให้จัดเป็น normal
                        - LDL 130–159 ให้จัดเป็น moderate
                        - LDL ≥ 160 ให้จัดเป็น high
                        - หากมีทั้งค่าปกติและผิดปกติเล็กน้อย ให้จัดเป็น moderate
                        - หากมีค่าผิดปกติรุนแรงเพียงค่าเดียว ให้จัดเป็น high
                        - ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON
                        """

        def build_prompt():
            parts = []
            if ner_structured:
                parts.append(f"ค่าตรวจ (ระบุโดย NER):\n{ner_structured}")
            elif manual_text.strip():
                parts.append(f"ค่าตรวจ:\n{manual_text.strip()}")

            if uploaded_image:
                parts.append("(มีภาพผลตรวจแนบมาด้วย กรุณาอ่านค่าเพิ่มเติมจากภาพ)")

            if age > 0:
                parts.append(f"อายุ: {age} ปี")
            if gender != "ไม่ระบุ":
                parts.append(f"เพศ: {gender}")
            if bmi_v > 0:
                parts.append(f"BMI: {bmi_v:.1f}")
            if symptoms:
                parts.append(f"อาการ: {', '.join(symptoms)}")

            parts.append("""
        วิเคราะห์และตอบเป็น JSON รูปแบบนี้เท่านั้น:
        {
        "risk_level": "normal | moderate | high",
        "abnormal_values": [
            {
            "name": "ชื่อค่า",
            "value": "ค่าที่ได้",
            "reference": "ค่ามาตรฐาน",
            "status": "high|low|normal",
            "label": "สูง|ต่ำ|ปกติ"
            }
        ],
        "explanation_th": "คำอธิบายทุกค่าเป็นภาษาไทย เข้าใจง่าย",
        "recommendation": "คำแนะนำดูแลสุขภาพเบื้องต้น เขียนเป็นข้อๆ"
        }
        """)
            return "\n\n".join(parts)

        result = None
        with st.spinner(f"🧠 {selected_label} กำลังวิเคราะห์..."):
            try:
                user_prompt = build_prompt()

                # ------------------------------------------------------------------
                # Anthropic Claude
                # ------------------------------------------------------------------
                if provider == "anthropic":
                    client = anthropic.Anthropic(api_key=api_key)

                    user_content = []
                    if uploaded_image:
                        user_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": uploaded_mimetype,
                                "data": uploaded_image
                            }
                        })

                    user_content.append({
                        "type": "text",
                        "text": user_prompt
                    })

                    response = client.messages.create(
                        model=model_id,   # ใช้ model ที่เลือกจาก sidebar
                        max_tokens=2000,
                        system=SYSTEM_PROMPT,
                        messages=[
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ]
                    )

                    raw = response.content[0].text.strip()

                # ------------------------------------------------------------------
                # Google Gemini
                # ------------------------------------------------------------------
                elif provider == "google":
                    from google import genai
                    from PIL import Image
                    import io
                    import base64

                    client = genai.Client(api_key=api_key)

                    contents = []

                    # ถ้ามีรูปภาพ ให้แปลง bytes -> PIL Image
                    if uploaded_image:
                        image_bytes = base64.b64decode(uploaded_image)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        contents.append(pil_image)

                    # เพิ่มข้อความ prompt
                    contents.append(f"{SYSTEM_PROMPT}\n\n{user_prompt}")

                    response = client.models.generate_content(
                        model=model_id,
                        contents=contents
                    )

                    raw = response.text.strip()

                # ------------------------------------------------------------------
                # OpenAI ChatGPT
                # ------------------------------------------------------------------
                elif provider == "openai":
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key)

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]

                    # หมายเหตุ: โค้ดนี้รองรับเฉพาะข้อความ
                    # หากต้องการส่งภาพด้วย ให้เพิ่ม logic สำหรับ vision models
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_completion_tokens=2000,
                    )

                    raw = response.choices[0].message.content.strip()

                else:
                    raise ValueError(f"ไม่รองรับ provider: {provider}")

                # ------------------------------------------------------------------
                # Parse JSON
                # ------------------------------------------------------------------
                cleaned = re.sub(r"```json|```", "", raw).strip()
                result = json.loads(cleaned)

                step_results["llm_analysis"] = True

            except json.JSONDecodeError:
                st.error("❌ AI ตอบในรูปแบบที่ไม่รองรับ ลองใหม่อีกครั้ง")
                with st.expander("ดูผลดิบ"):
                    st.text(raw if 'raw' in locals() else "ไม่มีข้อมูล")
                st.stop()

            except Exception as e:
                error_text = str(e).lower()

                if "authentication" in error_text or "api key" in error_text:
                    st.error("❌ API Key ไม่ถูกต้อง")
                else:
                    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                st.stop()

        t_end = time.perf_counter()
        total_time = round(t_end - t_start, 2)
        time_saving = compute_time_saving(total_time)
        # Step 5 — display
        step_results["result_display"] = result is not None
        wf_report = run_workflow_test(step_results)

        # บันทึก log
        append_run_log({
            "run_id":           run_id,
            "risk_level":       result.get("risk_level","—") if result else "—",
            "total_time_sec":   total_time,
            "ner_merged_count": ner_output.get("merged_count",0),
            "ner_loaded":       ner_output.get("ner_loaded",False),
            "workflow_steps":   wf_report,
            "workflow_all_pass": wf_report.get("overall",False),
        })

        # Store in session
        st.session_state["last_result"]  = result
        st.session_state["last_run_id"]  = run_id
        st.session_state["last_wf"]      = wf_report
        st.session_state["last_time"]    = total_time
        st.session_state["last_saving"]  = time_saving
        st.session_state["last_ner"]     = ner_output

    # ── Render result ──────────────────────────────────────────────────────────
    if "last_result" in st.session_state and st.session_state["last_result"]:
        result      = st.session_state["last_result"]
        run_id      = st.session_state["last_run_id"]
        wf_report   = st.session_state["last_wf"]
        total_time  = st.session_state["last_time"]
        time_saving = st.session_state["last_saving"]

        def risk_card(level):
            if level == "high":
                return "risk-high", "🔴 ระดับความเสี่ยง: สูง — ควรพบแพทย์โดยเร็ว"
            if level == "moderate":
                return "risk-moderate", "🟡 ระดับความเสี่ยง: ปานกลาง — ติดตามดูแลอย่างต่อเนื่อง"
            return "risk-normal", "🟢 ระดับความเสี่ยง: ปกติ — ผลตรวจอยู่ในเกณฑ์ดี"

        def badge(status):
            return {"high":"badge-high","low":"badge-low"}.get(status,"badge-normal")

        with right_col:
            result_placeholder.empty()

            css, txt = risk_card(result.get("risk_level","normal"))
            st.markdown(f'<div class="{css}"><strong>{txt}</strong></div>',
                        unsafe_allow_html=True)

            # Workflow steps strip
            wf_labels = {
                "file_upload":"อัปโหลด","ocr_extraction":"OCR",
                "ner_processing":"NER","llm_analysis":"AI","result_display":"แสดงผล",
            }
            wf_cols = st.columns(5)
            for i, (step, label) in enumerate(wf_labels.items()):
                info = wf_report.get(step, {"passed":False,"icon":"⬜"})
                with wf_cols[i]:
                    bg = "#f0fdf4" if info["passed"] else "#fef2f2"
                    bd = "#bbf7d0" if info["passed"] else "#fecaca"
                    st.markdown(
                        f'<div class="step-box" style="background:{bg};border-color:{bd}">'
                        f'{info["icon"]}<br><small>{label}</small></div>',
                        unsafe_allow_html=True
                    )

            st.markdown(
                f'<div style="font-size:.8rem;color:#6b7280;margin:.4rem 0">'
                f'⏱️ AI ใช้ <b>{total_time}s</b> | ประหยัดเวลา '
                f'<b>{time_saving["saving_pct"]}%</b> เทียบอ่านมือ (10 นาที)</div>',
                unsafe_allow_html=True
            )

            # Abnormal values table
            abnormal = result.get("abnormal_values", [])
            if abnormal:
                st.markdown("**📋 สรุปค่าตรวจ**")
                for v in abnormal:
                    ca, cb, cc, cd = st.columns([3,2,2,1.5])
                    with ca: st.markdown(f"**{v.get('name','')}**")
                    with cb: st.markdown(f"`{v.get('value','—')}`")
                    with cc: st.markdown(f"ref: `{v.get('reference','—')}`")
                    with cd:
                        b = badge(v.get("status","normal"))
                        st.markdown(f'<span class="{b}">{v.get("label","")}</span>',
                                    unsafe_allow_html=True)

            # Explanation
            st.markdown("**💬 คำอธิบาย**")
            st.info(result.get("explanation_th",""))

            # Recommendation
            st.markdown("**💡 คำแนะนำ**")
            st.markdown(f'<div class="rec-box">{result.get("recommendation","")}</div>',
                        unsafe_allow_html=True)

            st.markdown("""<div class="disclaimer">
            ⚠️ ข้อมูลนี้เป็นเพียงเบื้องต้น <strong>ไม่ใช่การวินิจฉัยโรค</strong>
            — หากค่าผิดปกติหรือมีอาการผิดปกติ กรุณาพบแพทย์
            </div>""", unsafe_allow_html=True)

            # ── User Feedback ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**⭐ ให้คะแนนความพึงพอใจ**")
            fb_c1, fb_c2 = st.columns(2)
            with fb_c1:
                sat = st.slider("ความพึงพอใจ", 1, 5, 4, key="sat_slider")
            with fb_c2:
                acc = st.slider("ความถูกต้องของผล", 1, 5, 4, key="acc_slider")
            cmt = st.text_input("ความคิดเห็น (ไม่บังคับ)", key="cmt_input")
            if st.button("📨 ส่ง Feedback", key="fb_btn"):
                save_feedback(sat, acc, cmt, run_id)
                st.success("✅ บันทึก Feedback แล้ว ขอบคุณ!")

            with st.expander("🔍 JSON ดิบ"):
                st.json(result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — System Validation
# ══════════════════════════════════════════════════════════════════════════════
with validation_tab:
    render_validation_dashboard(
        api_key=api_key,
        provider=provider,
        model_id=model_id,
    )