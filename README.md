# 🧬 Health Screener & Lab Explainer

ระบบ AI สำหรับช่วยวิเคราะห์ผลตรวจสุขภาพ (Lab Results) พร้อมอธิบายผลเป็นภาษาไทยแบบเข้าใจง่าย  
ใช้ NER + Multi-LLM (Claude / Gemini / GPT) พร้อมระบบ Validation สำหรับทดสอบความแม่นยำ

> ⚠️ ระบบนี้เป็นเพียงเครื่องมือช่วยวิเคราะห์เบื้องต้น ไม่ใช่การวินิจฉัยโรค  
> ควรปรึกษาแพทย์ผู้เชี่ยวชาญก่อนตัดสินใจทางการแพทย์

## 🚀 Features

### 🔬 Input รองรับ 2 แบบ
- อัปโหลดภาพผลแลป (Image Upload)
- กรอกข้อมูลแบบข้อความ (Manual Input)

### 🤖 AI Models
- Claude (Anthropic)
- Gemini (Google)
- GPT (OpenAI)

### 🧠 NLP Pipeline
- Named Entity Recognition (NER)
- ใช้ `bert-base-NER` สำหรับดึงค่าตรวจอัตโนมัติ

### 📊 ผลลัพธ์การวิเคราะห์
- ระดับความเสี่ยง:
  - 🟢 ปกติ (Normal)
  - 🟡 ปานกลาง (Moderate)
  - 🔴 สูง (High)
- ตารางค่าผิดปกติ
- คำอธิบายภาษาไทย
- คำแนะนำด้านสุขภาพ

### 📈 Validation System
- ทดสอบความแม่นยำของ AI
- Benchmark test cases
- วิเคราะห์ performance ของ pipeline

### 📦 Logging & Feedback
- บันทึกผลการวิเคราะห์
- เก็บ user feedback
- ใช้สำหรับปรับปรุงโมเดล

## 📁 Project Structure

- app.py                  # Streamlit main app
- ner_pipeline.py         # NER extraction pipeline
- validation.py           # Benchmark & validation system
- validation_log.json     # logs
- requirements.txt        # dependencies
- README.md               # documentation

## ⚙️ Installation

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. ตรวจสอบระบบ

```bash
python -c "import streamlit, anthropic, openai, google.genai, PIL, transformers, torch, torchvision, pandas, plotly; print('All packages installed successfully!')"
```
หากขึ้น: All packages installed successfully! แปลว่าพร้อมใช้งาน 🎉

### 3. รันแอป

```bash
python -m streamlit run app.py
```

เปิดเบราว์เซอร์: http://localhost:8501


## 🖥️ วิธีใช้งาน

### 🔹 Sidebar

ผู้ใช้สามารถ:

* เลือก AI Model (Claude / Gemini / GPT)
* ใส่ API Key
* เปิด/ปิดระบบ NER


### 🔹 วิเคราะห์ผลตรวจ

1. อัปโหลดภาพ หรือกรอกค่าตรวจ
2. ใส่ข้อมูลเพิ่มเติม:

   * อายุ
   * เพศ
   * BMI
   * อาการ
3. กดปุ่ม **🔬 วิเคราะห์ผลตรวจ**

ระบบจะ:

* ดึงค่าด้วย NER
* วิเคราะห์ด้วย LLM
* สรุประดับความเสี่ยง
* แสดงผลเป็นภาษาไทย
* ให้คำแนะนำสุขภาพ


### 🔹 Validation Tab

ใช้สำหรับ:

* ทดสอบ accuracy ของระบบ
* รัน benchmark dataset
* ตรวจสอบ pipeline การทำงาน


## 🧠 AI Workflow

Input (Image / Text)
        ↓
OCR / Parsing
        ↓
NER (bert-base-NER)
        ↓
Structured Prompt
        ↓
LLM (Claude / Gemini / GPT)
        ↓
JSON Output
        ↓
UI Rendering


## 📊 Output Format (JSON)

```json
{
  "risk_level": "normal | moderate | high",
  "abnormal_values": [
    {
      "name": "ชื่อค่า",
      "value": "ค่าที่ตรวจได้",
      "reference": "ค่ามาตรฐาน",
      "status": "high | low | normal",
      "label": "สูง | ต่ำ | ปกติ"
    }
  ],
  "explanation_th": "คำอธิบายผลเป็นภาษาไทย",
  "recommendation": "คำแนะนำการดูแลสุขภาพ"
}
```

## 📊 Risk Classification

### 🟢 Normal

* ค่าทั้งหมดอยู่ในเกณฑ์ปกติ
* borderline ยังถือว่าปกติ

### 🟡 Moderate

* ผิดปกติเล็กน้อยถึงปานกลาง
* เช่น:

  * LDL 130–159
  * Cholesterol 200–239
  * HbA1c 5.7–7.4

### 🔴 High

* ผิดปกติรุนแรง
* เช่น:

  * LDL ≥ 160
  * HbA1c ≥ 7.5
  * AST/ALT ≥ 100

## 🛠️ Tech Stack

* 🧠 LLMs: Claude / Gemini / GPT
* 🧾 UI: Streamlit
* 🧠 NLP: HuggingFace Transformers
* 📊 Data: Pandas
* 📈 Visualization: Plotly
* 🖼️ Image: PIL
* ⚡ Backend: Python

## 📌 System Notes

* รองรับทั้ง Image และ Text input
* ใช้ NER ช่วย pre-process ก่อนส่งเข้า LLM
* มี validation pipeline สำหรับ testing
* เก็บ log และ feedback ผู้ใช้

## 📈 Validation System

ระบบสามารถ:

* รัน test cases มาตรฐาน
* วัดความถูกต้องของ pipeline
* เก็บ metrics การทำงาน
* เปรียบเทียบกับ manual analysis

## ⭐ Disclaimer

* ใช้เพื่อช่วยอธิบายผลเบื้องต้นเท่านั้น
* ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์
* ไม่สามารถแทนแพทย์ได้

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

แล้วเข้า: http://localhost:8501
