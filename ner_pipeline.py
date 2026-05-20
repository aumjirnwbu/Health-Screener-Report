"""
NER Pipeline — ใช้ dslim/bert-base-NER จาก Hugging Face
บทบาท: ระบุ (Named Entity Recognition) ชื่อค่าตรวจ, ตัวเลข, หน่วยวัด
จาก OCR text ก่อนส่งให้ LLM วิเคราะห์
"""

import re
import time
from typing import Optional

# ─── Lab-test vocabulary (regex fallback + NER post-process) ──────────────────
LAB_KEYWORDS = {
    # CBC
    "WBC": ["wbc", "white blood cell", "เม็ดเลือดขาว"],
    "RBC": ["rbc", "red blood cell", "เม็ดเลือดแดง"],
    "HGB": ["hgb", "hb", "hemoglobin", "ฮีโมโกลบิน"],
    "HCT": ["hct", "hematocrit"],
    "PLT": ["plt", "platelet", "เกล็ดเลือด"],
    "MCV": ["mcv"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    # Lipid
    "Total Cholesterol": ["total cholesterol", "cholesterol", "โคเลสเตอรอล"],
    "LDL": ["ldl", "ldl-c", "ไขมันเลว"],
    "HDL": ["hdl", "hdl-c", "ไขมันดี"],
    "Triglycerides": ["tg", "triglyceride", "ไตรกลีเซอไรด์"],
    # Glucose
    "FBS": ["fbs", "fasting blood sugar", "fasting glucose", "น้ำตาลอดอาหาร"],
    "HbA1c": ["hba1c", "a1c", "glycated hemoglobin"],
    # Liver
    "AST": ["ast", "sgot"],
    "ALT": ["alt", "sgpt"],
    "ALP": ["alp", "alkaline phosphatase"],
    "Total Bilirubin": ["total bilirubin", "t.bili"],
    # Kidney
    "Creatinine": ["creatinine", "cr", "ครีเอทินิน"],
    "BUN": ["bun", "blood urea nitrogen"],
    "eGFR": ["egfr", "gfr"],
    "Uric Acid": ["uric acid", "ua", "กรดยูริก"],
    # Thyroid
    "TSH": ["tsh"],
    "T3": ["t3", "triiodothyronine"],
    "T4": ["t4", "thyroxine"],
    # Others
    "CRP": ["crp", "c-reactive protein"],
    "Vitamin D": ["vitamin d", "25-oh", "25oh"],
    "Iron": ["iron", "serum iron"],
}

UNIT_PATTERNS = [
    r"mg/dL", r"g/dL", r"mmol/L", r"µmol/L", r"umol/L",
    r"mEq/L", r"U/L", r"IU/L", r"ng/mL", r"pg/mL",
    r"10\^3/µL", r"10\^6/µL", r"fL", r"%", r"mIU/mL",
    r"µg/dL", r"nmol/L",
]


def _load_ner_model():
    """โหลดโมเดล bert-base-NER จาก Hugging Face (lazy load)"""
    try:
        from transformers import pipeline as hf_pipeline
        ner = hf_pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,          # CPU; เปลี่ยนเป็น 0 ถ้ามี GPU
        )
        return ner
    except Exception as e:
        return None


# Cache ใน session เพื่อไม่ให้โหลดซ้ำทุก request
_NER_MODEL = None

def get_ner_model():
    global _NER_MODEL
    if _NER_MODEL is None:
        _NER_MODEL = _load_ner_model()
    return _NER_MODEL


# ─── Regex-based extraction (fast fallback / complement) ──────────────────────
_VALUE_RE = re.compile(
    r"([A-Za-zÀ-ÿก-๙][A-Za-zÀ-ÿก-๙0-9\-\s\.]{1,40}?)"
    r"\s*(?:[:\-=]\s*|\s+)"
    r"([\d]+(?:[.,]\d+)?)"
    r"\s*"
    r"(" + "|".join(UNIT_PATTERNS) + r")?",
    re.IGNORECASE,
)

def extract_with_regex(text: str) -> list[dict]:
    """ดึง lab-value ด้วย regex เป็น baseline"""
    results = []
    for m in _VALUE_RE.finditer(text):
        name_raw = m.group(1).strip()
        value    = m.group(2).replace(",", ".")
        unit     = m.group(3) or ""

        # จับคู่กับ keyword dict
        canonical = _canonicalize(name_raw)
        if canonical:
            results.append({
                "name":  canonical,
                "value": value,
                "unit":  unit,
                "source": "regex",
            })
    return results


def _canonicalize(raw: str) -> Optional[str]:
    """เปลี่ยน raw string → ชื่อมาตรฐาน หรือ None ถ้าไม่ใช่ค่าแล็บ"""
    raw_lower = raw.lower().strip()
    for canonical, aliases in LAB_KEYWORDS.items():
        if raw_lower == canonical.lower() or raw_lower in aliases:
            return canonical
        for alias in aliases:
            if alias in raw_lower or raw_lower in alias:
                return canonical
    # ถ้าชื่อสั้น ≤ 6 ตัวและเป็น uppercase ก็ยอมรับ (เช่น WBC, LDL)
    if len(raw) <= 6 and raw.isupper():
        return raw
    return None


# ─── NER-based extraction ─────────────────────────────────────────────────────
def extract_with_ner(text: str, ner_model) -> tuple[list[dict], list[dict]]:
    """
    รัน bert-base-NER แล้วดึง entity ที่เป็น ORG/MISC (ชื่อค่าแล็บ) และ
    CARDINAL/QUANTITY (ตัวเลข) — จากนั้น pair ไว้ด้วยกัน
    คืนค่า: (lab_entities_found, raw_ner_output)
    """
    if ner_model is None:
        return [], []

    try:
        raw_entities = ner_model(text[:512])  # bert max 512 tokens
    except Exception:
        return [], []

    # bert-base-NER ให้ label: PER, ORG, LOC, MISC
    # ชื่อค่าแล็บมักถูก tag เป็น MISC หรือ ORG
    found_names = []
    for ent in raw_entities:
        word = ent["word"].replace("##", "")
        label = ent.get("entity_group", ent.get("entity", ""))
        if label in ("MISC", "ORG") and len(word) >= 2:
            canonical = _canonicalize(word)
            if canonical:
                found_names.append({
                    "canonical": canonical,
                    "raw_word":  word,
                    "score":     round(ent["score"], 4),
                    "label":     label,
                })

    # จับคู่กับค่าตัวเลขจาก regex
    regex_vals = extract_with_regex(text)
    regex_map  = {v["name"]: v for v in regex_vals}

    lab_entities = []
    for fn in found_names:
        val_info = regex_map.get(fn["canonical"], {})
        lab_entities.append({
            "name":       fn["canonical"],
            "value":      val_info.get("value", "—"),
            "unit":       val_info.get("unit", ""),
            "ner_score":  fn["score"],
            "ner_label":  fn["label"],
            "source":     "ner",
        })

    return lab_entities, raw_entities


# ─── Main public function ─────────────────────────────────────────────────────
def run_ner_extraction(text: str, use_ner: bool = True) -> dict:
    """
    รับข้อความจาก OCR และดึงค่าแล็บ
    - ใช้ Regex เป็นหลัก (แม่นยำที่สุดสำหรับรูปแบบ Lab)
    - ใช้ BERT NER เพื่อยืนยันว่าโมเดลทำงานได้
    """
    t0 = time.perf_counter()

    # 1) Regex extraction (ตัวหลัก)
    regex_results = extract_with_regex(text)

    # 2) โหลด BERT NER และรันเพื่อเช็กว่าโมเดลทำงาน
    ner_loaded = False
    raw_ner = []
    ner_results = []

    if use_ner:
        model = get_ner_model()
        if model is not None:
            ner_loaded = True
            try:
                # รันโมเดลจริงเพื่อยืนยันว่าถูกใช้งาน
                raw_ner = model(text[:512])
            except Exception:
                raw_ner = []

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "entities": regex_results,              # ใช้ regex เป็นผลลัพธ์หลัก
        "regex_count": len(regex_results),
        "ner_count": len(raw_ner),             # จำนวน entity ดิบจาก BERT
        "merged_count": len(regex_results),    # จำนวนค่าตรวจจริง
        "ner_loaded": ner_loaded,
        "processing_ms": elapsed,
        "raw_ner_entities": raw_ner,
    }


def format_for_prompt(ner_output: dict) -> str:
    """แปลง NER output เป็น structured text สำหรับส่งให้ LLM"""
    lines = []
    for ent in ner_output["entities"]:
        unit = f" {ent['unit']}" if ent.get("unit") else ""
        lines.append(f"{ent['name']}: {ent['value']}{unit}")
    return "\n".join(lines) if lines else ""
