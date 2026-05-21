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
    "WBC": ["wbc", "white blood cell", "เม็ดเลือดขาว", "wbc count"],
    "RBC": ["rbc", "red blood cell", "เม็ดเลือดแดง", "rbc count"],
    "HGB": ["hgb", "hb", "hemoglobin", "ฮีโมโกลบิน", "ความเข้มข้นเลือด"],
    "HCT": ["hct", "hematocrit", "ความเข้มข้นของเลือด"],
    "PLT": ["plt", "platelet", "เกล็ดเลือด", "platelet count"],
    "MCV": ["mcv"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    
    # Lipid
    "Total Cholesterol": ["total cholesterol", "cholesterol", "โคเลสเตอรอล", "คอเลสเตอรอล", "tc"],
    "LDL": ["ldl", "ldl-c", "ldl c", "ไขมันเลว", "ldl-cholesterol"],
    "HDL": ["hdl", "hdl-c", "hdl c", "ไขมันดี", "hdl-cholesterol"],
    "Triglycerides": ["tg", "triglyceride", "ไตรกลีเซอไรด์", "triglycerides"],
    
    # Glucose
    "FBS": ["fbs", "fasting blood sugar", "fasting glucose", "น้ำตาลอดอาหาร", "น้ำตาลในเลือด"],
    "HbA1c": ["hba1c", "a1c", "glycated hemoglobin", "น้ำตาลสะสม", "น้ำตาลเฉลี่ยสะสม"],
    
    # Liver
    "AST": ["ast", "sgot", "ast (sgot)"],
    "ALT": ["alt", "sgpt", "alt (sgpt)"],
    "ALP": ["alp", "alkaline phosphatase"],
    "Total Bilirubin": ["total bilirubin", "t.bili", "t-bili"],
    "Albumin": ["albumin", "alb", "อัลบูมิน"], # เพิ่มเติม: โปรตีนตับ/เลือด
    
    # Kidney
    "Creatinine": ["creatinine", "cr", "ครีเอทินิน", "b-creatinine"],
    "BUN": ["bun", "blood urea nitrogen"],
    "eGFR": ["egfr", "gfr", "estimated gfr"],
    "Uric Acid": ["uric acid", "ua", "กรดยูริก"],
    "Creatinine": ["creatinine", "cr", "ครีเอทินิน", "b-creatinine", "creatinine (blood)"],
    
    # ➕ เพิ่มกลุ่ม Electrolyte (เกลือแร่)
    "Sodium": ["sodium", "na", "โซเดียม"],
    "Potassium": ["potassium", "k", "โพแทสเซียม"],
    "Chloride": ["chloride", "cl", "คลอไรด์"],
    "Bicarbonate": ["bicarbonate", "co2", "total co2", "hco3"],

    # Thyroid
    "TSH": ["tsh"],
    "T3": ["t3", "triiodothyronine", "free t3", "ft3"],
    "T4": ["t4", "thyroxine", "free t4", "ft4"],
    
    # Others
    "CRP": ["crp", "c-reactive protein"],
    "Vitamin D": ["vitamin d", "25-oh", "25oh", "25-hydroxyvitamin d"],
    "Iron": ["iron", "serum iron"],

    # ในหมวด Glucose (อัปเดต FBS และ HbA1c)
    "FBS": ["fbs", "fasting blood sugar", "fasting glucose", "น้ำตาลอดอาหาร", "น้ำตาลในเลือด", "fasting blood glucose", "fpg"],
    "HbA1c": ["hba1c", "a1c", "glycated hemoglobin", "น้ำตาลสะสม", "น้ำตาลเฉลี่ยสะสม"],

    # ในหมวด Lipid (อัปเดต LDL ให้ครอบคลุมคำว่า Direct)
    "LDL": ["ldl", "ldl-c", "ldl c", "ไขมันเลว", "ldl-cholesterol", "ldl cholesterol"],

    # ในหมวด Kidney (อัปเดต Creatinine)
    "Creatinine": ["creatinine", "cr", "ครีเอทินิน", "b-creatinine", "creatinine (blood)"],

    # ค่าทางกายภาพและความดัน 
    "BMI": ["bmi", "body mass index", "ดัชนีมวลกาย"],
    "Blood Pressure": ["blood pressure", "bp", "ความดัน", "ความดันโลหิต"],

    # เพิ่มกลุ่มความดันโลหิต (Blood Pressure) 
    "Blood Pressure": ["blood pressure", "bp", "ความดัน", "ความดันโลหิต", "sys/dia"],
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
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=-0,          # ⚠️ ตรงนี้ให้ใช้เป็น -1 สำหรับ CPU หรือ 0 สำหรับ GPU ครับ (-0 อาจทำให้บาง Library สับสนได้)
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
    # ✅ เพิ่ม \(\) เข้าไปเพื่อให้รองรับวงเล็บในชื่อ และเพิ่มความยาวเป็น 60 เผื่อชื่อยาว
    r"([A-Za-zÀ-ÿก-๙][A-Za-zÀ-ÿก-๙0-9\-\s\.\(\)\/]{1,60}?)"
    r"\s*(?:[:\-=]\s*|\s+)"
    # ✅ เพิ่ม (?:\/\d+)? เข้าไปด้านหลัง เพื่อให้ดึงตัวเลขแบบ 180/100 (ความดัน) ได้
    r"([\d]+(?:[.,]\d+)?(?:\/\d+)?)"
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
    รัน Biomedical NER แล้วใช้เทคนิคระยะห่างอักษร (Proximity Mapping) 
    ในการควานหาตัวเลขที่อยู่ใกล้ชื่อแล็บที่สุด เพื่อป้องกันการแมตช์ชื่อพลาด
    """
    if ner_model is None:
        return [], []

    try:
        raw_entities = ner_model(text)
    except Exception:
        return [], []

    MEDICAL_LABELS = {"Disease", "Chemical", "Diagnostic_procedure", "Medication", "Lab_value"}
    found_names = []
    
    for ent in raw_entities:
        word = ent["word"].replace("##", "")
        label = ent.get("entity_group", ent.get("entity", ""))
        if label in MEDICAL_LABELS and len(word) >= 2:
            canonical = _canonicalize(word)
            if canonical:
                found_names.append({
                    "canonical": canonical,
                    "start": ent.get("start", 0),
                    "end": ent.get("end", len(text)),
                    "score": round(ent["score"], 4),
                    "label": label,
                })

    # ดึงตัวเลขและหน่วยทั้งหมดในข้อความออกมากางไว้ก่อน
    # Regex ตัวนี้จะโฟกัสจับเฉพาะก้อนตัวเลข [ขอบเขตสั้นๆ] เพื่อเอาไปผูกกับชื่อแล็บด้านบน
    number_pattern = re.compile(r"([\d]+(?:[.,]\d+)?)\s*(" + "|".join(UNIT_PATTERNS) + r")?", re.IGNORECASE)
    all_numbers = []
    for num_match in number_pattern.finditer(text):
        all_numbers.append({
            "value": num_match.group(1).replace(",", "."),
            "unit": num_match.group(2) or "",
            "start": num_match.start(),
        })

    lab_entities = []
    for fn in found_names:
        best_value = "—"
        best_unit = ""
        min_distance = 999999
        
        # วิ่งหาตัวเลขที่อยู่ "เยื้องหลัง" ชื่อแล็บตัวนั้นๆ ในระยะที่ใกล้ที่สุด
        for num in all_numbers:
            # ตัวเลขควรอยู่หลังชื่อแล็บ (num['start'] >= fn['end']) 
            if num["start"] >= fn["start"]:
                distance = num["start"] - fn["end"]
                if distance < min_distance and distance < 30: # ระยะห่างไม่ควรเกิน 30 ตัวอักษร
                    min_distance = distance
                    best_value = num["value"]
                    best_unit = num["unit"]

        # ถ้าหาจากตัวเลขใกล้เคียงไม่เจอจริงๆ ค่อยไป Fallback ดึงจาก Regex Map แบบเก่า
        if best_value == "—":
            regex_vals = extract_with_regex(text)
            regex_map = {v["name"]: v for v in regex_vals}
            val_info = regex_map.get(fn["canonical"], {})
            best_value = val_info.get("value", "—")
            best_unit = val_info.get("unit", "")

        lab_entities.append({
            "name":       fn["canonical"],
            "value":      best_value,
            "unit":       best_unit,
            "ner_score":  fn["score"],
            "ner_label":  fn["label"],
            "source":     "ner",
        })

    return lab_entities, raw_entities


# ─── Main public function ─────────────────────────────────────────────────────
def run_ner_extraction(text: str, use_ner: bool = True) -> dict:
    """
    รับข้อความจาก OCR และดึงค่าแล็บ
    - ใช้ Regex เป็นฐานข้อมูลหลัก 
    - ใช้ BioBERT เพื่อช่วยเก็บตกค่าแล็บที่แพทเทิร์นแปลกๆ หลุดจาก Regex
    """
    t0 = time.perf_counter()

    # 1) Regex extraction (ตัวหลัก)
    regex_results = extract_with_regex(text)

    # 2) เตรียมตัวแปรสำหรับ NER
    ner_loaded = False
    raw_ner = []
    ner_results = []

    # เปลี่ยนโครงสร้างตรงนี้: เช็กก่อนว่าผู้ใช้สั่งเปิดไหม แล้วค่อยลองดึงโมเดลมาทำงาน
    if use_ner:
        model = get_ner_model()  # 👈 ต้องดึงโมเดลผ่านฟังก์ชันนี้ขึ้นมาก่อน
        if model is not None:
            ner_loaded = True
            try:
                # รันสกัดข้อมูลด้วยโมเดลแพทย์ที่เราปรับปรุงในจุดที่ 1
                ner_results, raw_ner = extract_with_ner(text, model)
            except Exception:
                raw_ner = []
                ner_results = []
    
    # 3) รวมร่างแบบป้องกันตัวซ้ำ (หากชื่อซ้ำกัน ให้ยึดค่าแล็บจาก Regex เป็นหลัก)
    merged_entities = {v["name"]: v for v in regex_results}
    for nv in ner_results:
        if nv["name"] not in merged_entities:
            merged_entities[nv["name"]] = nv

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "entities": list(merged_entities.values()),  # ✅ รวมพลังดึงค่ามาครบถ้วนแน่นอน
        "regex_count": len(regex_results),
        "ner_count": len(ner_results), 
        "merged_count": len(merged_entities),    
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
