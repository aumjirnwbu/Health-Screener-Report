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
    r"([A-Za-zÀ-ÿก-๙0-9\-\s\(\)\/]{2,60}?)"  # ดึงชื่อยาวๆ รวมวงเล็บ
    r"(?:[:\-=]\s*|\s+)"                     # ตัวคั่น (:, -, =, หรือแค่เว้นวรรค)
    r"([\d]+(?:[.,]\d+)?(?:\/\d+)?)"         # ตัวเลข (ทศนิยม หรือ / แบบความดัน)
    r"(?:\s*([A-Za-zก-๙\/\%]+))?",            # หน่วย (ถ้ามี)
    re.IGNORECASE,
)

def extract_with_regex(text: str) -> list[dict]:
    results = []
    # ประมวลผลทีละบรรทัด ป้องกันตัวเลขกระโดดข้ามบรรทัด
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        match = _VALUE_RE.search(line)
        if match:
            name_raw = match.group(1).strip()
            val_raw = match.group(2).strip()
            unit_raw = match.group(3).strip() if match.group(3) else ""
            
            canon = _canonicalize(name_raw)
            if canon:
                results.append({
                    "name": canon,
                    "value": val_raw,
                    "unit": unit_raw,
                    "source": "regex"
                })
    return results


def _canonicalize(raw_name: str) -> str:
    text = raw_name.lower().strip()
    
    # 1. ลบข้อความในวงเล็บทิ้งไปเลย ป้องกันความสับสน (เช่น (FPG), (Blood))
    text = re.sub(r'\(.*?\)', '', text).strip()
    
    # 2. ดักจับคำเฉพาะที่มักจะโดนดึงผิด (เช็ก Substring)
    if "hba1c" in text or "a1c" in text: return "HbA1c"
    if "hdl" in text: return "HDL"
    if "ldl" in text: return "LDL"
    if "fbs" in text or "fasting" in text or "fpg" in text: return "FBS"
    if "bmi" in text or "mass index" in text: return "BMI"
    if "pressure" in text: return "Blood Pressure"
    if "creatinine" in text: return "Creatinine"
    
    # 3. วนลูปเช็ก Exact Match แบบปกติ
    for canon, aliases in LAB_KEYWORDS.items():
        if text in aliases:
            return canon
            
    # 4. วนลูปเช็ก Substring (กรณีพิมพ์ชื่อยาวเกิน)
    for canon, aliases in LAB_KEYWORDS.items():
        for alias in aliases:
            if len(alias) >= 3 and alias in text:
                return canon
                
    return ""


# ─── NER-based extraction ─────────────────────────────────────────────────────
def extract_with_ner(text: str, ner_model) -> tuple[list[dict], list[dict]]:
    if ner_model is None:
        return [], []

    try:
        raw_entities = ner_model(text)
    except Exception:
        return [], []

    # 1. รัน Regex ก่อนเลย เพราะแม่นยำกว่าในรูปแบบข้อความบรรทัดๆ
    regex_results = extract_with_regex(text)
    final_dict = {item["name"]: item for item in regex_results}

    # 2. ให้ NER เก็บตกตัวที่ Regex อาจจะหลุด (เช่น ข้อความแบบร่ายยาว)
    MEDICAL_LABELS = {"Disease", "Chemical", "Diagnostic_procedure", "Medication", "Lab_value"}
    
    for ent in raw_entities:
        word = ent["word"].replace("##", "")
        label = ent.get("entity_group", ent.get("entity", ""))
        
        if label in MEDICAL_LABELS and len(word) >= 2:
            canon = _canonicalize(word)
            
            # ถ้า NER เจอชื่อมาตรฐานที่ Regex หา "ไม่เจอ" ค่อยพยายามจับคู่ตัวเลขให้
            if canon and canon not in final_dict:
                # ลองหาตัวเลขที่อยู่ใกล้ๆ (Proximity Logic) ภายในระยะ 30 ตัวอักษร
                # อัปเดต Regex ตรงนี้ให้รองรับตัวเลขแบบความดัน (เช่น 180/100) ด้วย
                text_after_entity = text[ent.get("end", 0):ent.get("end", 0)+30]
                number_match = re.search(r"([\d]+(?:[.,]\d+)?(?:\/\d+)?)", text_after_entity)
                
                if number_match:
                    final_dict[canon] = {
                        "name": canon,
                        "value": number_match.group(1),
                        "unit": "",
                        "ner_score": round(ent["score"], 4),
                        "ner_label": label,
                        "source": "ner" # บอกให้รู้ว่าดึงมาจาก NER + Proximity
                    }

    # คืนค่าผลลัพธ์รอบเดียวจบตรงนี้ โค้ดขยะด้านล่างถูกเคลียร์ทิ้งหมดแล้ว
    return list(final_dict.values()), raw_entities


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
