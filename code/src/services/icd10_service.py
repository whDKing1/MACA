"""
ICD-10 coding service — Automated medical code assignment.

Provides:
  - Text-to-ICD-10 code mapping
  - Code validation and hierarchy lookup
  - DRGs grouping logic based on ICD-10 + procedure codes
"""

from __future__ import annotations
import structlog

logger = structlog.get_logger(__name__)

# Comprehensive ICD-10-CM code database (subset for demonstration)
ICD10_DATABASE = {
    "A00-B99": {
        "name": "Certain infectious and parasitic diseases",
        "codes": {
            "A41.9": "Sepsis, unspecified organism",
            "A49.9": "Bacterial infection, unspecified",
            "B34.9": "Viral infection, unspecified",
        },
    },
    "C00-D49": {
        "name": "Neoplasms",
        "codes": {
            "C34.90": "Malignant neoplasm of unspecified part of bronchus or lung",
            "C50.919": "Malignant neoplasm of unspecified site of breast",
        },
    },
    "D50-D89": {
        "name": "Blood diseases",
        "codes": {
            "D64.9": "Anemia, unspecified",
            "D69.6": "Thrombocytopenia, unspecified",
        },
    },
    "E00-E89": {
        "name": "Endocrine, nutritional and metabolic diseases",
        "codes": {
            "E11.9": "Type 2 diabetes mellitus without complications",
            "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
            "E03.9": "Hypothyroidism, unspecified",
            "E78.5": "Hyperlipidemia, unspecified",
        },
    },
    "F01-F99": {
        "name": "Mental and behavioral disorders",
        "codes": {
            "F32.9": "Major depressive disorder, single episode, unspecified",
            "F41.1": "Generalized anxiety disorder",
        },
    },
    "G00-G99": {
        "name": "Nervous system diseases",
        "codes": {
            "G43.909": "Migraine, unspecified, not intractable",
            "G47.00": "Insomnia, unspecified",
        },
    },
    "I00-I99": {
        "name": "Circulatory system diseases",
        "codes": {
            "I10": "Essential (primary) hypertension",
            "I21.9": "Acute myocardial infarction, unspecified",
            "I50.9": "Heart failure, unspecified",
            "I63.9": "Cerebral infarction, unspecified",
            "I25.10": "Atherosclerotic heart disease of native coronary artery",
        },
    },
    "J00-J99": {
        "name": "Respiratory system diseases",
        "codes": {
            "J06.9": "Acute upper respiratory infection, unspecified",
            "J11.1": "Influenza with other respiratory manifestations",
            "J18.1": "Lobar pneumonia, unspecified organism",
            "J18.9": "Pneumonia, unspecified organism",
            "J44.1": "COPD with acute exacerbation",
            "J45.909": "Unspecified asthma, uncomplicated",
        },
    },
    "K00-K95": {
        "name": "Digestive system diseases",
        "codes": {
            "K21.0": "GERD with esophagitis",
            "K35.80": "Unspecified acute appendicitis",
            "K80.20": "Calculus of gallbladder without cholecystitis",
        },
    },
    "N00-N99": {
        "name": "Genitourinary system diseases",
        "codes": {
            "N39.0": "Urinary tract infection, site not specified",
            "N18.9": "Chronic kidney disease, unspecified",
        },
    },
    "U00-U85": {
        "name": "Codes for special purposes",
        "codes": {
            "U07.1": "COVID-19, virus identified",
        },
    },
}

# DRGs grouping reference (MS-DRGs, simplified)
DRG_GROUPS = {
    "J18": {"drg": "193", "desc": "Simple Pneumonia & Pleurisy w MCC", "weight": 1.4, "los": 4.5},
    "I21": {"drg": "280", "desc": "Acute Myocardial Infarction w MCC", "weight": 2.1, "los": 5.2},
    "I50": {"drg": "291", "desc": "Heart Failure & Shock w MCC", "weight": 1.6, "los": 5.0},
    "J44": {"drg": "190", "desc": "COPD w MCC", "weight": 1.3, "los": 4.0},
    "A41": {"drg": "871", "desc": "Septicemia or Severe Sepsis w MCC", "weight": 2.3, "los": 6.5},
    "E11": {"drg": "637", "desc": "Diabetes w MCC", "weight": 1.2, "los": 3.8},
    "K35": {"drg": "343", "desc": "Appendectomy w/o CC/MCC", "weight": 1.5, "los": 2.5},
    "I63": {"drg": "061", "desc": "Ischemic Stroke w Thrombolytic", "weight": 2.5, "los": 5.8},
    "N39": {"drg": "690", "desc": "Kidney & UTI w/o MCC", "weight": 0.8, "los": 3.2},
}


def lookup_icd10(code: str) -> dict | None:
    """Look up an ICD-10 code in the database."""
    for category, info in ICD10_DATABASE.items():
        if code in info["codes"]:
            return {
                "code": code,
                "description": info["codes"][code],
                "category": info["name"],
            }
    return None


def search_icd10_by_text(text: str) -> list[dict]:
    """Search ICD-10 codes by description text (simple keyword matching)."""
    text_lower = text.lower()
    results = []
    for category, info in ICD10_DATABASE.items():
        for code, desc in info["codes"].items():
            if text_lower in desc.lower():
                results.append({
                    "code": code,
                    "description": desc,
                    "category": info["name"],
                })
    return results


def get_drg_group(icd10_code: str) -> dict | None:
    """Get DRGs grouping for a given ICD-10 code prefix."""
    prefix = icd10_code.split(".")[0] if "." in icd10_code else icd10_code[:3]
    drg = DRG_GROUPS.get(prefix)
    if drg:
        return {
            "drg_code": drg["drg"],
            "description": drg["desc"],
            "weight": drg["weight"],
            "mean_los": drg["los"],
        }
    return None


def validate_icd10_code(code: str) -> bool:
    """Check if an ICD-10 code exists in our database."""
    return lookup_icd10(code) is not None
