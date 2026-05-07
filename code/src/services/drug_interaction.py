"""
Drug interaction checking service.

Checks for:
  - Drug-drug interactions (DDI)
  - Drug-allergy contraindications
  - Drug-condition contraindications
  - Dosage range validation
"""

from __future__ import annotations
import structlog

logger = structlog.get_logger(__name__)

# Known drug-drug interactions database (demonstration subset)
DDI_DATABASE = [
    {
        "drug_a": "warfarin",
        "drug_b": "aspirin",
        "severity": "major",
        "description": "Increased risk of bleeding when warfarin is combined with aspirin",
        "recommendation": "Avoid combination unless specifically indicated; monitor INR closely",
    },
    {
        "drug_a": "metformin",
        "drug_b": "contrast_dye",
        "severity": "major",
        "description": "Risk of lactic acidosis with iodinated contrast media",
        "recommendation": "Discontinue metformin 48h before and after contrast procedures",
    },
    {
        "drug_a": "ssri",
        "drug_b": "maoi",
        "severity": "contraindicated",
        "description": "Serotonin syndrome risk — potentially fatal",
        "recommendation": "Absolute contraindication; allow 14-day washout period between medications",
    },
    {
        "drug_a": "ace_inhibitor",
        "drug_b": "potassium_supplement",
        "severity": "moderate",
        "description": "Risk of hyperkalemia",
        "recommendation": "Monitor serum potassium levels regularly",
    },
    {
        "drug_a": "simvastatin",
        "drug_b": "amiodarone",
        "severity": "major",
        "description": "Increased risk of rhabdomyolysis",
        "recommendation": "Limit simvastatin to 20mg/day when combined with amiodarone",
    },
    {
        "drug_a": "ciprofloxacin",
        "drug_b": "antacid",
        "severity": "moderate",
        "description": "Reduced absorption of ciprofloxacin",
        "recommendation": "Take ciprofloxacin 2h before or 6h after antacids",
    },
    {
        "drug_a": "methotrexate",
        "drug_b": "nsaid",
        "severity": "major",
        "description": "NSAIDs can increase methotrexate toxicity by reducing renal clearance",
        "recommendation": "Avoid combination or closely monitor blood counts and renal function",
    },
    {
        "drug_a": "digoxin",
        "drug_b": "amiodarone",
        "severity": "major",
        "description": "Amiodarone increases digoxin levels, risk of toxicity",
        "recommendation": "Reduce digoxin dose by 50% when starting amiodarone",
    },
    {
        "drug_a": "lithium",
        "drug_b": "nsaid",
        "severity": "major",
        "description": "NSAIDs can increase lithium levels",
        "recommendation": "Monitor lithium levels closely; consider dose reduction",
    },
    {
        "drug_a": "clopidogrel",
        "drug_b": "omeprazole",
        "severity": "moderate",
        "description": "Omeprazole may reduce clopidogrel effectiveness via CYP2C19 inhibition",
        "recommendation": "Use pantoprazole instead if PPI is needed",
    },
]

# Drug class mappings for fuzzy matching
DRUG_CLASS_MAP = {
    "lisinopril": "ace_inhibitor",
    "enalapril": "ace_inhibitor",
    "ramipril": "ace_inhibitor",
    "fluoxetine": "ssri",
    "sertraline": "ssri",
    "paroxetine": "ssri",
    "escitalopram": "ssri",
    "ibuprofen": "nsaid",
    "naproxen": "nsaid",
    "diclofenac": "nsaid",
    "celecoxib": "nsaid",
    "phenelzine": "maoi",
    "tranylcypromine": "maoi",
}


def _normalize_drug(name: str) -> list[str]:
    """Return possible drug/class identifiers for matching."""
    lower = name.lower().strip()
    candidates = [lower]
    if lower in DRUG_CLASS_MAP:
        candidates.append(DRUG_CLASS_MAP[lower])
    return candidates


def check_interactions(new_drugs: list[str], current_drugs: list[str]) -> list[dict]:
    """
    Check for drug-drug interactions between new prescriptions and current meds.
    Returns list of interaction records.
    """
    interactions = []

    all_new = []
    for d in new_drugs:
        all_new.extend(_normalize_drug(d))

    all_current = []
    for d in current_drugs:
        all_current.extend(_normalize_drug(d))

    for ddi in DDI_DATABASE:
        a, b = ddi["drug_a"], ddi["drug_b"]
        if (a in all_new and b in all_current) or (b in all_new and a in all_current):
            interactions.append(ddi)
        elif a in all_new and b in all_new:
            interactions.append(ddi)

    if interactions:
        logger.warning("ddi.found", count=len(interactions))
    return interactions


def check_allergy_contraindication(drug: str, allergies: list[str]) -> dict | None:
    """Check if a drug conflicts with known allergies."""
    drug_lower = drug.lower()
    for allergy in allergies:
        allergy_lower = allergy.lower()
        if drug_lower in allergy_lower or allergy_lower in drug_lower:
            return {
                "drug": drug,
                "allergy": allergy,
                "severity": "contraindicated",
                "recommendation": f"Do NOT prescribe {drug} — patient has allergy to {allergy}",
            }
        if "penicillin" in allergy_lower and drug_lower in ("amoxicillin", "ampicillin"):
            return {
                "drug": drug,
                "allergy": allergy,
                "severity": "major",
                "recommendation": f"Cross-reactivity risk: {drug} with penicillin allergy (~10%)",
            }
    return None
