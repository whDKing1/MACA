"""
Coding Agent — ICD-10 automatic coding and DRGs grouping.

Responsibilities:
  - Map diagnoses to ICD-10-CM codes with high precision
  - Determine DRGs grouping based on principal diagnosis + procedures
  - Provide coding confidence scores and rationale
  - Cross-validate codes against diagnosis descriptions
"""

from __future__ import annotations
import json
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)

CODING_SYSTEM_PROMPT = """You are a certified medical coding specialist (CCS) with expertise in ICD-10-CM and DRGs grouping. Given diagnosis information and treatment details, assign accurate medical codes.

Return a JSON object:
{
  "primary_icd10": {
    "code": "exact ICD-10-CM code (e.g., J18.1)",
    "description": "official code description",
    "confidence": 0.92,
    "category": "category name"
  },
  "secondary_icd10_codes": [
    {
      "code": "ICD-10 code",
      "description": "description",
      "confidence": 0.85,
      "category": "category"
    }
  ],
  "drg_group": {
    "drg_code": "DRG number (e.g., 193)",
    "description": "DRG description",
    "weight": 1.2,
    "mean_los": 4.5
  },
  "coding_notes": "rationale for code selection",
  "coding_confidence": 0.90
}

Rules:
- Use the most specific ICD-10-CM code available (4th-7th character level).
- Primary code should match the principal diagnosis.
- Include comorbidity and complication codes as secondary.
- DRGs weight and mean length of stay should be realistic estimates.
- Confidence reflects how certain the code assignment is.
- Return ONLY valid JSON, no markdown fences."""


def coding_agent(state) -> dict:
    """
    LangGraph node: Assign ICD-10 codes and DRGs grouping.
    Reads: state.diagnosis, state.treatment_plan
    Writes: state.coding_result, state.current_agent
    """
    logger.info("coding_agent.start")

    diagnosis = state.diagnosis
    treatment = state.treatment_plan

    if not diagnosis:
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + ["No diagnosis available for coding"],
        }

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )

    context = json.dumps(
        {"diagnosis": diagnosis, "treatment_plan": treatment},
        indent=2,
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(content=CODING_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical data for coding:\n\n{context}\n\n"
                "Assign ICD-10 codes and DRGs group."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        coding_data = json.loads(content)

        logger.info(
            "coding_agent.success",
            primary_code=coding_data.get("primary_icd10", {}).get("code"),
        )
        return {
            "coding_result": coding_data,
            "current_agent": "coding",
        }
    except json.JSONDecodeError as e:
        logger.error("coding_agent.json_error", error=str(e))
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + [f"Coding JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("coding_agent.error", error=str(e))
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + [f"Coding error: {e}"],
        }
