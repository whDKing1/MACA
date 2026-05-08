"""
Treatment Agent — Evidence-based treatment recommendation.

Responsibilities:
  - Generate treatment plan based on confirmed diagnosis
  - Check drug-drug interactions (DDI)
  - Verify contraindications against patient allergies / history
  - Provide non-pharmacological treatment suggestions
  - Include evidence references for each recommendation
"""

from __future__ import annotations
import json
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from ..llm.provider import get_llm_provider

logger = structlog.get_logger(__name__)

TREATMENT_SYSTEM_PROMPT = """You are an expert clinical pharmacologist and treatment specialist. Given a patient's diagnosis and clinical data, provide a comprehensive, evidence-based treatment plan.

Return a JSON object:
{
  "diagnosis_addressed": "the primary diagnosis being treated",
  "medications": [
    {
      "drug_name": "brand name",
      "generic_name": "generic name",
      "dosage": "e.g., 500mg",
      "route": "oral|iv|im|topical|etc",
      "frequency": "e.g., twice daily",
      "duration": "e.g., 7 days",
      "contraindications": ["list any relevant"],
      "side_effects": ["common side effects"]
    }
  ],
  "drug_interactions": [
    {
      "drug_a": "drug 1",
      "drug_b": "drug 2 (can be current medication)",
      "severity": "none|minor|moderate|major|contraindicated",
      "description": "interaction details",
      "recommendation": "what to do"
    }
  ],
  "non_drug_treatments": ["physical therapy", "dietary changes", "etc."],
  "lifestyle_recommendations": ["exercise", "sleep hygiene", "etc."],
  "follow_up_plan": "when and what to check",
  "warnings": ["critical warnings for this treatment"],
  "evidence_references": ["guideline or study reference"]
}

Rules:
- ALWAYS check the patient's current medications for interactions.
- ALWAYS check allergies before recommending any drug.
- Flag any major or contraindicated interactions prominently.
- Provide at least one non-drug treatment option.
- Return ONLY valid JSON, no markdown fences."""


def treatment_agent(state) -> dict:
    """
    LangGraph 节点函数：
    根据诊断结果和患者信息生成基于证据的治疗方案，包含药物、交互检查、非药物治疗等。

    参数:
        state: 全局状态对象，应包含 patient_info 与 diagnosis 字段。

    返回:
        dict: 包含 treatment_plan、current_agent 以及可能的 errors 的部分状态更新。
    """
    logger.info("treatment_agent.start")

    diagnosis = state.diagnosis
    patient_info = state.patient_info

    if not diagnosis:
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + ["No diagnosis available for treatment planning"],
        }

    llm = get_llm_provider(temperature=0.2)

    context = json.dumps(
        {"patient_info": patient_info, "diagnosis": diagnosis},
        indent=2,
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(content=TREATMENT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical context:\n\n{context}\n\n"
                "Provide a comprehensive treatment plan with drug interaction checks."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        treatment_data = json.loads(content)

        logger.info(
            "treatment_agent.success",
            medications_count=len(treatment_data.get("medications", [])),
        )
        return {
            "treatment_plan": treatment_data,
            "current_agent": "treatment",
        }
    except json.JSONDecodeError as e:
        logger.error("treatment_agent.json_error", error=str(e))
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + [f"Treatment JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("treatment_agent.error", error=str(e))
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + [f"Treatment error: {e}"],
        }
