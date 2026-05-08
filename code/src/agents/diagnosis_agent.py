"""
Diagnosis Agent — Differential diagnosis based on structured patient data.

Responsibilities:
  - Analyze symptoms + lab results against medical knowledge
  - Generate ranked differential diagnosis list with confidence scores
  - Provide evidence chains for each candidate diagnosis
  - Recommend additional tests if information is insufficient
  - Integrates with GraphRAG knowledge graph when available
"""

from __future__ import annotations
import json
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from ..llm.provider import get_llm_provider
from ..graph.state import MAX_DIAGNOSIS_RETRIES

logger = structlog.get_logger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert diagnostician performing differential diagnosis. Given structured patient information, provide a comprehensive differential diagnosis.

Return a JSON object with this structure:
{
  "primary_diagnosis": {
    "disease_name": "most likely diagnosis",
    "icd10_hint": "approximate ICD-10 code (e.g., J18.9)",
    "confidence": 0.85,
    "evidence": ["supporting finding 1", "supporting finding 2"],
    "reasoning": "clinical reasoning explanation"
  },
  "differential_list": [
    {
      "disease_name": "alternative diagnosis",
      "icd10_hint": "ICD-10 code",
      "confidence": 0.6,
      "evidence": ["evidence 1"],
      "reasoning": "why this is considered"
    }
  ],
  "recommended_tests": ["test 1 to confirm/rule out", "test 2"],
  "clinical_notes": "overall clinical impression",
  "knowledge_sources": ["source 1", "source 2"],
  "needs_more_info": false
}

Rules:
- Confidence scores must be between 0 and 1.
- Provide at least 2-3 differential diagnoses.
- List evidence from the patient data that supports each diagnosis.
- If critical information is missing, set needs_more_info to true.
- Use standard medical terminology and ICD-10 code hints.
- Return ONLY valid JSON, no markdown fences."""


def diagnosis_agent(state) -> dict:
    """
    LangGraph node: Generate differential diagnosis from patient info.
    Reads: state.patient_info
    Writes: state.diagnosis, state.needs_more_info, state.current_agent
    """
    logger.info("diagnosis_agent.start")

    patient_info = state.patient_info
    if not patient_info:
        # patient_info 为空时，无法做诊断，必须回退。
        # 同时递增重试计数，防止因上游持续失败导致的无限回环。
        new_count = state.diagnosis_retry_count + 1
        logger.warning(
            "diagnosis_agent.no_patient_info",
            retry_count=new_count,
            max_retries=MAX_DIAGNOSIS_RETRIES,
        )
        return {
            "diagnosis": None,
            "needs_more_info": True,
            "diagnosis_retry_count": new_count,
            "current_agent": "diagnosis",
            "errors": state.errors + ["No patient info available for diagnosis"],
        }

    llm = get_llm_provider(temperature=0.2)

    # -------------------------------------------------------------------------
    # 3. 准备 LLM 输入
    # -------------------------------------------------------------------------
    # 将患者信息字典转换为格式化的 JSON 字符串，方便 LLM 阅读。
    # indent=2：添加 2 个空格的缩进，使 JSON 美观易读。
    # ensure_ascii=False：允许输出非 ASCII 字符（如中文、特殊符号），
    # 否则 JSON 会将它们转义为 \uXXXX，影响 LLM 理解。
    patient_summary = json.dumps(patient_info, indent=2, ensure_ascii=False)

    messages = [
        SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Patient information:\n\n{patient_summary}\n\nProvide your differential diagnosis."
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()


        diagnosis_data = json.loads(content)
        # 从诊断字典中提取并移除 "needs_more_info" 字段，方便单独处理。
        # pop 的第二个参数 False 是默认值，如果字段不存在则返回 False。
        needs_more = diagnosis_data.pop("needs_more_info", False)

        # 诊断回环计数器：当 LLM 判定信息不足需要回退时，计数 +1。
        # 如果 needs_more 为 False，计数器保持不变（不重置，因为同一轮 Pipeline
        # 中之前的回退记录仍有意义——路由函数需要知道累计回退了几次）。
        new_count = state.diagnosis_retry_count + 1 if needs_more else state.diagnosis_retry_count

        # 记录诊断成功日志，特别打印出首要诊断的名称，方便追踪。
        logger.info(
            "diagnosis_agent.success",
            primary=diagnosis_data.get("primary_diagnosis", {}).get("disease_name"),# 先安全地获取 primary_diagnosis 内部的 disease_name，如果不存在就用空字符串。
            needs_more_info=needs_more,
            retry_count=new_count,
        )
        return {
            "diagnosis": diagnosis_data,
            "needs_more_info": needs_more,
            "diagnosis_retry_count": new_count,
            "current_agent": "diagnosis",
        }

    # -------------------------------------------------------------------------
    # 5. 错误处理
    # -------------------------------------------------------------------------
    # 如果 LLM 返回的内容无法解析为 JSON。
    except json.JSONDecodeError as e:
        logger.error("diagnosis_agent.json_error", error=str(e))
        return {
            "diagnosis": None,
            "needs_more_info": False,
            "diagnosis_retry_count": state.diagnosis_retry_count,
            "current_agent": "diagnosis",
            "errors": state.errors + [f"Diagnosis JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("diagnosis_agent.error", error=str(e))
        return {
            "diagnosis": None,
            "needs_more_info": False,
            "diagnosis_retry_count": state.diagnosis_retry_count,
            "current_agent": "diagnosis",
            "errors": state.errors + [f"Diagnosis error: {e}"],
        }
