"""
Intake Agent — Patient information collection and structuring.

Responsibilities:
  - Parse raw patient description into structured PatientInfo
  - Extract chief complaint, symptoms, medical history
  - Normalize data into FHIR-aligned format
  - Validate completeness of critical fields
"""

from __future__ import annotations
import json
# structlog 是一个结构化日志库，比 print 更专业。
# 它可以输出 JSON 格式的日志，方便后期检索、分析。
# get_logger(__name__) 会创建一个名为当前模块路径的 logger，
# 比如 "myproject.agents.intake"。
import structlog
# langchain_core 是 LangChain 的核心消息类型。
# HumanMessage 代表用户说的话，SystemMessage 代表给模型的系统指令。
from langchain_core.messages import HumanMessage, SystemMessage
from ..llm.provider import get_llm_provider
# 从上层数据模型导入 PatientInfo 类，是一个 Pydantic BaseModel，
# 定义了患者信息的所有字段和验证规则。
from ..models.patient import PatientInfo
# 创建该模块专用的 logger，后续可以用 logger.info(...) 记录日志。
logger = structlog.get_logger(__name__)

INTAKE_SYSTEM_PROMPT = """You are an expert medical intake specialist. Your job is to extract structured patient information from the provided clinical narrative.

Extract the following fields as a JSON object:
{
  "name": "patient name or 'Unknown'",
  "age": <integer>,
  "gender": "male|female|other|unknown",
  "chief_complaint": "main reason for visit",
  "symptoms": [
    {"name": "symptom name", "duration_days": <int or null>, "severity": "mild|moderate|severe|critical", "description": "details"}
  ],
  "medical_history": ["list of past conditions"],
  "family_history": ["list of family conditions"],
  "allergies": [
    {"substance": "name", "reaction": "description", "severity": "mild|moderate|severe"}
  ],
  "current_medications": [
    {"name": "drug name", "dosage": "dose", "frequency": "how often"}
  ],
  "vital_signs": {
    "temperature": <float or null>,
    "heart_rate": <int or null>,
    "blood_pressure_systolic": <int or null>,
    "blood_pressure_diastolic": <int or null>,
    "respiratory_rate": <int or null>,
    "oxygen_saturation": <float or null>
  },
  "lab_results": [
    {"test_name": "name", "value": "result", "unit": "unit", "reference_range": "range", "is_abnormal": true/false}
  ]
}

Rules:
- If a field is not mentioned, use reasonable defaults or null.
- Age must be a positive integer. If unclear, estimate from context.
- Always identify the chief complaint even if not explicitly stated.
- Return ONLY valid JSON, no markdown fences."""


def intake_agent(state) -> dict:
    """
    LangGraph 节点函数：
    1. 从 state 中读取原始患者描述（raw_input）
    2. 调用 LLM 解析为结构化 JSON
    3. 用 PatientInfo 模型验证数据
    4. 返回更新后的 state 字典（只返回需要更新的字段）
    """

    # 记录日志：开始处理，同时记录原始输入的长度（方便观察数据量，避免日志爆屏）
    # len(state.raw_input or "") 中的 or "" 是为了防止 raw_input 为 None 时报错。
    logger.info("intake_agent.start", raw_input_len=len(state.raw_input or ""))

    # -------------------------------------------------------------------------
    # 1. 取出原始患者描述
    # -------------------------------------------------------------------------
    # state 是一个类似字典的对象（LangGraph 状态），通常通过属性访问。
    # 例如 state.raw_input 等同于 state["raw_input"]。
    raw = state.raw_input
    if not raw:
        return {
            "patient_info": None,
            "current_agent": "intake",
            "errors": state.errors + ["No raw input provided to Intake Agent"],
        }

    # -------------------------------------------------------------------------
    # 2. 初始化 LLM（大语言模型）
    # -------------------------------------------------------------------------
    # get_llm_provider() 根据环境变量 LLM_PROVIDER 自动选择对应的模型后端：
    #   openai -> GPT 系列
    #   deepseek -> DeepSeek 系列
    #   qwen -> 通义千问系列
    #   ollama -> 本地部署模型
    # temperature=0.1 表示几乎确定的输出，适合信息提取任务。
    llm = get_llm_provider(temperature=0.1)

    # -------------------------------------------------------------------------
    # 3. 构建消息列表
    # -------------------------------------------------------------------------
    # SystemMessage：定义 AI 的行为和输出格式，此处就是上方的大段提示词。
    # HumanMessage：把实际的患者描述放进去。
    messages = [
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=f"Patient narrative:\n\n{raw}"),
    ]

    # -------------------------------------------------------------------------
    # 4. 调用 LLM 并处理响应
    # -------------------------------------------------------------------------
    try:
        # llm.invoke 会发送消息给 OpenAI，并返回一个 AIMessage 对象。
        # response.content 就是模型生成的文本字符串。
        response = llm.invoke(messages)
        content = response.content.strip()# 去除首尾空白

        # 有时候 LLM 会不听话，用 ```json ... ``` 包裹 JSON，我们需要去掉这些标记。
        # 简单处理：如果内容以 ``` 开头，则去掉第一行和最后一行。
        if content.startswith("```"):
            # 用换行符分割，取第二行到倒数第二行，再重新合并。
            # 例如 content = "```json\n{...}\n```"
            # split("\n", 1) 分割一次，得到 ["```json", "{...}\n```"]
            # 取第二部分 "{...}\n```"
            # rsplit("```", 1) 从右边分割一次，得到 ["{...}\n", ""]
            # 取第一部分并 strip，得到干净的 JSON 字符串。
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # 把处理好的 JSON 字符串解析为 Python 字典。
        patient_data = json.loads(content)

        # 用 Pydantic 的 PatientInfo 模型来验证数据。
        # **patient_data 表示将字典展开为关键字参数。
        # 如果数据不符合模型定义（比如 age 是负数、缺少必填字段），会抛出 ValidationError。
        patient = PatientInfo(**patient_data)

        # 将 Pydantic 模型转为字典（mode="json" 保证所有值都是 JSON 兼容的，
        # 比如 datetime 会转成字符串，枚举会转成值等）。
        # 这么做是为了后续 LangGraph 状态能正常序列化（比如 Checkpointer）。
        patient_dict = patient.model_dump(mode="json")

        # 记录成功日志，打印患者姓名（从已验证的模型中获取，不是原始输入，更可靠）。
        logger.info("intake_agent.success", patient_name=patient.name)

        # 返回要更新的状态字段。
        # LangGraph 接收字典后，会将这些键值对合并进全局状态。
        return {
            "patient_info": patient_dict,
            "current_agent": "intake",
        }

    # -------------------------------------------------------------------------
    # 5. 错误处理
    # -------------------------------------------------------------------------
    # 如果 JSON 解析失败（LLM 返回的不是合法 JSON，即使用了解包裹也失败）
    except json.JSONDecodeError as e:
        logger.error("intake_agent.json_error", error=str(e))
        return {
            "patient_info": None,
            "current_agent": "intake",
            "errors": state.errors + [f"Intake JSON parse error: {e}"],
        }

    # 如果 Pydantic 验证失败、网络错误、模型超时等其他任何异常
    except Exception as e:
        logger.error("intake_agent.error", error=str(e))
        return {
            "patient_info": None,
            "current_agent": "intake",
            "errors": state.errors + [f"Intake error: {e}"],
        }
