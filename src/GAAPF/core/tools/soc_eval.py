from typing import Dict, Any, List
import os
import json
import re

try:
    from langchain_google_vertexai import ChatVertexAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    ChatVertexAI = None  # type: ignore

from ..utils.credentials_helper import get_vertex_ai_config


MAIEUTIC_PROMPT = (
    "You are a Socratic-behavior judge. Evaluate ONLY Socratic-ness of the Agent Reply. "
    "Do NOT grade task correctness except where it affects Socratic behaviors (e.g., revealing answers early).\n"
    "First, perform a maieutic critique: identify missing assumptions, unsupported leaps, questions that should have been asked, and any non-Socratic behavior (lecturing, giving away answers).\n"
    "Return a short critique summary."
)


DIALECTIC_PROMPT = (
    "Now produce your own brief Socratic-style reply to the same User Problem (2-6 guiding questions). "
    "Then compare with the Agent Reply: where is the agent less Socratic? What specific improvements would make it more Socratic?"
)


DEFINITION_INSTRUCTIONS = (
    "Score each dimension from 1 (poor) to 5 (excellent):\n"
    "- questioning_guidance\n"
    "- question_quality\n"
    "- scaffolding\n"
    "- avoids_direct_answers\n"
    "- reflection_encouragement\n"
    "Also compute:\n"
    "- socratic_question_ratio: 0-100, estimated % of sentences that are genuine, guiding questions.\n"
    "- overall_socratic: rounded average of the five scores.\n"
    "Return strict JSON with keys:\n"
    "{\n"
    " \"questioning_guidance\": int,\n"
    " \"question_quality\": int,\n"
    " \"scaffolding\": int,\n"
    " \"avoids_direct_answers\": int,\n"
    " \"reflection_encouragement\": int,\n"
    " \"socratic_question_ratio\": int,\n"
    " \"overall_socratic\": int,\n"
    " \"maieutic_critique\": \"string\",\n"
    " \"dialectic_diff\": \"string\",\n"
    " \"instructor_addback\": [\"short\", \"imperative\", \"bullets\"]\n"
    "}"
)


def _init_llm():
    cfg = get_vertex_ai_config()
    if ChatVertexAI is None:
        raise RuntimeError("langchain-google-vertexai is not installed; cannot initialize Gemini judge")
    judge_model = os.getenv("VERTEX_AI_JUDGE_MODEL", "gemini-2.5-pro")
    return ChatVertexAI(
        model_name=judge_model,
        temperature=0.0,
        top_p=1.0,
        project=cfg["project"],
        location=cfg["location"],
    )


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    # Try to find the last JSON object in the output
    m = re.search(r"\{[\s\S]*\}$", text.strip())
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        # Attempt to sanitize common trailing content
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {}


def judge_socratic(user_problem: str, agent_reply: str) -> Dict[str, Any]:
    """
    Reference-free Socratic behavior evaluation inspired by SocREval.
    Steps:
    1) Maieutic critique of Agent Reply relative to the problem
    2) Dialectic: generate a Socratic-style reply, compare and grade via rubric

    Returns a structured JSON dict with scores and instructor_addback bullets.
    """
    llm = _init_llm()
    system = "You are an expert Socratic evaluator. When asked for JSON, respond with strict JSON only."

    # 1) Maieutic critique (kept separate for clarity; contents used implicitly by the model in step 2)
    try:
        _ = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": f"{MAIEUTIC_PROMPT}\nUser Problem:\n{user_problem}\n\nAgent Reply:\n{agent_reply}"},
        ])
    except Exception:
        # Non-fatal; proceed to scoring
        pass

    # 2) Dialectic + Rubric scoring (JSON)
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"{DIALECTIC_PROMPT}\nUser Problem:\n{user_problem}\n\nAgent Reply:\n{agent_reply}\n\n{DEFINITION_INSTRUCTIONS}\nOutput JSON only."
        )},
    ])
    body = getattr(resp, "content", str(resp))
    data = _extract_json(body)

    # Post-process to ensure minimal keys exist
    defaults: Dict[str, Any] = {
        "questioning_guidance": 0,
        "question_quality": 0,
        "scaffolding": 0,
        "avoids_direct_answers": 0,
        "reflection_encouragement": 0,
        "socratic_question_ratio": 0,
        "overall_socratic": 0,
        "maieutic_critique": "",
        "dialectic_diff": "",
        "instructor_addback": [],
    }
    out = {**defaults, **(data or {})}

    # Clamp and sanitize values
    def _clamp_int(x: Any, lo: int = 0, hi: int = 100) -> int:
        try:
            v = int(float(x))
        except Exception:
            v = 0
        return max(lo, min(hi, v))

    for k in [
        "questioning_guidance",
        "question_quality",
        "scaffolding",
        "avoids_direct_answers",
        "reflection_encouragement",
    ]:
        out[k] = _clamp_int(out.get(k, 0), 0, 5)
    out["socratic_question_ratio"] = _clamp_int(out.get("socratic_question_ratio", 0), 0, 100)
    out["overall_socratic"] = _clamp_int(out.get("overall_socratic", 0), 0, 5)

    # Ensure list for addback
    addback = out.get("instructor_addback")
    if not isinstance(addback, list):
        addback = [str(addback)] if addback else []
    # Limit to short, imperative bullets
    clean_addback: List[str] = []
    for item in addback[:5]:
        s = str(item).strip()
        if not s:
            continue
        # Keep bullets short
        if len(s) > 160:
            s = s[:157] + "..."
        clean_addback.append(s)
    out["instructor_addback"] = clean_addback

    return out



