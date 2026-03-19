from __future__ import annotations

from typing import Any, Dict, List, Literal
import re

from ai.llm_client import call_llm_structured

Decision = Literal["ACCEPT", "RETRY", "CLARIFY", "TEMPLATE"]

CONF_ACCEPT = 0.85
CONF_MIN = 0.70
MAX_RETRY = 1

VAGUE_TASK_PATTERNS = [
    r"^\s*fix( the)? issue\s*$",
    r"^\s*do project stuff\s*$",
    r"^\s*handle( it| this)?\s*$",
    r"^\s*do( it| stuff)?\s*$",
]

MULTI_ACTION_HINTS = [" and ", " then "]
VAGUE_STEP_PATTERNS = [
    r"^\s*do it\s*$",
    r"^\s*handle it\s*$",
    r"^\s*work on it\s*$",
]


def select_prompts(task_type: str, mode: str, strict: bool) -> Dict[str, str]:
    rules = [
        "You are an AI Task Simplifier.",
        "Break the user's task into clear, numbered steps.",
        "Each step must contain only one action.",
        "Use simple, direct language.",
        "Return only valid JSON that matches the required schema.",
        "If the task is unclear, set clarification_needed=true and ask one clarification question.",
        "Lower confidence_score if you are uncertain.",
    ]

    m = (mode or "Standard").lower()
    if m == "simplified":
        rules += [
            "Use very simple words.",
            "Use short instructions.",
            "Prefer 3 to 6 steps.",
        ]
    elif m == "voice-first":
        rules += [
            "Use short spoken-friendly instructions.",
            "Do not refer to visual elements.",
        ]
    elif m == "visual-assist":
        rules += [
            "Make steps easy to scan like a checklist.",
        ]
    elif m == "assistive":
        rules += [
            "Use supportive language.",
            "Avoid long or complex sentences.",
        ]

    if strict:
        rules += [
            "Be stricter than usual.",
            "Do not combine actions in one step.",
            "If outcome or format is unclear, ask for clarification.",
        ]

    system_prompt = "\n".join(rules) + f"\nTask type: {task_type}."
    user_prompt = "User task:\n{{TASK_TEXT}}"
    return {"system": system_prompt, "user": user_prompt}


def is_task_vague(task_text: str) -> bool:
    t = (task_text or "").strip().lower()
    if len(t) < 10:
        return True
    return any(re.match(p, t) for p in VAGUE_TASK_PATTERNS)


def steps_sequential(steps: List[Dict[str, Any]]) -> bool:
    nums = [s.get("step_number") for s in steps]
    return nums == list(range(1, len(nums) + 1))


def one_action_per_step(steps: List[Dict[str, Any]]) -> bool:
    for s in steps:
        instr = str(s.get("instruction", "")).lower()
        if any(h in instr for h in MULTI_ACTION_HINTS):
            return False
    return True


def vague_steps(steps: List[Dict[str, Any]]) -> bool:
    for s in steps:
        instr = str(s.get("instruction", "")).strip().lower()
        if any(re.match(p, instr) for p in VAGUE_STEP_PATTERNS):
            return True
        if len(instr) < 6:
            return True
    return False


def basic_relevance_check(task_text: str, steps: List[Dict[str, Any]], task_type: str) -> bool:
    task = (task_text or "").lower()
    step_text = " ".join(str(s.get("instruction", "")) for s in steps).lower()

    keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", task)][:6]
    overlap = sum(1 for k in keywords if k in step_text)

    if task_type.lower() == "reporting":
        if all(x not in step_text for x in ["report", "summary", "template"]):
            return False

    return overlap >= 1


def validate(task_text: str, task_type: str, resp: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    steps = resp.get("simplified_steps", [])
    conf = float(resp.get("confidence_score", 0.0))

    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("No steps returned")

    if conf < CONF_MIN:
        errors.append(f"Low confidence: {conf:.2f}")

    if steps:
        if not steps_sequential(steps):
            errors.append("Steps not sequential")
        if not one_action_per_step(steps):
            errors.append("Multiple actions detected in a step")
        if vague_steps(steps):
            errors.append("Vague or non-actionable steps detected")
        if not basic_relevance_check(task_text, steps, task_type):
            errors.append("Relevance check failed")

    return {"passed": len(errors) == 0, "errors": errors}


def clarification_response(task_id: str, message: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "CLARIFY",
        "confidence_score": 0.0,
        "reasons": ["Clarification required"],
        "simplified_steps": [],
        "fallback": {
            "type": "CLARIFICATION",
            "message": message,
            "template_steps": [],
        },
        "telemetry": {},
    }


def generic_template(task_type: str) -> List[str]:
    if task_type.lower() == "reporting":
        return [
            "Collect the needed information.",
            "Fill the report template.",
            "Write a short summary.",
            "Review for mistakes.",
            "Submit the report.",
        ]
    if task_type.lower() == "technical":
        return [
            "Describe the problem clearly.",
            "Check recent changes.",
            "Try the simplest fix first.",
            "Record any error details.",
            "Escalate if unresolved.",
        ]
    return [
        "Define the goal.",
        "List what is needed.",
        "Do the first small step.",
        "Check progress.",
        "Finish and confirm completion.",
    ]


def template_response(task_id: str, message: str, template_steps: List[str]) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "TEMPLATE",
        "confidence_score": 0.0,
        "reasons": ["Template fallback used"],
        "simplified_steps": [],
        "fallback": {
            "type": "TEMPLATE",
            "message": message,
            "template_steps": [
                {"step_number": i + 1, "instruction": s}
                for i, s in enumerate(template_steps)
            ],
        },
        "telemetry": {},
    }


def simplify_task(request: Dict[str, Any]) -> Dict[str, Any]:
    task_id = request.get("task_id", "")
    task_text = request.get("task_text", "")
    task_type = request.get("task_type", "Unknown")
    mode = request.get("accessibility_mode", "Standard")
    model = request.get("model") or "gemini-2.5-flash"

    if is_task_vague(task_text):
        return clarification_response(
            task_id,
            "Please provide more detail about the task outcome, deadline, or required format.",
        )

    attempt = 0
    last_errors: List[str] = []

    while attempt <= MAX_RETRY:
        strict = attempt == 1
        prompts = select_prompts(task_type, mode, strict=strict)

        system_prompt = prompts["system"]
        user_prompt = prompts["user"].replace("{{TASK_TEXT}}", task_text)

        try:
            resp = call_llm_structured(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2 if not strict else 0.0,
            )
        except Exception:
            if attempt < MAX_RETRY:
                attempt += 1
                continue
            return template_response(
                task_id,
                "Gemini service unavailable. Use this starter checklist.",
                generic_template(task_type),
            )

        resp["task_id"] = task_id
        validation = validate(task_text, task_type, resp)
        last_errors = validation["errors"]

        if validation["passed"]:
            conf = float(resp["confidence_score"])
            return {
                "task_id": task_id,
                "status": "ACCEPT" if conf >= CONF_ACCEPT else "ACCEPT",
                "confidence_score": conf,
                "reasons": [],
                "simplified_steps": resp["simplified_steps"],
                "fallback": {
                    "type": "NONE",
                    "message": "",
                    "template_steps": [],
                },
                "telemetry": {
                    "prompt_version": "v1.0",
                    "attempt": attempt + 1,
                    "validation_passed": True,
                },
            }

        if attempt < MAX_RETRY:
            attempt += 1
            continue

        if any("Relevance check failed" in e for e in last_errors) or any(
            "vague" in e.lower() for e in last_errors
        ):
            return clarification_response(
                task_id,
                "What is the expected output and who is it for?",
            )

        return template_response(
            task_id,
            "The task could not be simplified reliably. Use this starter checklist.",
            generic_template(task_type),
        )

    return template_response(
        task_id,
        "The task could not be processed. Use this starter checklist.",
        generic_template(task_type),
