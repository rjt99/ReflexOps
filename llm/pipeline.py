"""
ReflexOps — LLM Agent Pipeline (ALT-002: Log Volume Anomaly)
Uses LangChain agents + LCEL for workflow orchestration.
Each agent carries a 1-shot example so the model always returns valid JSON.
"""
import json
import re
import time as _time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.config import (
    OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL,
    FALLBACK_MODEL, DEFAULT_HEADERS,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT, RETRY_MAX_WAIT,
    ACTIVE_AGENTS,
)

INCIDENT_CONTEXT = """
=== INCIDENT CONTEXT ===
System      : Production IT / order-service
Volume      : /var/log (500 GB total)
Anomaly     : write_rate_mbps spiked 2.1 → 28.3 mbps at 01:18
Trigger     : order-service v2.1.4 deployed at 01:14 (4 min before spike)
Config diff : log.level changed INFO → DEBUG in v2.1.4
Effect      : disk filling at ~40x normal rate
Downstream  : checkout-service and notification-service depend on order-service
"""

# Output schema description — no literal braces so it is safe inside any prompt
JSON_FORMAT = """
=== OUTPUT FORMAT (STRICT) ===
Reply with ONLY a valid JSON object. No markdown fences, no prose before or after.

The object has exactly FOUR top-level keys in this order:

  1. "reasoning"    — JSON array of exactly 4 strings (your step-by-step analysis).
                      CLOSE the array with ] before writing the next key.
  2. "output"       — single string: one concise sentence with the key finding or decision.
                      This key is OUTSIDE and AFTER the reasoning array — never inside it.
  3. "event_label"  — single string: 6 words or fewer, written as a headline for an event feed.
                      Include 1-2 key numbers (e.g. severity level, delta, %, confidence).
                      Example: "CRITICAL anomaly — 146σ spike detected"
  4. "crux"         — single string: 3-5 key metrics or values separated by · (middle dot).
                      These are the most important numbers from your analysis.
                      Example: "28.3 mbps · 146σ · 173 min to full · CRITICAL"

CRITICAL RULE: "output", "event_label", and "crux" are ALL sibling keys at the top level.
None of them must appear inside the "reasoning" array.
"""

# ── 1-shot examples per agent ─────────────────────────────────────────────────
# Each string is injected into the system prompt as a concrete worked example.
# The model sees exactly what a correct response looks like before it answers.

_SHOT_DETECT = '''
=== EXAMPLE (1-shot) ===
USER INPUT:
  Metric: cpu_usage_pct on api-gateway | Baseline: 18% (stddev 2%) | Current: 94% at 03:45
  Disk state: 310 GB used / 500 GB total (62%)
  Tasks: confirm anomaly, compute stddev deviation, project impact, assign severity.

CORRECT RESPONSE:
{
  "reasoning": [
    "The current cpu_usage_pct of 94% is far above the 18% baseline, a difference of 76 percentage points.",
    "With a standard deviation of 2%, this spike is 38 standard deviations above the mean, which is statistically impossible as normal noise.",
    "At this CPU saturation level, request latency will degrade within minutes and the gateway may start dropping connections.",
    "Severity is CRITICAL because the deviation is extreme, the service is customer-facing, and no immediate cause has been identified yet."
  ],
  "output": "CRITICAL anomaly confirmed — api-gateway CPU at 94% (38σ above baseline); imminent request failure risk.",
  "event_label": "CRITICAL — CPU 94% · 38σ spike",
  "crux": "94% CPU · 38σ above baseline · 62% disk · CRITICAL"
}
=== END EXAMPLE ===
'''

_SHOT_CORRELATE = '''
=== EXAMPLE (1-shot) ===
USER INPUT:
  Detect Agent confirmed: CRITICAL anomaly — api-gateway CPU at 94% (38σ above baseline).
  Event data (±15 min around anomaly at 03:45):
    03:30 : auth-service v3.0.1 deployed with new JWT validation logic
    03:31 : auth-service response time climbs from 12ms to 340ms
    03:44 : api-gateway starts logging "upstream timeout" errors at 600/min
    03:45 : api-gateway CPU hits 94%; payment-service and order-service CPU NORMAL
  Tasks: identify correlated event, compute time delta, note scope, build fingerprint.

CORRECT RESPONSE:
{
  "reasoning": [
    "auth-service v3.0.1 was deployed 15 minutes before the api-gateway CPU spike, making it the strongest temporal candidate.",
    "auth-service response time jumped from 12ms to 340ms immediately after the deploy, generating upstream timeout errors at the gateway.",
    "Only the api-gateway is CPU-saturated; payment-service and order-service are unaffected, which rules out infrastructure-wide causes.",
    "The causal fingerprint is: auth-service v3.0.1 deploy → slow JWT validation → gateway retry storms → CPU exhaustion."
  ],
  "output": "auth-service v3.0.1 deploy (03:30) correlated — slow JWT validation causing gateway retry storms; 15-min lag to CPU spike.",
  "event_label": "Deploy correlated — +15 min · 1 service affected",
  "crux": "v3.0.1 deploy · +15 min lag · 12ms→340ms latency · 600 timeouts/min"
}
=== END EXAMPLE ===
'''

_SHOT_DIAGNOSE = '''
=== EXAMPLE (1-shot) ===
USER INPUT:
  Correlate Agent output: auth-service v3.0.1 deploy correlated — slow JWT validation causing gateway retry storms.
  Correlation fingerprint:
    deploy      : auth-service v3.0.1 at 03:30 (15 min before CPU spike)
    config_diff : JWT validation rewritten; RSA key size increased 2048→4096
    effect      : auth response time 12ms → 340ms; gateway upstream timeouts 600/min
    scope       : only api-gateway CPU affected; all other services normal
  Hypotheses:
    H1 — slow_crypto_operation  (4096-bit RSA is CPU-heavy per request)
    H2 — memory_leak            (new code leaking heap, causing GC pauses)
    H3 — external_traffic_spike (sudden load increase causing timeouts)
  Tasks: score hypotheses, justify H1, rule out H2/H3, confirm root cause.

CORRECT RESPONSE:
{
  "reasoning": [
    "H1 slow_crypto_operation scores 88%: RSA-4096 requires ~4× more CPU per signing operation than RSA-2048, which directly explains the auth latency jump.",
    "H2 memory_leak scores 8%: a memory leak would cause gradual degradation over hours, not an immediate latency spike seconds after deploy.",
    "H3 external_traffic_spike scores 4%: gateway request rate metrics show no increase, and all other upstream services remain healthy.",
    "If unresolved, auth-service will continue to saturate the gateway CPU, causing request timeouts for all authenticated users within the next 20 minutes."
  ],
  "output": "Root cause confirmed: slow_crypto_operation (88% confidence) — RSA-4096 key in v3.0.1 is CPU-prohibitive at current request rate.",
  "event_label": "Root cause: slow crypto · H1 88%",
  "crux": "H1 88% · H2 8% · H3 4% · impact: auth timeouts in ~20 min"
}
=== END EXAMPLE ===
'''

_SHOT_RECOMMEND = '''
=== EXAMPLE (1-shot) ===
USER INPUT:
  Diagnose Agent output: Root cause confirmed: slow_crypto_operation (88%) — RSA-4096 key is CPU-prohibitive.
  Remediation options:
    A1 — Scale out api-gateway (add 2 instances) : buys time, does not fix root cause. Risk: LOW.
    A2 — Revert JWT key to RSA-2048 via config   : fixes root cause instantly. Risk: MEDIUM.
    A3 — Rollback auth-service to v3.0.0         : fixes root cause, reverts all v3.0.1 features. Risk: HIGH.
    A4 — Page on-call security engineer          : required for key-change approval. Risk: NONE.
  Guardrail policy: LOW → auto-execute | MEDIUM → approval | HIGH → approval + second reviewer.
  Tasks: rank by urgency × (1/risk), label AUTO-EXECUTE or REQUIRES-APPROVAL, note time window, produce action list.

CORRECT RESPONSE:
{
  "reasoning": [
    "NONE · A4 — Page on-call: zero service risk; execute immediately to bring human oversight before any key changes.",
    "LOW · A1 — Scale out gateway: auto-execute; adds 2 instances; buys ~15 min without touching root cause.",
    "MEDIUM · A2 — Revert RSA key to 2048: requires approval; stops root cause instantly; modifies live crypto config.",
    "HIGH · A3 — Rollback auth-service: requires approval + second reviewer; reverts all v3.0.1 features; use only if A2 fails within 10 min."
  ],
  "output": "4 actions ranked: A4 AUTO-EXECUTE (page on-call), A1 AUTO-EXECUTE (scale gateway), A2 REQUIRES-APPROVAL (revert RSA key), A3 REQUIRES-APPROVAL+REVIEW (rollback).",
  "event_label": "4 actions — 2 auto · 2 need approval",
  "crux": "A4 AUTO · A1 AUTO · A2 APPROVAL · A3 APPROVAL+2 · window: ~10 min",
  "risk_actions": [
    {"level": "NONE",   "action": "A4 — Page on-call engineer",    "detail": "AUTO-EXECUTE — immediate human oversight, zero service risk."},
    {"level": "LOW",    "action": "A1 — Scale out api-gateway",     "detail": "AUTO-EXECUTE — adds 2 instances, buys ~15 min, no downtime."},
    {"level": "MEDIUM", "action": "A2 — Revert RSA key to 2048",   "detail": "REQUIRES APPROVAL — stops root cause instantly; live crypto config change."},
    {"level": "HIGH",   "action": "A3 — Rollback auth-service",    "detail": "REQUIRES APPROVAL + 2nd REVIEWER — reverts all v3.0.1 features; use only if A2 fails."}
  ]
}
=== END EXAMPLE ===
'''

# Extra format instruction appended only to the Recommend agent prompt
_RECOMMEND_EXTRA = """
=== ADDITIONAL OUTPUT REQUIREMENT (Recommend Agent only) ===
Add a fifth top-level key "risk_actions" — a JSON array of exactly 4 objects, one per action.
Each object must have exactly these three string keys:
  "level"  : exactly one of "NONE", "LOW", "MEDIUM", "HIGH"
  "action" : the action code and short name (e.g. "A1 — Run log rotation")
  "detail" : one sentence — the guardrail decision (AUTO-EXECUTE or REQUIRES-APPROVAL) and why.

Each reasoning point in the "reasoning" array must begin with its risk level in the format:
  "LEVEL · Action description..."
  e.g. "LOW · A1 — Run log rotation: AUTO-EXECUTE; frees 185 GB, buys ~96 min."
"""


# ── LangChain LLM factory ──────────────────────────────────────────────────────

def _make_llm(api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        default_headers=DEFAULT_HEADERS,
    )


# ── Response parser ────────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """
    Fix the common malformation where the model places "output" inside the
    reasoning array instead of as a sibling key, e.g.:
        {"reasoning": ["a","b","c","d","output":"conclusion"}
    is repaired to:
        {"reasoning": ["a","b","c","d"],"output":"conclusion"}
    """
    # Pattern: a comma then "output": appears inside the reasoning array
    # (missing the closing ] before "output")
    repaired = re.sub(
        r',\s*"output"\s*:',
        '],"output":',
        text,
    )
    # If the original already had a proper closing ] before "output" this
    # substitution changes nothing harmful (] followed by ] is still valid
    # as long as we don't double-close — guard by removing any ]] introduced)
    repaired = re.sub(r'\]\s*\](\s*,\s*"output")', r']\1', repaired)
    return repaired


def parse_response(raw: str) -> dict:
    """
    Parse LLM reply into {chunks: list[str], output: str}.
    Tries JSON first, then a structural repair pass, then sentence-splitting.
    Raises ValueError if nothing usable is returned.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    for attempt in (cleaned, _repair_json(cleaned)):
        try:
            data   = json.loads(attempt)
            chunks = (data.get("reasoning") or data.get("steps")
                      or data.get("analysis") or data.get("observations") or [])
            output = (data.get("output") or data.get("conclusion") or data.get("summary") or "")
            if isinstance(chunks, list) and chunks:
                risk_actions = data.get("risk_actions", [])
                # Validate: must be a list of dicts with level/action/detail
                if not isinstance(risk_actions, list):
                    risk_actions = []
                risk_actions = [
                    r for r in risk_actions
                    if isinstance(r, dict) and r.get("level") and r.get("action")
                ]
                return {
                    "chunks":       [str(c) for c in chunks[:6]],
                    "output":       str(output),
                    "event_label":  str(data.get("event_label", "")),
                    "crux":         str(data.get("crux", "")),
                    "risk_actions": risk_actions,
                }
        except Exception:
            pass

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 20]
    if sentences:
        return {
            "chunks":       sentences[:-1][:5] if len(sentences) > 1 else sentences,
            "output":       sentences[-1][:200],
            "event_label":  "",
            "crux":         "",
            "risk_actions": [],
        }

    raise ValueError(f"LLM returned unusable response: {raw[:120]!r}")


# ── LangChain agent builders ───────────────────────────────────────────────────

def _build_chain(system_text: str, api_key: str, model: str):
    """Return an LCEL chain: prompt | llm | str_parser.

    SystemMessage is a pre-built object so LangChain never template-parses
    its content (which contains literal JSON braces in the 1-shot examples).
    Only the human turn uses a template variable ({user_input}).
    """
    llm = _make_llm(api_key, model)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ])
    return prompt | llm | StrOutputParser()


def _invoke_with_retry(chain, user_input: str, api_key: str, model: str) -> str:
    """Invoke chain; on rate-limit wait then retry with fallback model."""
    try:
        return chain.invoke({"user_input": user_input})
    except Exception as e:
        cause = getattr(e, "__cause__", None) or e
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if status == 429:
            retry_after = int(
                getattr(getattr(cause, "response", None), "headers", {}).get("Retry-After", 30)
            )
            _time.sleep(min(retry_after, RETRY_MAX_WAIT))
            try:
                return chain.invoke({"user_input": user_input})
            except Exception:
                fallback_chain = _build_chain(
                    chain.steps[0].messages[0].content, api_key, FALLBACK_MODEL
                )
                return fallback_chain.invoke({"user_input": user_input})
        raise


# ── Agent runners ──────────────────────────────────────────────────────────────

def run_detect(api_key: str, model: str, context: dict) -> dict:
    system = (
        "You are the Detect Agent in an autonomous IT operations system called ReflexOps.\n"
        "Your role: analyze raw telemetry to confirm whether a genuine anomaly exists and quantify its severity.\n"
        f"{INCIDENT_CONTEXT}"
        f"{JSON_FORMAT}"
        f"{_SHOT_DETECT}"
        "Now answer the real task below using the same JSON format as the example."
    )
    user = (
        "Telemetry snapshot just received:\n"
        "- Metric      : write_rate_mbps on /var/log\n"
        "- Baseline    : 2.1 mbps (6-hour rolling avg), stddev = 0.18 mbps\n"
        "- Current     : 28.3 mbps at 01:18\n"
        "- Disk state  : 207 GB used / 500 GB total (41.4%)\n\n"
        "Tasks:\n"
        "1. Confirm this is a real anomaly (not noise).\n"
        "2. Compute how many standard deviations above baseline.\n"
        "3. Project minutes until disk is full at current write rate.\n"
        "4. Assign severity: INFO / WARNING / CRITICAL."
    )
    chain = _build_chain(system, api_key, model)
    raw = _invoke_with_retry(chain, user, api_key, model)
    return parse_response(raw)


def run_correlate(api_key: str, model: str, context: dict) -> dict:
    detect_out = context.get("Detect", {}).get("output", "")
    system = (
        "You are the Correlate Agent in ReflexOps.\n"
        "Your role: given a confirmed anomaly, scan recent events to find correlated signals and build a causal fingerprint.\n"
        f"{INCIDENT_CONTEXT}"
        f"{JSON_FORMAT}"
        f"{_SHOT_CORRELATE}"
        "Now answer the real task below using the same JSON format as the example."
    )
    user = (
        f"Detect Agent confirmed: {detect_out}\n\n"
        "Available event data (±15 min around anomaly at 01:18):\n"
        "- 01:14 : order-service v2.1.4 deployed to production by ci-bot\n"
        "- 01:14 : config change — log.level: INFO → DEBUG (in v2.1.4 build)\n"
        "- 01:16 : order-service log volume starts climbing (8,400 lines/min vs baseline 1,200)\n"
        "- 01:18 : order-service log volume reaches 48,000 lines/min (40× baseline)\n"
        "- 01:18 : payment-service, auth-service, checkout-service — all log volumes NORMAL\n\n"
        "Tasks:\n"
        "1. Identify the most likely correlated event.\n"
        "2. Compute time delta between deploy and spike.\n"
        "3. Note which services are affected vs unaffected.\n"
        "4. Build a concise correlation fingerprint."
    )
    chain = _build_chain(system, api_key, model)
    raw = _invoke_with_retry(chain, user, api_key, model)
    return parse_response(raw)


def run_diagnose(api_key: str, model: str, context: dict) -> dict:
    correlate_out = context.get("Correlate", {}).get("output", "")
    system = (
        "You are the Diagnose Agent in ReflexOps.\n"
        "Your role: reason over the correlation fingerprint to identify the root cause with confidence scores.\n"
        f"{INCIDENT_CONTEXT}"
        f"{JSON_FORMAT}"
        f"{_SHOT_DIAGNOSE}"
        "Now answer the real task below using the same JSON format as the example."
    )
    user = (
        f"Correlate Agent output: {correlate_out}\n\n"
        "Correlation fingerprint:\n"
        "  deploy      : order-service v2.1.4 at 01:14 (4 min before spike)\n"
        "  config_diff : log.level INFO → DEBUG\n"
        "  volume_ratio: 40× above baseline (48,000 vs 1,200 lines/min)\n"
        "  scope       : only order-service affected — all others normal\n\n"
        "Hypotheses to score:\n"
        "  H1 — debug_log_in_production  (deploy changed log.level to DEBUG)\n"
        "  H2 — connection_pool_leak     (unbounded connections causing log flood)\n"
        "  H3 — traffic_spike_amplified  (external load causing extra log lines)\n\n"
        "Tasks:\n"
        "1. Score each hypothesis with confidence % (must sum to ~100%).\n"
        "2. Explain why H1 is dominant or not.\n"
        "3. State what evidence rules out H2 and H3.\n"
        "4. Confirm the root cause and estimate impact if unresolved."
    )
    chain = _build_chain(system, api_key, model)
    raw = _invoke_with_retry(chain, user, api_key, model)
    return parse_response(raw)


def run_recommend(api_key: str, model: str, context: dict) -> dict:
    diagnose_out = context.get("Diagnose", {}).get("output", "")
    system = (
        "You are the Recommend Agent in ReflexOps.\n"
        "Your role: translate the root cause diagnosis into a ranked action plan with risk ratings and guardrail decisions.\n"
        f"{INCIDENT_CONTEXT}"
        f"{JSON_FORMAT}"
        f"{_RECOMMEND_EXTRA}"
        f"{_SHOT_RECOMMEND}"
        "Now answer the real task below using the same JSON format as the example."
    )
    user = (
        f"Diagnose Agent output: {diagnose_out}\n\n"
        "Available remediation options:\n"
        "  A1 — Run log rotation on /var/log     : frees ~185 GB, buys ~96 min. Risk: LOW.\n"
        "  A2 — Change log.level to INFO via API : stops root cause instantly. Risk: MEDIUM (live config push).\n"
        "  A3 — Rollback order-service to v2.1.3 : stops root cause, reverts feature. Risk: HIGH (affects checkout + notifications).\n"
        "  A4 — Page on-call engineer            : human review. Risk: NONE.\n\n"
        "Guardrail policy:\n"
        "  LOW risk    → auto-execute allowed\n"
        "  MEDIUM risk → require human approval\n"
        "  HIGH risk   → require human approval + second reviewer\n\n"
        "Tasks:\n"
        "1. Rank actions by urgency × (1/risk).\n"
        "2. Mark each as AUTO-EXECUTE or REQUIRES-APPROVAL.\n"
        "3. Note time window before disk fills (~173 min from anomaly time).\n"
        "4. Produce a clean action list."
    )
    chain = _build_chain(system, api_key, model)
    raw = _invoke_with_retry(chain, user, api_key, model)
    return parse_response(raw)


# ── Registry ───────────────────────────────────────────────────────────────────

_RUNNERS = {
    "Detect":    run_detect,
    "Correlate": run_correlate,
    "Diagnose":  run_diagnose,
    "Recommend": run_recommend,
}


def run_agent(name: str, api_key: str, model: str, context: dict) -> dict:
    """Dispatch to the correct LangChain agent runner. Raises on unknown agent or LLM failure."""
    runner = _RUNNERS.get(name)
    if not runner:
        raise ValueError(f"Unknown agent: {name}")
    return runner(api_key, model, context)


def get_api_key() -> str:
    return OPENROUTER_API_KEY


def get_model() -> str:
    return OPENROUTER_MODEL
