# ReflexOps — Project Summary

---

## Problem Statement

Production engineering teams face a class of incidents where a single misconfiguration — such as a debug-logging flag accidentally enabled in a deployment — goes undetected for several minutes and silently consumes shared infrastructure resources. Without automated detection and correlation, on-call engineers are paged only after a threshold breach, by which point the incident has often cascaded into downstream service failures. In the modelled scenario, an `order-service` deployment with `log.level=DEBUG` caused disk write rate to spike 40× above baseline, filling a 500 GB volume in under three hours and crashing dependent services — resulting in 103 minutes of customer-facing downtime.

---

## Target Stakeholders

| Persona | Role |
|---|---|
| Site Reliability Engineers (SREs) | Primary users — receive agent recommendations and approve remediation actions |
| Platform / DevOps Engineers | Operators responsible for deployment pipelines and infrastructure health |
| Engineering Managers | Consumers of post-incident summaries and MTTR metrics |
| CTO / VP Engineering | Strategic audience for demonstrating AI-driven operational efficiency |

---

## Why This Problem Matters — Operational & Business Relevance

- **MTTR cost**: Every minute of P1 downtime in a customer-facing service has direct revenue and reputation impact. The average cost of unplanned downtime across industries exceeds $5,000 per minute.
- **Alert fatigue**: Traditional threshold-based alerting fires after the fact. Engineers spend the majority of incident time correlating logs, deployments, and metrics manually — work that is repetitive and time-sensitive.
- **Skill concentration risk**: Root-cause analysis depends on tribal knowledge. Automating the detection-to-recommendation chain democratises that knowledge and reduces dependency on individual experts.
- **Scale gap**: As microservice counts grow, the number of observable signals grows faster than teams can staff. Autonomous agents provide a scalable first-responder layer.

---

## Solution Workflow

```
Production Telemetry
   │  (disk_metrics, log_volume, deployment_events, service_map)
   ▼
Simulation Engine ──► Anomaly Trigger (write_rate spike at 01:18)
                              │
                              ▼
                    ┌─────────────────────┐
                    │   DETECT AGENT 🔍   │  Confirms anomaly, quantifies σ deviation,
                    │                     │  projects time-to-full, assigns severity
                    └────────┬────────────┘
                             │  output passed as context
                             ▼
                    ┌─────────────────────┐
                    │ CORRELATE AGENT 🔗  │  Scans event window, links deployment
                    │                     │  to spike, builds causal fingerprint
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  DIAGNOSE AGENT 🧠  │  Scores hypotheses with confidence %,
                    │                     │  eliminates alternates, confirms root cause
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │ RECOMMEND AGENT 💡  │  Ranks remediation actions by urgency × (1/risk),
                    │                     │  applies guardrail policy, outputs risk-action grid
                    └─────────────────────┘
                             │
                             ▼
                    Dashboard UI ──► Agent Reasoning panel · Event Feed · Risk Action Grid
```

---

## AI Approach

| Technique | Applied |
|---|---|
| **Multi-Agent LLM Pipeline** | Sequential 4-agent chain — each agent receives the prior agent's output as context, enabling progressive reasoning from raw telemetry to ranked remediation |
| **1-Shot Prompting** | Each agent's system prompt includes one complete worked example from a different incident domain, anchoring output structure without overfitting to the target scenario |
| **Structured JSON Output Enforcement** | Agents are constrained to return a strict JSON schema (`reasoning`, `output`, `event_label`, `crux`, `risk_actions`) with an automated repair pass for common malformations |
| **Context Chaining** | Each agent's `output` field is passed into the next agent's user prompt, building a causal chain: Detection → Correlation → Diagnosis → Recommendation |
| **Guardrail Policy Encoding** | Risk levels (NONE / LOW / MEDIUM / HIGH) are embedded in the prompt and enforced in the output schema — the model decides AUTO-EXECUTE vs REQUIRES-APPROVAL per action |
| **RAG** | Not used — incident context is injected directly as structured prompt data |
| **Speech / Vision** | Not used |

---

## Key Technologies & Frameworks

| Layer | Technology |
|---|---|
| **LLM Orchestration** | LangChain (LCEL) — `ChatOpenAI`, `ChatPromptTemplate`, `StrOutputParser`, `SystemMessage` |
| **LLM Provider** | OpenRouter API (OpenAI-compatible, model-agnostic routing) |
| **UI Framework** | Streamlit — single-page app with custom CSS, real-time rerun loop |
| **Visualisation** | Plotly — interactive time-series and bar charts |
| **Data** | Python / Pandas, JSON static datasources |
| **Configuration** | python-dotenv, centralised `llm/config.py` |
| **Language** | Python 3.10+ |

---

## What Was Built

- **Autonomous Incident Response Pipeline** — a 4-agent LangChain chain that detects, correlates, diagnoses, and recommends remediation for a production disk-exhaustion incident, triggered automatically by anomaly detection in a simulated telemetry stream.
- **ReflexOps Dashboard** — a Streamlit-based operations console with:
  - Live metric cards, disk usage chart, log volume chart
  - Agent Activity panel with typewriter-style reasoning stream
  - Risk Action Grid — 4 colour-coded tiles (NONE / LOW / MEDIUM / HIGH) with guardrail decisions
  - Agent Reasoning panel — per-agent bullet-point reasoning blocks
  - Event Feed — structured agent event cards with headline labels and crux metrics
  - Simulation engine — 48-step timeline with configurable speed, pause, and reset
- **Structured Prompt Engineering** — per-agent system prompts with incident context, strict JSON schema, 1-shot examples, and output repair logic.

---

## Models Used

| Model | Role |
|---|---|
| `qwen/qwen3-next-80b-a3b-instruct:free` | Primary — all 4 agent calls via OpenRouter |
| `openai/gpt-oss-20b:free` | Fallback — invoked automatically on rate-limit (429) retry failure |

---

## Datasets Used

All datasets are synthetic, purpose-built for the demo scenario:

| File | Description | Rows |
|---|---|---|
| `disk_metrics.json` | Time-series telemetry: disk %, write rate, event flags — 01:00 to 04:25 | 48 |
| `log_volume.json` | Per-service log lines/min across the incident window | 16 |
| `alerts.json` | 13 pre-authored alert events with trigger indices and pipeline flags | 13 |
| `deployment_events.json` | 4 deployment records including the root-cause deploy (v2.1.4) | 4 |
| `service_map.json` | Service topology, dependencies, criticality, and on-call roster | — |
| `agent_responses.json` | Pre-computed fallback responses for demo without API key | — |

---

## Token Usage (Per Full Pipeline Run)

| Component | Est. Input Tokens | Est. Output Tokens |
|---|---|---|
| Detect Agent | ~900 | ~480 |
| Correlate Agent | ~950 | ~480 |
| Diagnose Agent | ~1,000 | ~480 |
| Recommend Agent | ~1,100 | ~480 |
| **Total per run** | **~3,950** | **~1,920** |
| **Full demo (2 pipeline runs)** | **~7,900** | **~3,840** |
| **Grand total** | **~11,740 tokens** | |

> Token counts are estimates based on prompt construction. Actual usage varies by model tokeniser and response verbosity.

---

## End-to-End Latency

| Phase | Duration |
|---|---|
| Simulation to trigger (step 9 / 01:18) | ~4 seconds at Normal speed |
| Detect agent LLM call | 5 – 12 seconds |
| Correlate agent LLM call | 5 – 12 seconds |
| Diagnose agent LLM call | 5 – 12 seconds |
| Recommend agent LLM call | 6 – 15 seconds |
| **Total pipeline (detection to recommendation)** | **~25 – 55 seconds** |

> Latency is dominated by model inference time on the free-tier OpenRouter endpoint. Production deployment with a paid-tier model would reduce agent call latency to 2 – 5 seconds each (~10 – 20 seconds total).

---

## Expected Impact & Value

| Dimension | Current (Manual) | With ReflexOps |
|---|---|---|
| **Time to detect** | 15 – 30 min (alert threshold breach) | < 5 min (anomaly spike detected) |
| **Time to root cause** | 30 – 90 min (manual log and event correlation) | < 1 min (Correlate + Diagnose agents) |
| **Time to remediation plan** | Engineer-dependent | Immediate (Recommend agent with risk tiering) |
| **On-call cognitive load** | High — full context reconstruction each incident | Low — agent chain pre-populates context and ranks actions |
| **Knowledge democratisation** | Tribal — senior SRE required | Encoded — any on-call engineer can action the recommendation |
| **Scalability** | Linear with team headcount | Constant — agents scale to any number of concurrent incidents |

---

## Key Differentiators & Innovation

1. **Progressive context chaining** — unlike single-shot LLM queries, each agent in the chain receives structured output from the prior agent, enabling multi-step causal reasoning that mirrors how a senior SRE thinks through an incident.
2. **Guardrail-aware recommendations** — the Recommend agent does not just list options; it applies a risk policy (NONE/LOW → auto-execute, MEDIUM/HIGH → human approval) encoded directly in the prompt, producing actionable decisions rather than suggestions.
3. **Structured output repair** — a deterministic post-processing step (`_repair_json`) recovers from the most common LLM JSON malformation (keys leaking into arrays) without requiring a retry, reducing latency and API cost.
4. **Simulation-as-ground-truth** — the demo uses a fully authored 48-step incident timeline as ground truth, allowing the AI pipeline to be evaluated against a known scenario while still generating live, non-deterministic LLM responses.
5. **Risk action grid UI** — translates dense LLM reasoning into four colour-coded, decision-ready tiles that an on-call engineer can act on in seconds, bridging the gap between AI output and operational workflow.

---

## Demo Flow Overview

| Step | What Happens |
|---|---|
| 1. **Login** | Enter credentials (`admin / reflexops`) to access the dashboard |
| 2. **Press ▶ Start** | Simulation begins — disk metrics, log volume, and events advance in real time |
| 3. **01:14 — Deployment** | `order-service v2.1.4` deployed; deploy event appears on chart and event feed |
| 4. **01:18 — Anomaly trigger** | Write rate spikes 40×; simulation freezes; agent pipeline activates |
| 5. **🔍 Detect runs** | LLM confirms anomaly (146σ above baseline), projects 173 min to disk full, assigns CRITICAL |
| 6. **🔗 Correlate runs** | LLM links the deploy (4 min prior) and `log.level=DEBUG` config change as root cause fingerprint |
| 7. **🧠 Diagnose runs** | LLM scores hypotheses: H1 debug logging 89%, H2 connection leak 7%, H3 traffic spike 4% |
| 8. **💡 Recommend runs** | LLM produces 4 ranked actions with risk levels and guardrail decisions |
| 9. **Risk Action Grid appears** | NONE / LOW / MEDIUM / HIGH tiles displayed above findings for immediate decision-making |
| 10. **Panels persist** | Agent Reasoning, Event Feed, and Recommendations remain visible after pipeline completes |

---

*Built with LangChain · OpenRouter · Streamlit · Plotly*
