# ReflexOps

ReflexOps is an autonomous operations intelligence application designed for real-time incident detection, root cause analysis, and guided remediation. Built with Streamlit and powered by multi-agent LLM pipelines using LangChain, it monitors production telemetry streams, automatically flags anomalies, and generates prioritized, risk-evaluated remediation plans.


## TO RUN

Configure environment variables:
Create a .env file in the root directory:
OPENROUTER_API_KEY="your_openrouter_api_key_here"

Start the Streamlit dashboard:
streamlit run app.py

## Major Features

* **Multi-Agent Incident Response Pipeline**
  * **Detect Agent:** Evaluates raw telemetry against rolling baselines, quantifies deviation, estimates time to exhaustion, and assigns incident severity.
  * **Correlate Agent:** Analyzes temporal event windows to connect deployment events, configuration diffs, and service-level metrics into a causal fingerprint.
  * **Diagnose Agent:** Scores competing root cause hypotheses using Bayesian-style confidence rankings and eliminates alternative explanations.
  * **Recommend Agent:** Produces prioritized remediation actions mapped directly to operational runbooks.

* **Risk-Aware Guardrail Framework**
  * Evaluates remediation actions across risk categories (None, Low, Medium, High).
  * Automatically categorizes actions into autonomous execution or approval-required workflows based on service risk profiles.

* **Real-Time Telemetry Dashboard**
  * Interactive time-series visualisations tracking disk utilization, write throughput, and per-service log volume.
  * Live status overview indicating incident progression, active alert tiers, and dependent service health.

* **Interactive Simulation Engine**
  * Step-based playback simulation with configurable playback speeds, pause controls, and reset capabilities.
  * Deterministic test bed designed to model escalating failure scenarios such as debug logging exhaustion and cascading microservice outages.
