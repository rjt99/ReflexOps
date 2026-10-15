import streamlit as st
import json
import time
import os
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
from llm.pipeline import run_agent, get_api_key, get_model

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReflexOps",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Matte black-gray + Elastic / English Blue palette
COLORS = {
    "bg":           "#080B10",
    "surface":      "#0F1319",
    "surface2":     "#141920",
    "border":       "#1C2435",
    "border2":      "#253046",
    "blue":         "#1558A8",   # matte elastic / English blue
    "blue_mid":     "#1E6FC7",
    "blue_bright":  "#2D8EE8",
    "blue_deep":    "#0C2A52",
    "green":        "#10B981",
    "amber":        "#F59E0B",
    "red":          "#EF4444",
    "cyan":         "#06B6D4",
    "text":         "#DDE4EE",
    "text2":        "#8A96AA",
    "text3":        "#6B7A94",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header                     {{ visibility: hidden; }}
[data-testid="stSidebar"],
[data-testid="collapsedControl"]              {{ display: none !important; }}
[data-testid="stAppViewContainer"]            {{ background: {COLORS['bg']}; }}
[data-testid="stMain"] .block-container       {{ padding: 0.6rem 1.2rem 0.6rem 1.2rem !important;
                                                 max-width: 100% !important; }}

/* ── Login centering ── */
.login-wrap {{
    display: flex; align-items: center; justify-content: center;
    min-height: calc(100vh - 80px);
}}
.login-card {{
    padding: 10px 40px;
}}
.login-logo {{
    font-size: 2.1rem; font-weight: 800; letter-spacing: -1px;
    color: {COLORS['blue_bright']}; text-align: center; margin-bottom: 2px;
}}
.login-logo span {{ color: {COLORS['blue_bright']}; }}
.login-sub {{
    color: {COLORS['text3']}; font-size: 0.7rem; text-align: center;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 30px;
}}
.login-divider {{
    border: none; border-top: 1px solid {COLORS['border2']};
    margin: 0 0 24px 0;
}}

/* ── Top bar ── */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
    border-radius: 10px; padding: 9px 18px; margin-bottom: 10px;
}}
.topbar-logo {{
    font-size: 1.1rem; font-weight: 800; color: {COLORS['text']}; letter-spacing: -0.3px;
}}
.topbar-logo span {{ color: {COLORS['blue_bright']}; }}

/* ── Sim controls ── */
.simbar {{
    background: {COLORS['surface2']}; border: 1px solid {COLORS['border']};
    border-radius: 8px; padding: 7px 14px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 12px;
}}

/* ── Metric cards — top-accent style ── */
.mcard {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-top: 2px solid var(--mc, {COLORS['blue']});
    border-radius: 10px; padding: 13px 15px;
    background-image: linear-gradient(180deg, color-mix(in srgb, var(--mc, {COLORS['blue']}) 8%, transparent) 0%, transparent 60%);
}}
.mcard-label {{ color: {COLORS['text3']}; font-size: 0.66rem; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 5px; }}
.mcard-value {{ font-size: 1.65rem; font-weight: 700; color: var(--mc, {COLORS['blue_bright']});
                line-height: 1; }}
.mcard-sub   {{ color: {COLORS['text3']}; font-size: 0.67rem; margin-top: 5px; }}

/* ── Section label ── */
.slabel {{
    color: {COLORS['text3']}; font-size: 0.66rem; text-transform: uppercase;
    letter-spacing: 1px; padding-bottom: 5px; border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 8px;
}}

/* ── Alert items — header-strip style ── */
.aitem {{
    background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
    border-radius: 8px; margin-bottom: 6px; overflow: hidden; font-size: 0.75rem;
}}
.aitem-head {{
    padding: 4px 10px;
    display: flex; justify-content: space-between; align-items: center;
    background: color-mix(in srgb, var(--ac, {COLORS['blue']}) 15%, transparent);
}}
.aitem-sev  {{ color: var(--ac); font-weight: 700; font-size: 0.65rem;
               text-transform: uppercase; letter-spacing: 0.5px; }}
.aitem-meta {{ color: {COLORS['text3']}; font-size: 0.65rem; }}
.aitem-body {{ padding: 7px 10px; }}
.aitem-title {{ color: {COLORS['text']}; font-weight: 600; margin-bottom: 2px; }}
.aitem-msg   {{ color: {COLORS['text2']}; line-height: 1.4; }}

/* ── Event feed ── */
.eitem {{
    display: flex; gap: 9px; align-items: flex-start;
    padding: 7px 10px; background: {COLORS['surface']}; border-radius: 7px;
    border-left: none; border-top: 1px solid {COLORS['border']};
    margin-bottom: 4px; font-size: 0.74rem;
}}
.eitem-label {{ color: {COLORS['text']}; font-weight: 600; }}
.eitem-time  {{ color: {COLORS['text3']}; font-size: 0.67rem; }}

/* ── Right pane: agent panel ── */
.agent-panel {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border2']};
    border-top: 2px solid {COLORS['blue']};
    border-radius: 10px; overflow: hidden; height: 100%;
}}
.ap-header {{
    background: {COLORS['surface2']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 10px 14px;
    display: flex; justify-content: space-between; align-items: flex-start;
}}
.ap-title {{
    font-size: 0.7rem; font-weight: 700; color: {COLORS['text']};
    text-transform: uppercase; letter-spacing: 0.8px;
}}
.ap-trigger {{ color: {COLORS['text2']}; font-size: 0.72rem; margin-top: 2px; }}

/* ── Pipeline step chips ── */
.step-row {{
    display: flex; flex-wrap: wrap; gap: 5px;
    padding: 10px 14px; border-bottom: 1px solid {COLORS['border']};
    background: {COLORS['bg']};
}}
.step {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border-radius: 5px; font-size: 0.68rem; font-weight: 600;
}}
.s-idle    {{ background: {COLORS['surface2']}; color: {COLORS['text3']};
              border: 1px solid {COLORS['border']}; }}
.s-run     {{ background: {COLORS['blue_deep']}; color: #93C5FD;
              border: 1px solid {COLORS['blue']}; box-shadow: 0 0 8px rgba(21,88,168,0.4); }}
.s-done    {{ background: #064E3B; color: #6EE7B7;
              border: 1px solid #10B981; }}
.step-arrow {{ color: {COLORS['border2']}; font-size: 0.6rem; align-self: center; }}

/* ── Current-agent "what" highlight section ── */
.agent-what {{
    background: linear-gradient(135deg, {COLORS['blue_deep']} 0%, #0A1929 100%);
    border: 1px solid {COLORS['blue']}; border-left: 3px solid {COLORS['blue_bright']};
    border-radius: 8px; padding: 10px 14px; margin: 10px 14px 0 14px;
}}
.aw-label {{
    color: {COLORS['blue_bright']}; font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 6px; font-weight: 700;
}}
.aw-agent {{
    color: {COLORS['text']}; font-weight: 700; font-size: 0.78rem; margin-bottom: 3px;
}}
.aw-role {{
    color: {COLORS['text3']}; font-size: 0.68rem; font-weight: 400;
}}
.aw-what {{
    color: #93C5FD; font-size: 0.74rem; line-height: 1.55; margin-top: 5px;
}}

/* ── Reasoning terminal (dull / muted) ── */
.rterm {{
    background: #050709; padding: 12px 14px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 0.72rem; border-bottom: 1px solid {COLORS['border']};
    margin-top: 10px;
}}
.rterm-label {{ color: {COLORS['text3']}; font-size: 0.6rem; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 8px; }}
.rchunk {{ color: #7EB8D4; line-height: 1.75; display: block; }}
.rchunk::before {{ content: '  › '; color: #4A85A8; }}
.rcursor {{ color: {COLORS['blue']}; animation: blink 0.9s step-end infinite; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}

/* ── Final output — bright highlighted ── */
.output-final {{
    background: linear-gradient(135deg, #052E1E 0%, #041F15 100%);
    border: 1px solid {COLORS['green']}44; border-left: 3px solid {COLORS['green']};
    border-radius: 8px; padding: 10px 14px; margin: 10px 14px;
}}
.of-label {{
    color: {COLORS['green']}; font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 5px; font-weight: 700;
}}
.of-agent {{
    color: #6EE7B7; font-size: 0.68rem; font-weight: 700;
    margin-bottom: 4px; opacity: 0.8;
}}
.of-text {{
    color: #ECFDF5; font-weight: 600; font-size: 0.76rem; line-height: 1.5;
}}

/* ── Completed outputs section (history) ── */
.out-section {{ padding: 10px 14px; }}
.out-label   {{ color: {COLORS['text3']}; font-size: 0.62rem; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 7px; border-bottom: 1px solid {COLORS['border']};
                padding-bottom: 5px; }}
.orow {{
    display: flex; gap: 8px; padding: 6px 8px; border-radius: 6px;
    margin-bottom: 4px; background: {COLORS['bg']};
    border: 1px solid {COLORS['border']}; font-size: 0.72rem;
    border-left: 2px solid {COLORS['green']};
}}
.orow-agent {{ color: #6EE7B7; font-weight: 700; white-space: nowrap; min-width: 74px;
               font-size: 0.7rem; }}
.orow-text  {{ color: {COLORS['text2']}; line-height: 1.4; font-size: 0.7rem; }}

/* ── Collapsible reasoning trace ── */
details.rdetails summary {{
    color: {COLORS['blue_bright']}; font-size: 0.62rem; cursor: pointer;
    user-select: none; padding: 4px 0; letter-spacing: 0.5px;
    text-transform: uppercase; list-style: none; font-weight: 600;
}}
details.rdetails summary::-webkit-details-marker {{ display: none; }}
details.rdetails summary::before {{ content: '▸  '; color: {COLORS['blue']}; }}
details.rdetails[open] summary::before {{ content: '▾  '; color: {COLORS['blue_bright']}; }}
details.rdetails[open] summary {{ color: {COLORS['text2']}; }}

/* ── Pipeline idle placeholder ── */
.ap-idle {{
    padding: 32px 20px; text-align: center;
    color: {COLORS['text3']}; font-size: 0.76rem;
}}
.ap-idle-icon {{ font-size: 2rem; margin-bottom: 10px; opacity: 0.3; }}

/* ── Pulse dot ── */
.pdot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
    animation: pdot 1.2s ease-in-out infinite;
}}
@keyframes pdot {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.3;transform:scale(0.7)}} }}

/* ── Phase badge ── */
.pbadge {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.8px;
    border: 1px solid;
}}
</style>
""", unsafe_allow_html=True)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_disk_metrics():
    return json.loads(Path("datasource/disk_metrics.json").read_text(encoding="utf-8"))

@st.cache_data
def load_log_volume():
    return json.loads(Path("datasource/log_volume.json").read_text(encoding="utf-8"))

@st.cache_data
def load_alerts():
    return json.loads(Path("datasource/alerts.json").read_text(encoding="utf-8"))

@st.cache_data
def load_service_map():
    return json.loads(Path("datasource/service_map.json").read_text(encoding="utf-8"))

@st.cache_data
def load_agent_responses():
    return json.loads(Path("datasource/agent_responses.json").read_text(encoding="utf-8"))

# ── Session state ──────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "logged_in": False, "username": "",
        "sim_running": False, "sim_index": 0, "sim_speed": 0.4,
        "active_alerts": [], "seen_alert_ids": set(),
        "pipeline_active": False, "pipeline_alert_id": None,
        "pipeline_agent_idx": 0, "pipeline_chunk_idx": 0,
        "pipeline_revealed": [],
        "pipeline_queue": [], "triggered_pipeline_ids": set(),
        "pipeline_history": [],
        # LLM state
        "llm_model": get_model(),
        "llm_fetching": False,
        "llm_prefetched": {},   # {agent_idx: {chunks, output}} pre-fetched by LLM
        "pipeline_context": {}, # accumulates each agent's output for next agent's context
        "pipeline_failed": False,
        "pipeline_error_msg": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
USERS = {"admin": "reflexops", "ops": "ops123"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def sev_color(sev):
    return {"info": COLORS["cyan"], "warning": COLORS["amber"],
            "critical": COLORS["red"], "resolved": COLORS["green"]}.get(sev, COLORS["blue"])

def disk_color(pct):
    if pct >= 90: return COLORS["red"]
    if pct >= 70: return COLORS["amber"]
    return COLORS["green"]

def get_phase(idx, total):
    p = idx / max(total - 1, 1)
    if p < 0.17: return "NORMAL",    COLORS["green"], "All systems nominal"
    if p < 0.22: return "DEPLOY",    COLORS["cyan"],  "Deployment detected — watching for side effects"
    if p < 0.30: return "ANOMALY",   COLORS["amber"], "Anomaly detected — agent pipeline activated"
    if p < 0.58: return "FILLING",   COLORS["amber"], "Disk filling at 40x normal rate — debug log active"
    if p < 0.60: return "ROTATION",  COLORS["cyan"],  "Log rotation executed — root cause still running"
    if p < 0.88: return "REFILLING", COLORS["red"],   "Rotation ineffective — escalating — no on-call ACK"
    if p < 0.95: return "INCIDENT",  COLORS["red"],   "CRITICAL: disk full — service crashed — cascade active"
    return                "RECOVERING",COLORS["green"],"On-call responding — rollback in progress"

# ── Pipeline state machine ─────────────────────────────────────────────────────
def start_pipeline(alert_id):
    st.session_state.sim_running        = False   # freeze timeline at detect trigger
    st.session_state.pipeline_active    = True
    st.session_state.pipeline_alert_id  = alert_id
    st.session_state.pipeline_agent_idx = 0
    st.session_state.pipeline_chunk_idx = 0
    st.session_state.pipeline_revealed  = []
    st.session_state.llm_prefetched     = {}
    st.session_state.pipeline_context   = {}
    st.session_state.llm_fetching       = False
    st.session_state.pipeline_failed    = False
    st.session_state.pipeline_error_msg = None

def end_pipeline():
    if st.session_state.pipeline_revealed:
        st.session_state.pipeline_history.append({
            "alert_id": st.session_state.pipeline_alert_id,
            "agents":   list(st.session_state.pipeline_revealed),
        })
    st.session_state.pipeline_active    = False
    st.session_state.pipeline_alert_id  = None
    st.session_state.pipeline_agent_idx = 0
    st.session_state.pipeline_chunk_idx = 0
    st.session_state.pipeline_revealed  = []
    st.session_state.llm_prefetched     = {}
    st.session_state.pipeline_context   = {}
    st.session_state.llm_fetching       = False
    if st.session_state.pipeline_queue:
        start_pipeline(st.session_state.pipeline_queue.pop(0))

def _agent_def_list():
    """Return the agent definition list for the active pipeline (ALT-002 only for now)."""
    return load_agent_responses()["pipelines"].get(
        st.session_state.pipeline_alert_id, {}
    ).get("agents", [])

def advance_pipeline():
    """
    Advance the pipeline by one step.
    - If the current agent has no pre-fetched data yet: call LLM (or fallback), store result.
    - If pre-fetched data exists: reveal one chunk at a time.
    Returns True if more work remains.
    """
    agents = _agent_def_list()
    ai     = st.session_state.pipeline_agent_idx
    ci     = st.session_state.pipeline_chunk_idx

    if ai >= len(agents):
        end_pipeline(); return False

    agent_def  = agents[ai]
    agent_name = agent_def["name"]

    # ── Ensure the revealed slot exists ──────────────────────────────────
    if len(st.session_state.pipeline_revealed) <= ai:
        st.session_state.pipeline_revealed.append({
            "name":         agent_name,
            "icon":         agent_def["icon"],
            "role":         agent_def["role"],
            "chunks_shown": [],
            "complete":     False,
            "output":       None,
            "event_label":  None,
            "crux":         None,
            "risk_actions": [],
            "from_llm":     False,
            "error":        None,
        })

    # ── Call LLM for this agent if not yet done ───────────────────────────
    if ai not in st.session_state.llm_prefetched:
        api_key = get_api_key()
        if not api_key:
            st.session_state.pipeline_failed    = True
            st.session_state.pipeline_error_msg = "No OPENROUTER_API_KEY found in .env"
            st.session_state.pipeline_active    = False
            return False
        st.session_state.llm_fetching = True
        try:
            result = run_agent(
                name    = agent_name,
                api_key = api_key,
                model   = st.session_state.llm_model,
                context = st.session_state.pipeline_context,
            )
        except Exception as e:
            st.session_state.llm_fetching       = False
            st.session_state.pipeline_failed    = True
            st.session_state.pipeline_error_msg = str(e)
            st.session_state.pipeline_active    = False
            return False
        st.session_state.llm_fetching = False
        st.session_state.llm_prefetched[ai] = result
        st.session_state.pipeline_revealed[ai]["from_llm"] = True

    # ── Reveal chunks one at a time ───────────────────────────────────────
    prefetched = st.session_state.llm_prefetched[ai]
    chunks     = prefetched.get("chunks", [])

    if ci < len(chunks):
        st.session_state.pipeline_revealed[ai]["chunks_shown"].append(chunks[ci])
        st.session_state.pipeline_chunk_idx += 1
        return True

    # ── Agent complete — seal and advance ─────────────────────────────────
    output       = prefetched.get("output", "")
    event_label  = prefetched.get("event_label", "")
    crux         = prefetched.get("crux", "")
    risk_actions = prefetched.get("risk_actions", [])
    st.session_state.pipeline_revealed[ai]["complete"]     = True
    st.session_state.pipeline_revealed[ai]["output"]       = output
    st.session_state.pipeline_revealed[ai]["event_label"]  = event_label
    st.session_state.pipeline_revealed[ai]["crux"]         = crux
    st.session_state.pipeline_revealed[ai]["risk_actions"] = risk_actions

    # Accumulate context for the next agent
    st.session_state.pipeline_context[agent_name] = {"output": output}

    st.session_state.pipeline_agent_idx += 1
    st.session_state.pipeline_chunk_idx  = 0

    if st.session_state.pipeline_agent_idx >= len(agents):
        end_pipeline(); return False
    return True

def check_triggers(idx, alerts_data):
    # Only fire one pipeline per session — ignore any triggers after the first
    if st.session_state.triggered_pipeline_ids:
        return
    for a in alerts_data:
        if a.get("pipeline_trigger") and a["trigger_index"] == idx:
            st.session_state.triggered_pipeline_ids.add(a["id"])
            start_pipeline(a["id"])
            break

# ── Login page ─────────────────────────────────────────────────────────────────
def login_page():
    _, col, _ = st.columns([1, 1.05, 1])
    with col:
        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="login-card">'
            f'<div class="login-logo">⚡ Reflex<span>Ops</span></div>'
            f'<div class="login-sub">Autonomous Operations Intelligence</div>'
            f'<hr class="login-divider">',
            unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="reflexops")
            submitted = st.form_submit_button("Sign In  →", use_container_width=True, type="primary")
        if submitted:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown('</div>', unsafe_allow_html=True)

# ── Top bar ────────────────────────────────────────────────────────────────────
def render_topbar(idx, total, disk_metrics):
    phase, pcolor, pdesc = get_phase(idx, total)
    api_key = get_api_key()
    llm_active = bool(api_key)

    c1, c2, c3 = st.columns([3, 3, 3])
    with c1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:6px 0">'
            f'<span style="font-size:1.1rem;font-weight:800;color:{COLORS["text"]}">'
            f'⚡ Reflex<span style="color:{COLORS["blue_bright"]}">Ops</span></span>'
            f'<span style="color:{COLORS["text3"]};font-size:0.7rem">'
            f'Autonomous Operations Intelligence'
            f'</span></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:4px 0">'
            f'<span class="pbadge" style="color:{pcolor};border-color:{pcolor}44;background:{pcolor}11">'
            f'{phase}</span>'
            f'<span style="color:{COLORS["text3"]};font-size:0.68rem">{pdesc}</span></div>',
            unsafe_allow_html=True)
    with c3:
        status_col, btn_col = st.columns([3, 1])
        with status_col:
            if not llm_active:
                st.markdown(
                    f'<div style="padding:6px 0;font-size:0.72rem">'
                    f'<span class="pdot" style="background:{COLORS["red"]}"></span>'
                    f'<span style="color:{COLORS["red"]};font-weight:600">No API Key</span>'
                    f'<span style="color:{COLORS["text3"]};margin-left:8px">add OPENROUTER_API_KEY to .env</span>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                if st.session_state.llm_fetching:
                    mode_color, mode_label = COLORS["amber"], "⟳  Calling LLM..."
                elif st.session_state.pipeline_failed:
                    mode_color, mode_label = COLORS["red"], "⚠  LLM Error"
                else:
                    mode_color, mode_label = COLORS["green"], "LLM Ready"
                st.markdown(
                    f'<div style="padding:6px 0;font-size:0.72rem">'
                    f'<span class="pdot" style="background:{mode_color}"></span>'
                    f'<span style="color:{mode_color};font-weight:600">{mode_label}</span>'
                    f'<span style="color:{COLORS["text3"]};margin-left:8px">'
                    f'{st.session_state.llm_model.split("/")[-1]}</span>'
                    f'</div>',
                    unsafe_allow_html=True)
        with btn_col:
            if st.button("Sign Out", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

# ── Sim controls ───────────────────────────────────────────────────────────────
def render_sim_controls(idx, total):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 5])
    with c1:
        if not st.session_state.sim_running:
            label = "▶ Resume" if idx > 0 else "▶ Start"
            if st.button(label, use_container_width=True, type="primary"):
                st.session_state.sim_running = True
                st.rerun()
        else:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.sim_running = False
                st.rerun()
    with c2:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.sim_running            = False
            st.session_state.sim_index              = 0
            st.session_state.active_alerts          = []
            st.session_state.seen_alert_ids         = set()
            st.session_state.pipeline_active        = False
            st.session_state.pipeline_alert_id      = None
            st.session_state.pipeline_agent_idx     = 0
            st.session_state.pipeline_chunk_idx     = 0
            st.session_state.pipeline_revealed      = []
            st.session_state.pipeline_queue         = []
            st.session_state.triggered_pipeline_ids = set()
            st.session_state.pipeline_history       = []
            st.session_state.llm_prefetched         = {}
            st.session_state.pipeline_context       = {}
            st.session_state.llm_fetching           = False
            st.session_state.pipeline_failed        = False
            st.session_state.pipeline_error_msg     = None
            st.rerun()
    with c3:
        speed_map = {"Slow": 1.0, "Normal": 0.4, "Fast": 0.15}
        chosen = st.selectbox("spd", list(speed_map.keys()), index=1, label_visibility="collapsed")
        st.session_state.sim_speed = speed_map[chosen]
    with c4:
        pct  = idx / max(total - 1, 1)
        _, pcolor, _ = get_phase(idx, total)
        pipe_q = len(st.session_state.pipeline_queue)
        queue_note = f' · {pipe_q} queued' if pipe_q else ''
        st.markdown(
            f'<div style="display:flex;flex-direction:column;gap:3px">'
            f'<div style="display:flex;align-items:center;gap:10px;height:28px">'
            f'<div style="flex:1;background:{COLORS["surface2"]};border-radius:3px;height:4px">'
            f'<div style="background:{pcolor};width:{pct*100:.1f}%;height:4px;border-radius:3px"></div></div>'
            f'<span style="color:{COLORS["text3"]};font-size:0.7rem;white-space:nowrap">'
            f'Step {idx+1}/{total}{queue_note}</span></div>'
            f'<div style="color:{COLORS["text3"]};font-size:0.6rem;letter-spacing:0.4px;opacity:0.7;padding-left:2px">'
            f'Scenario · Undetected debug-logging misconfiguration escalates to disk exhaustion and cascading service failure</div>'
            f'</div>',
            unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
def render_metrics(current, active_alerts):
    pct   = current["used_pct"]
    rate  = current["write_rate_mbps"]
    dc    = disk_color(pct)
    rc    = COLORS["red"] if rate > 20 else (COLORS["amber"] if rate > 5 else COLORS["green"])
    crit  = sum(1 for a in active_alerts if a["severity"] == "critical")
    warn  = sum(1 for a in active_alerts if a["severity"] == "warning")
    ac    = COLORS["red"] if crit else (COLORS["amber"] if warn else COLORS["green"])
    crash = 2 if pct >= 100 else 0
    sc    = COLORS["red"] if crash else (COLORS["amber"] if pct > 70 else COLORS["green"])

    for col, label, val, sub, color in zip(
        st.columns(4),
        ["Disk Used", "Write Rate", "Active Alerts", "Services Down"],
        [f"{pct:.1f}%", f"{rate:.1f} mbps", f"{crit}C / {warn}W" if active_alerts else "0",
         str(crash)],
        [f"{current['used_gb']} GB / {current['total_gb']} GB", "baseline: 2.1 mbps",
         f"{len(active_alerts)} total events", "cascade in progress" if crash else "all healthy"],
        [dc, rc, ac, sc],
    ):
        with col:
            st.markdown(
                f'<div class="mcard" style="--mc:{color}">'
                f'<div class="mcard-label">{label}</div>'
                f'<div class="mcard-value">{val}</div>'
                f'<div class="mcard-sub">{sub}</div></div>',
                unsafe_allow_html=True)

# ── Disk chart ─────────────────────────────────────────────────────────────────
def render_disk_chart(disk_metrics, idx):
    visible = disk_metrics[:idx + 1]
    df = pd.DataFrame(visible)
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=70,  fillcolor="#10B981", opacity=0.03, line_width=0)
    fig.add_hrect(y0=70, y1=85,  fillcolor="#F59E0B", opacity=0.04, line_width=0)
    fig.add_hrect(y0=85, y1=100, fillcolor="#EF4444", opacity=0.06, line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color="#F59E0B", line_width=1, opacity=0.4,
                  annotation_text="70%", annotation_font_color="#F59E0B", annotation_font_size=10)
    fig.add_hline(y=85, line_dash="dot", line_color="#EF4444", line_width=1, opacity=0.4,
                  annotation_text="85%", annotation_font_color="#EF4444", annotation_font_size=10)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["used_pct"], name="Disk %",
        line=dict(color=COLORS["blue_bright"], width=2),
        fill="tozeroy", fillcolor=f'rgba(45,142,232,0.07)', mode="lines",
        hovertemplate="<b>%{x}</b><br>Disk: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["write_rate_mbps"], name="Write mbps",
        line=dict(color=COLORS["cyan"], width=1.5, dash="dot"), mode="lines", yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Write: %{y:.1f} mbps<extra></extra>"))
    ec_map = {"deploy": COLORS["cyan"], "anomaly": COLORS["amber"], "alert": COLORS["amber"],
              "action": COLORS["green"], "incident": COLORS["red"], "resolved": COLORS["green"]}
    for row in visible:
        if row["event"]:
            c = ec_map.get(row["event"].split(":")[0], COLORS["text3"])
            fig.add_vline(x=row["timestamp"], line_dash="dash", line_color=c, line_width=1, opacity=0.5)
    fig.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text2"], size=10), margin=dict(l=0, r=0, t=4, b=0), height=200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor=COLORS["border"], tickfont=dict(size=9),
                   color=COLORS["text3"], zeroline=False),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor=COLORS["border"],
                   tickfont=dict(size=9), color=COLORS["text3"], zeroline=False),
        yaxis2=dict(overlaying="y", side="right", range=[0, 55], showgrid=False,
                    tickfont=dict(size=9), color=COLORS["cyan"]),
        hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Log volume chart ───────────────────────────────────────────────────────────
def render_log_chart(log_data, disk_idx):
    disk_metrics = load_disk_metrics()
    cur_ts  = disk_metrics[min(disk_idx, len(disk_metrics) - 1)]["timestamp"]
    visible = [e for e in log_data if e["timestamp"] <= cur_ts] or [log_data[0]]
    latest  = visible[-1]
    apps    = ["order-service", "payment-service", "auth-service", "checkout-service"]
    values  = [latest[a]["lines_per_min"] for a in apps]
    colors  = [COLORS["red"]   if latest[a].get("log_level") in ("CRASHED", "ERROR")
               else COLORS["amber"] if latest[a].get("log_level") == "DEBUG"
               else COLORS["blue_bright"] for a in apps]
    short_apps = ["order", "payment", "auth", "checkout"]
    fig = go.Figure(go.Bar(
        x=short_apps, y=values, marker_color=colors,
        text=[f"{v:,}" for v in values], textposition="outside",
        textfont=dict(size=9, color=COLORS["text2"]),
        hovertemplate="<b>%{x}</b><br>%{y:,} lines/min<extra></extra>"))
    fig.add_hline(y=1200, line_dash="dot", line_color=COLORS["text3"], line_width=1, opacity=0.5,
                  annotation_text="baseline", annotation_font_color=COLORS["text3"], annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text2"], size=10), margin=dict(l=0, r=0, t=4, b=0), height=200,
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], tickfont=dict(size=9),
                   color=COLORS["text3"], range=[0, max(max(values) * 1.25, 2000)]),
        xaxis=dict(tickfont=dict(size=9), color=COLORS["text3"]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Shared agent palette (used by both alerts and events) ──────────────────────
_AGENT_COLOR = {
    "Detect":    "#F59E0B",   # amber
    "Correlate": "#22D3EE",   # cyan
    "Diagnose":  "#60A5FA",   # blue
    "Recommend": "#34D399",   # green
}
_AGENT_ICON = {"Detect": "🔍", "Correlate": "🔗", "Diagnose": "🧠", "Recommend": "💡"}

# ── Alerts ─────────────────────────────────────────────────────────────────────
def render_alerts(active_alerts):
    """Reasoning-points panel — one colour-coded block per completed agent."""
    alert_id = st.session_state.pipeline_alert_id
    revealed = st.session_state.pipeline_revealed
    history  = st.session_state.pipeline_history

    # Fall back to last completed run if pipeline has ended
    if not alert_id:
        if not history:
            st.markdown(
                f'<div style="color:{COLORS["text3"]};font-size:0.75rem;'
                f'text-align:center;padding:16px">Waiting for anomaly detection…</div>',
                unsafe_allow_html=True)
            return
        alert_id = history[-1]["alert_id"]

    trigger_ts = next((a["timestamp"] for a in active_alerts if a["id"] == alert_id), "")
    status     = "RUNNING" if st.session_state.pipeline_active else "COMPLETE"

    # ── Pipeline header strip ────────────────────────────────────────────────
    dot_color = COLORS["blue_bright"] if st.session_state.pipeline_active else COLORS["green"]
    st.markdown(
        f'<div style="background:{COLORS["surface2"]};border:1px solid {COLORS["border"]};'
        f'border-left:3px solid {dot_color};border-radius:7px;'
        f'padding:7px 12px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">'
        f'<span style="color:{COLORS["text"]};font-size:0.72rem;font-weight:700">🤖 Agent Pipeline</span>'
        f'<span style="color:{dot_color};font-size:0.62rem;font-weight:700;letter-spacing:0.8px">'
        f'{status} · {alert_id} · {trigger_ts}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # ── One block per completed agent — reasoning as bullet points ───────────
    completed = (
        [r for r in revealed  if r.get("complete") and r.get("chunks_shown")] if revealed else
        [r for r in history[-1]["agents"] if r.get("complete") and r.get("chunks_shown")] if history else []
    )

    if not completed:
        st.markdown(
            f'<div style="color:{COLORS["text3"]};font-size:0.72rem;padding:8px 4px">'
            f'<span style="animation:blink 1s step-end infinite">▌</span> Agents initialising…</div>',
            unsafe_allow_html=True)
        return

    for ag in completed:
        color  = _AGENT_COLOR.get(ag["name"], COLORS["text2"])
        icon   = _AGENT_ICON.get(ag["name"], "🤖")
        chunks = ag.get("chunks_shown", [])
        points = "".join(
            f'<div style="display:flex;gap:6px;margin-bottom:4px">'
            f'<span style="color:{color};margin-top:1px;flex-shrink:0">›</span>'
            f'<span style="color:{COLORS["text2"]};font-size:0.7rem;line-height:1.5">{c}</span>'
            f'</div>'
            for c in chunks
        )
        st.markdown(
            f'<div style="border-left:3px solid {color};border-radius:0 6px 6px 0;'
            f'background:{COLORS["surface"]};padding:8px 12px;margin-bottom:7px">'
            f'<div style="color:{color};font-size:0.68rem;font-weight:700;'
            f'margin-bottom:6px;letter-spacing:0.4px">{icon} {ag["name"]} · {ag.get("role","")}</div>'
            f'{points}'
            f'</div>',
            unsafe_allow_html=True)

# ── Events ─────────────────────────────────────────────────────────────────────
def render_events(disk_metrics, idx):
    """Agent pipeline event feed — identical card format for every entry."""
    alert_id = st.session_state.pipeline_alert_id
    revealed = st.session_state.pipeline_revealed
    history  = st.session_state.pipeline_history

    # Fall back to last completed run if pipeline has ended
    if not alert_id:
        if not history:
            st.markdown(
                f'<div style="color:{COLORS["text3"]};font-size:0.75rem;'
                f'text-align:center;padding:16px">Waiting for pipeline trigger…</div>',
                unsafe_allow_html=True)
            return
        alert_id = history[-1]["alert_id"]

    trigger_ts = next(
        (a["timestamp"] for a in load_alerts() if a["id"] == alert_id), ""
    )

    # Pipeline-start entry (same card structure as agent entries)
    entries = [{
        "icon":      "🤖",
        "name":      "Pipeline",
        "color":     COLORS["blue_bright"],
        "label":     f"Pipeline activated · {alert_id}",
        "crux":      "Detect → Correlate → Diagnose → Recommend",
        "timestamp": trigger_ts,
    }]

    completed = (
        [r for r in revealed  if r.get("complete") and r.get("output")] if revealed else
        [r for r in history[-1]["agents"] if r.get("complete") and r.get("output")] if history else []
    )
    for ag in completed:
        label = ag.get("event_label") or ""
        if not label:
            label = (ag["output"][:55] + "…") if len(ag["output"]) > 55 else ag["output"]
        entries.append({
            "icon":      _AGENT_ICON.get(ag["name"], "🤖"),
            "name":      ag["name"],
            "color":     _AGENT_COLOR.get(ag["name"], COLORS["text2"]),
            "label":     label,
            "crux":      ag.get("crux") or "",
            "timestamp": trigger_ts,
        })

    # Newest (last completed) at top
    for row in reversed(entries):
        crux_html = (
            f'<div style="color:{COLORS["text3"]};font-size:0.62rem;'
            f'margin-top:2px;letter-spacing:0.2px">{row["crux"]}</div>'
            if row["crux"] else ""
        )
        st.markdown(
            f'<div style="display:flex;gap:9px;align-items:flex-start;'
            f'padding:7px 10px;background:{COLORS["surface"]};border-radius:7px;'
            f'border-left:3px solid {row["color"]};margin-bottom:5px">'
            f'<span style="font-size:0.85rem;margin-top:1px">{row["icon"]}</span>'
            f'<div style="flex:1">'
            f'<div style="color:{row["color"]};font-size:0.66rem;font-weight:700;'
            f'letter-spacing:0.5px;margin-bottom:2px">{row["name"].upper()}</div>'
            f'<div style="color:{COLORS["text"]};font-size:0.71rem;line-height:1.4">{row["label"]}</div>'
            f'{crux_html}'
            f'<div style="color:{COLORS["text3"]};font-size:0.6rem;margin-top:3px">{row["timestamp"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True)

# ── Agent pipeline panel ───────────────────────────────────────────────────────
def render_agent_panel():
    responses = load_agent_responses()
    active    = st.session_state.pipeline_active
    revealed  = st.session_state.pipeline_revealed
    history   = st.session_state.pipeline_history
    alert_id  = st.session_state.pipeline_alert_id
    a_idx     = st.session_state.pipeline_agent_idx

    # Determine which pipeline definition to show
    if active:
        pdef       = responses["pipelines"].get(alert_id, {})
        all_agents = pdef.get("agents", [])
        tname      = pdef.get("trigger_name", alert_id or "")
        status_dot = f'<span class="pdot" style="background:{COLORS["blue_bright"]}"></span>'
        status_txt = f'<span style="color:{COLORS["blue_bright"]};font-size:0.7rem;font-weight:700">PROCESSING</span>'
        aid_txt    = f'<span style="color:{COLORS["text3"]};font-size:0.68rem;margin-left:6px">{alert_id}</span>'
    elif history:
        last       = history[-1]
        pdef       = responses["pipelines"].get(last["alert_id"], {})
        all_agents = pdef.get("agents", [])
        tname      = pdef.get("trigger_name", last["alert_id"])
        status_dot = f'<span class="pdot" style="background:{COLORS["green"]}"></span>'
        status_txt = f'<span style="color:{COLORS["green"]};font-size:0.7rem;font-weight:700">COMPLETE</span>'
        aid_txt    = f'<span style="color:{COLORS["text3"]};font-size:0.68rem;margin-left:6px">{last["alert_id"]}</span>'
    else:
        all_agents = []
        tname      = ""
        status_dot = f'<span class="pdot" style="background:{COLORS["text3"]};animation:none"></span>'
        status_txt = f'<span style="color:{COLORS["text3"]};font-size:0.7rem">WAITING</span>'
        aid_txt    = ""

    # ── Panel header ──
    st.markdown(
        f'<div class="agent-panel">'
        f'<div class="ap-header">'
        f'<div><div class="ap-title">🤖 Agent Pipeline</div>'
        f'<div class="ap-trigger">{tname}</div></div>'
        f'<div>{status_dot}{status_txt}{aid_txt}</div>'
        f'</div>',
        unsafe_allow_html=True)

    # ── LLM error banner ──
    if st.session_state.pipeline_failed and st.session_state.pipeline_error_msg:
        st.markdown(
            f'<div style="background:#1A0808;border:1px solid {COLORS["red"]}66;'
            f'border-left:3px solid {COLORS["red"]};border-radius:8px;'
            f'padding:12px 16px;margin:12px 14px">'
            f'<div style="color:{COLORS["red"]};font-size:0.6rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px">LLM Error</div>'
            f'<div style="color:#FCA5A5;font-size:0.74rem;font-family:monospace;line-height:1.5;word-break:break-all">'
            f'{st.session_state.pipeline_error_msg}'
            f'</div></div>',
            unsafe_allow_html=True)

    # ── Step chips ──
    if all_agents:
        done_names = set()
        if history and not active:
            done_names = {ag["name"] for ag in history[-1]["agents"] if ag.get("complete")}
        chips = ""
        for i, ag in enumerate(all_agents):
            if active:
                if i < a_idx:   cls = "s-done";  sfx = " ✓"
                elif i == a_idx: cls = "s-run";   sfx = " ⟳"
                else:            cls = "s-idle";  sfx = ""
            else:
                cls = "s-done" if ag["name"] in done_names else "s-idle"
                sfx = " ✓" if ag["name"] in done_names else ""
            chips += f'<span class="step {cls}">{ag["icon"]} {ag["name"]}{sfx}</span>'
            if i < len(all_agents) - 1:
                chips += '<span class="step-arrow">▶</span>'
        st.markdown(f'<div class="step-row">{chips}</div>', unsafe_allow_html=True)
    else:
        # Idle state — show only the 4 active agents
        idle_chips = "".join(
            f'<span class="step s-idle">{icon} {name}</span>'
            + ('<span class="step-arrow">▶</span>' if i < 3 else "")
            for i, (icon, name) in enumerate([
                ("🔍","Detect"),("🔗","Correlate"),("🧠","Diagnose"),("💡","Recommend")
            ])
        )
        st.markdown(f'<div class="step-row">{idle_chips}</div>', unsafe_allow_html=True)

    # ── Idle placeholder ──
    if not active and not history:
        st.markdown(
            f'<div class="ap-idle">'
            f'<div class="ap-idle-icon">🤖</div>'
            f'Start the simulation — agents activate automatically<br>'
            f'when anomaly triggers are detected in the data stream.'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Active: "what this agent is doing" section + dull reasoning ──
    if active and a_idx < len(all_agents):
        cur_def = all_agents[a_idx]
        llm_label = "LLM" if get_api_key() else "Demo"
        source_badge = (
            f'<span style="background:{COLORS["blue_deep"]};color:{COLORS["blue_bright"]};'
            f'font-size:0.58rem;padding:1px 7px;border-radius:4px;margin-left:8px;vertical-align:middle">LLM</span>'
            if get_api_key() else
            f'<span style="background:{COLORS["surface2"]};color:{COLORS["text3"]};'
            f'font-size:0.58rem;padding:1px 7px;border-radius:4px;margin-left:8px;vertical-align:middle">DEMO</span>'
        )
        st.markdown(
            f'<div class="agent-what">'
            f'<div class="aw-label">Current Task</div>'
            f'<div class="aw-agent">{cur_def["icon"]} {cur_def["name"]}{source_badge}'
            f'<span class="aw-role"> — {cur_def["role"]}</span></div>'
            f'<div class="aw-what">{cur_def["what"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

        # ── Dull reasoning stream ──
        if st.session_state.llm_fetching:
            st.markdown(
                f'<div class="rterm">'
                f'<div class="rterm-label">Calling {llm_label} · {st.session_state.llm_model.split("/")[-1]}</div>'
                f'<span style="color:{COLORS["text3"]};font-size:0.72rem">'
                f'Waiting for response<span class="rcursor">▌</span></span>'
                f'</div>',
                unsafe_allow_html=True)
        elif revealed:
            cur    = revealed[-1]
            if not cur.get("complete"):
                chunks = "".join(f'<span class="rchunk">{c}</span>' for c in cur["chunks_shown"])
                cursor = '<span class="rcursor">▌</span>'
                err_note = ""
                if cur.get("error"):
                    err_note = (f'<div style="color:{COLORS["amber"]};font-size:0.67rem;margin-top:6px">'
                                f'⚠ LLM error — showing fallback</div>')
                st.markdown(
                    f'<div class="rterm">'
                    f'<div class="rterm-label">Reasoning trace</div>'
                    f'{chunks}{cursor}{err_note}'
                    f'</div>',
                    unsafe_allow_html=True)

    # ── Completed agents — each shown as a bright output strip ──
    completed = []
    if active:
        completed = [r for r in revealed if r.get("complete") and r.get("output")]
    elif history:
        completed = [r for r in history[-1]["agents"] if r.get("complete") and r.get("output")]

    # ── Risk action grid (Recommend agent output, shown when pipeline is done) ──
    _RISK_COLOR  = {"NONE": "#34D399", "LOW": "#22D3EE", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}
    _RISK_BG     = {"NONE": "#052E1E", "LOW": "#042838", "MEDIUM": "#2D1B00", "HIGH": "#2D0707"}
    _RISK_LABEL  = {"NONE": "✓ Auto-Execute", "LOW": "✓ Auto-Execute",
                    "MEDIUM": "⚠ Needs Approval", "HIGH": "⛔ Approval + 2nd Review"}

    recommend_slot = next(
        (r for r in (revealed or (history[-1]["agents"] if history else []))
         if r.get("name") == "Recommend" and r.get("complete")),
        None,
    )
    risk_actions = recommend_slot.get("risk_actions", []) if recommend_slot else []

    if risk_actions:
        st.markdown(
            f'<div style="border-top:1px solid {COLORS["border"]};margin-top:10px;padding:10px 14px 4px">'
            f'<div style="color:{COLORS["text3"]};font-size:0.6rem;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:700;margin-bottom:8px">Recommended Actions</div>',
            unsafe_allow_html=True)

        cols = st.columns(2)
        for i, ra in enumerate(risk_actions[:4]):
            lvl    = ra.get("level", "LOW").upper()
            color  = _RISK_COLOR.get(lvl, COLORS["text2"])
            bg     = _RISK_BG.get(lvl, COLORS["surface"])
            badge  = _RISK_LABEL.get(lvl, "")
            action = ra.get("action", "")
            detail = ra.get("detail", "")
            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {color}55;'
                    f'border-top:3px solid {color};border-radius:7px;'
                    f'padding:9px 11px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">'
                    f'<span style="color:{color};font-size:0.68rem;font-weight:800;letter-spacing:0.8px">{lvl}</span>'
                    f'<span style="color:{color};font-size:0.58rem;opacity:0.85">{badge}</span>'
                    f'</div>'
                    f'<div style="color:{COLORS["text"]};font-size:0.7rem;font-weight:600;'
                    f'line-height:1.35;margin-bottom:5px">{action}</div>'
                    f'<div style="color:{COLORS["text2"]};font-size:0.65rem;line-height:1.4">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if completed:
        st.markdown(
            f'<div style="padding:4px 0 0 0;border-top:1px solid {COLORS["border"]};margin-top:4px">',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{COLORS["text3"]};font-size:0.6rem;text-transform:uppercase;'
            f'letter-spacing:1.2px;padding:6px 14px 2px 14px;font-weight:700">Findings</div>',
            unsafe_allow_html=True)
        for ag in completed:
            chunks = ag.get("chunks_shown", [])
            thinking_html = ""
            if chunks:
                chunk_items = "".join(f'<span class="rchunk">{c}</span>' for c in chunks)
                thinking_html = (
                    f'<details class="rdetails" style="margin-top:8px">'
                    f'<summary>{len(chunks)} reasoning steps</summary>'
                    f'<div class="rterm" style="margin-top:6px;border-radius:6px;border-bottom:none">'
                    f'{chunk_items}'
                    f'</div></details>'
                )
            st.markdown(
                f'<div class="output-final">'
                f'<div class="of-agent">{ag["icon"]} {ag["name"]} · {ag.get("role","")}</div>'
                f'<div class="of-text">{ag["output"]}</div>'
                f'{thinking_html}'
                f'</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── History (previous runs) ──
    if len(history) > (0 if not active else 1):
        past = history[:-1] if not active else history
        if past:
            with st.expander(f"Pipeline history — {len(past)} previous run(s)"):
                for run in reversed(past):
                    rname = responses["pipelines"].get(run["alert_id"], {}).get("trigger_name", run["alert_id"])
                    agents_str = " → ".join(f'{a["icon"]} {a["name"]}' for a in run["agents"])
                    st.markdown(
                        f'<div style="background:{COLORS["surface2"]};border:1px solid {COLORS["border"]};'
                        f'border-radius:7px;padding:8px 12px;margin-bottom:5px;font-size:0.73rem">'
                        f'<div style="color:{COLORS["text3"]};font-size:0.65rem;margin-bottom:3px">'
                        f'{run["alert_id"]} — {rname}</div>'
                        f'<div style="color:{COLORS["text2"]}">{agents_str}</div></div>',
                        unsafe_allow_html=True)

# ── Dashboard ──────────────────────────────────────────────────────────────────
def dashboard_page():
    disk_metrics = load_disk_metrics()
    log_volume   = load_log_volume()
    alerts_data  = load_alerts()

    idx     = st.session_state.sim_index
    current = disk_metrics[min(idx, len(disk_metrics) - 1)]
    total   = len(disk_metrics)

    # Collect new alert notifications
    for a in alerts_data:
        if a["trigger_index"] <= idx and a["id"] not in st.session_state.seen_alert_ids:
            st.session_state.active_alerts.append(a)
            st.session_state.seen_alert_ids.add(a["id"])

    # Check for pipeline triggers at exactly the new sim index
    check_triggers(idx, alerts_data)

    # ── Top bar + sim controls ──
    render_topbar(idx, total, disk_metrics)
    render_sim_controls(idx, total)

    # ── Two-pane main ──────────────────────────────────────────────────────
    left, right = st.columns([5, 4], gap="medium")

    # LEFT: metrics → charts side by side → alerts + events
    with left:
        render_metrics(current, st.session_state.active_alerts)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        ch_left, ch_right = st.columns(2, gap="small")
        with ch_left:
            st.markdown('<div class="slabel">Disk Usage & Write Rate</div>', unsafe_allow_html=True)
            render_disk_chart(disk_metrics, idx)
        with ch_right:
            st.markdown('<div class="slabel">Log Volume by Service</div>', unsafe_allow_html=True)
            render_log_chart(log_volume, idx)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        ev_left, ev_right = st.columns(2, gap="small")
        with ev_left:
            st.markdown('<div class="slabel">Agent Reasoning</div>', unsafe_allow_html=True)
            render_alerts(st.session_state.active_alerts)
        with ev_right:
            st.markdown('<div class="slabel">Event Feed</div>', unsafe_allow_html=True)
            render_events(disk_metrics, idx)

    # RIGHT: agent pipeline
    with right:
        st.markdown('<div class="slabel">Agent Activity</div>', unsafe_allow_html=True)
        render_agent_panel()

    # ── Advance loop ───────────────────────────────────────────────────────
    if st.session_state.pipeline_active:
        # No sleep when LLM is fetching (advance_pipeline blocks during the call)
        if not st.session_state.llm_fetching:
            time.sleep(0.18)
        advance_pipeline()
        st.rerun()
    elif st.session_state.sim_running:
        if idx < total - 1:
            time.sleep(st.session_state.sim_speed)
            st.session_state.sim_index += 1
            st.rerun()
        else:
            st.session_state.sim_running = False
            st.rerun()

# ── Router ─────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    login_page()
else:
    dashboard_page()
