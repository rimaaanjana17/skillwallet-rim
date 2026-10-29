"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone - Project 1

Run locally:   streamlit run app.py
Deploy:        push app.py + requirements.txt to GitHub -> Streamlit Community Cloud
No database. Works without an API key in demonstration mode.
"""

import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # app still runs in demo mode
    genai = None

# --------------------------------------------------------------------------
# Page setup and styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .3px; color: #1F2D3A; }
.hero { background: #1F2D3A; color: #fff; padding: 22px 28px; border-radius: 6px;
        border-left: 10px solid #F2C200; margin-bottom: 18px; }
.hero h1 { color: #fff !important; margin: 0; font-size: 2.2rem; }
.hero p { margin: 4px 0 0 0; color: #C9D3DB; }
.metric { background: #F4F6F7; border: 1px solid #D5DBDF; border-top: 4px solid #2F6B8F;
          padding: 14px 16px; border-radius: 4px; }
.metric .label { font-size: .8rem; color: #5B6B78; }
.metric .value { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #1F2D3A; line-height: 1.1; }
.pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600; color: #fff; }
.p-ok { background: #2E7D4F; } .p-warn { background: #C98A00; }
.p-bad { background: #B83A2E; } .p-crit { background: #7A1F17; } .p-na { background: #7A8791; }
.alert { padding: 12px 16px; border-radius: 4px; margin: 6px 0; border-left: 6px solid #F2C200; background: #FFF8DB; }
.alert.red { border-left-color: #B83A2E; background: #FBE9E7; }
.alert.blue { border-left-color: #2F6B8F; background: #E6F0F6; }
.demo-banner { background: #FFF8DB; border: 1px dashed #C98A00; padding: 8px 12px; border-radius: 4px; font-size: .88rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
SEVERITY_CLASS = {"Low": "p-ok", "Medium": "p-warn", "High": "p-bad", "Critical": "p-crit"}
STATUS_CLASS = {
    "Compliant": "p-ok", "Non-Compliant": "p-bad", "Not Visible": "p-na",
    "Safe": "p-ok", "Needs Attention": "p-warn", "Unsafe": "p-bad",
    "Monitor": "p-ok", "Repair Soon": "p-warn", "Urgent": "p-bad", "Immediate": "p-crit",
}


def pill(text: str, cls: str = None) -> str:
    cls = cls or STATUS_CLASS.get(text) or SEVERITY_CLASS.get(text, "p-na")
    return f'<span class="pill {cls}">{text}</span>'


def metric(label: str, value: str) -> str:
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'


def inr(amount: float) -> str:
    """Format a rupee amount as Lakhs / Crores."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:,.2f} L"
    return f"₹{amount:,.0f}"


def parse_json(text: str) -> dict:
    """Extract a JSON object from model output, tolerating code fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def load_image(uploaded) -> Image.Image:
    img = Image.open(uploaded).convert("RGB")
    img.thumbnail((1280, 1280))  # keeps API payload small
    return img


def gemini_json(prompt: str, image: Image.Image, api_key: str, model_name: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    response = model.generate_content([prompt, image])
    return parse_json(response.text)


# --------------------------------------------------------------------------
# Prompts (JSON data contracts)
# --------------------------------------------------------------------------
PPE_PROMPT = """You are a certified construction safety auditor (OSHA 1926 and Indian IS 456 / NBC 2016 practice).
Analyze this construction site photo. Audit PPE for every visible worker: helmets, high-visibility vests,
steel-toe boots, safety harnesses (where working at height). Also identify site hazards.
If an item cannot be seen, mark it "Not Visible" - never guess.
Respond with ONLY this JSON:
{
 "safety_score": <int 0-100>,
 "overall_status": "Safe" | "Needs Attention" | "Unsafe",
 "workers_detected": <int>,
 "ppe_detection": {
   "helmet": {"status": "Compliant|Non-Compliant|Not Visible", "note": "<short>"},
   "high_vis_vest": {"status": "...", "note": "..."},
   "steel_toe_boots": {"status": "...", "note": "..."},
   "safety_harness": {"status": "...", "note": "..."}
 },
 "hazards": [{"hazard": "<text>", "severity": "Low|Medium|High|Critical", "remedy": "<corrective action>"}],
 "summary": "<2 sentences>"
}"""

CRACK_PROMPT = """You are a structural engineer inspecting concrete / masonry defects (IS 456, IS 13935, IS 15988).
Classify the defect in this photo as one of: Hairline, Settlement, Structural Shear, Thermal Expansion, Spalling.
Estimate crack width in mm from visual cues, and state your confidence honestly. Respond with ONLY this JSON:
{
 "defect_type": "Hairline|Settlement|Structural Shear|Thermal Expansion|Spalling",
 "estimated_width_mm": <number>,
 "severity_score": <int 0-100, higher = worse>,
 "urgency": "Monitor" | "Repair Soon" | "Urgent" | "Immediate",
 "confidence": "Low|Medium|High",
 "root_causes": ["<cause>", ...],
 "recommended_repairs": [{"method": "<e.g. Epoxy injection, structural re-grouting>", "detail": "<short>"}],
 "monitoring_advice": "<text>",
 "disclaimer": "Photo-based estimate; confirm with an on-site structural engineer."
}"""

# --------------------------------------------------------------------------
# Demonstration-mode data
# --------------------------------------------------------------------------
DEMO_PPE = {
    "safety_score": 62, "overall_status": "Needs Attention", "workers_detected": 4,
    "ppe_detection": {
        "helmet": {"status": "Compliant", "note": "3 of 4 workers wearing helmets"},
        "high_vis_vest": {"status": "Non-Compliant", "note": "2 workers without vests"},
        "steel_toe_boots": {"status": "Not Visible", "note": "Footwear not clearly visible"},
        "safety_harness": {"status": "Non-Compliant", "note": "Worker at edge without harness"},
    },
    "hazards": [
        {"hazard": "Unprotected slab edge", "severity": "Critical", "remedy": "Install edge guardrails and toe boards before work continues."},
        {"hazard": "Loose rebar ends exposed", "severity": "High", "remedy": "Fit rebar caps on all vertical bars."},
        {"hazard": "Debris on walkway", "severity": "Medium", "remedy": "Daily housekeeping; designate material stacking zones."},
    ],
    "summary": "DEMO DATA. Partial PPE compliance with serious fall-protection gaps near the slab edge.",
}
DEMO_CRACK = {
    "defect_type": "Settlement", "estimated_width_mm": 3.5, "severity_score": 58, "urgency": "Repair Soon",
    "confidence": "Medium",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction", "Moisture variation near footing"],
    "recommended_repairs": [
        {"method": "Epoxy injection", "detail": "Seal active cracks after movement is arrested."},
        {"method": "Structural re-grouting", "detail": "Pressure-grout voids under the footing if settlement is confirmed."},
    ],
    "monitoring_advice": "Fix crack gauges across the crack and record width weekly for 6-8 weeks.",
    "disclaimer": "DEMO DATA. Photo-based estimate; confirm with an on-site structural engineer.",
}

# --------------------------------------------------------------------------
# Valuation model (transparent assumptions, editable in the UI)
# --------------------------------------------------------------------------
TIERS = ["Tier 1 (Metro)", "Tier 2", "Tier 3"]
STRUCTURES = {
    # name: (cost INR/sqft by tier, steel kg/sqft, cement bags/sqft)
    "Residential RCC": ([2200, 1800, 1500], 4.0, 0.40),
    "Commercial RCC": ([2800, 2300, 1900], 4.5, 0.45),
    "Industrial Steel Shed": ([2000, 1700, 1400], 7.0, 0.20),
    "Premium Villa": ([3200, 2600, 2100], 4.2, 0.42),
}
MARKET_RATE = [9000, 5500, 3200]  # completed market value INR/sqft by tier
LAND_SHARE = 0.30                  # share of market value that is land
MILESTONES = [
    ("Foundation", 15), ("Plinth & columns", 30), ("Superstructure", 60),
    ("Masonry & MEP", 80), ("Finishing & handover", 100),
]

# --------------------------------------------------------------------------
# Weather advisory
# --------------------------------------------------------------------------
WEATHER = {
    "Normal": {"tone": "blue", "alerts": [
        "Standard operations. Run the morning toolbox talk and PPE check.",
        "Keep hydration points stocked and inspect scaffolds at shift start.",
    ]},
    "High Wind": {"tone": "red", "alerts": [
        "Suspend crane and hoist operations when wind exceeds the manufacturer limit (typically 36 km/h / 10 m/s).",
        "Secure loose sheeting, formwork panels and scaffolding nets; stop work at height.",
        "Lower tower crane jib and set it to weathervane mode.",
    ]},
    "Heavy Rain": {"tone": "red", "alerts": [
        "Electrical hazard: de-energize temporary panels in wet zones, check ELCB/RCCB and cable joints.",
        "Concrete: pause pours; cover fresh concrete with tarpaulin and protect from wash-out.",
        "Excavations: check shoring and dewater; keep personnel away from trench edges.",
        "Slip risk: keep ramps and scaffold boards clear; use anti-skid footwear.",
    ]},
    "Extreme Heat": {"tone": "red", "alerts": [
        "Shift heavy work to early morning and evening; enforce rest breaks in shade every hour.",
        "Provide ORS and water; watch for heat-stress signs (dizziness, cramps, confusion).",
        "Concrete curing: start curing immediately, wet-cure at least 7 days, use curing compound or wet burlap; avoid pours above 38 °C without retarders and chilled water.",
    ]},
}

SYSTEM_PROMPT = (
    "You are a senior construction engineering assistant for Indian sites. Answer concisely and practically, "
    "citing IS codes (IS 456, IS 800, IS 1893, NBC 2016) where relevant. Give safety-first advice. "
    "Say when something needs on-site verification by a licensed engineer. Current weather condition on site: {weather}."
)


def demo_chat_reply(msg: str, weather: str) -> str:
    m = msg.lower()
    if any(k in m for k in ["cure", "curing"]):
        return "Cure concrete for at least 7 days (10+ for blended cements) per IS 456. Keep surfaces continuously moist, shade them in hot weather, and protect them from rain wash-out."
    if any(k in m for k in ["m20", "mix", "grade"]):
        return "M20 nominal mix is roughly 1:1.5:3 (cement:sand:aggregate) with a water-cement ratio near 0.5. Use design mixes for M25 and above."
    if any(k in m for k in ["crack", "settlement"]):
        return "Measure crack width with a gauge and track it weekly. Cracks under 0.3 mm are usually cosmetic; wider or growing cracks need a structural engineer's review before epoxy injection."
    if any(k in m for k in ["helmet", "ppe", "harness"]):
        return "Helmets, hi-vis vests and safety footwear are mandatory on site. Use a full-body harness anchored above the head whenever working at 1.8 m or higher on an unprotected edge."
    if any(k in m for k in ["cost", "budget", "price"]):
        return "Use the Planning & Valuation tab for a quick estimate. Typical RCC residential costs run about ₹1,500-2,200 per sq ft depending on city tier and finishes."
    return f"(Demo mode) Current site condition is **{weather}**. Add a Gemini API key in the sidebar for full, context-aware answers. Try asking about curing, mix design, cracks or PPE."


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google Gemini API key", type="password",
                            help="Get a free key at aistudio.google.com. Never share or commit it.")
    model_name = st.text_input("Gemini model", value="gemini-2.0-flash",
                               help="Any vision-capable Gemini model name.")
    live = bool(api_key) and genai is not None
    if live:
        st.success("Live mode: Gemini connected")
    else:
        st.warning("Demonstration mode: sample outputs are shown. Add an API key for real analysis.")
    st.divider()
    st.subheader("Architecture")
    st.markdown(
        "- **Input:** site photos, project parameters\n"
        "- **Vision AI:** Gemini multimodal prompts → JSON contracts\n"
        "- **Rules engine:** cost, valuation and weather protocols\n"
        "- **Output:** metric cards, JSON reports, chatbot\n"
        "- **State:** in-session only, no database"
    )
    st.divider()
    st.subheader("Navigation")
    st.markdown("1. Safety & PPE Audit\n2. Crack Diagnostics\n3. Planning & Valuation\n4. Site Assistant")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>Multimodal Construction Intelligence Platform</h1>'
    "<p>Site photos in. Safety scores, defect diagnosis, INR budgets and weather protocols out.</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Safety & PPE Audit", "Crack Diagnostics", "Planning & Valuation", "Site Assistant",
])

# --------------------------------------------------------------------------
# Tab 1 - PPE audit
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Site Safety & PPE Audit")
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="ppe_up")
    if up:
        img = load_image(up)
        col_img, col_btn = st.columns([2, 1])
        col_img.image(img, caption="Uploaded site photo", use_container_width=True)
        run = col_btn.button("Run safety audit", type="primary", key="ppe_run")
        if run:
            with st.spinner("Auditing PPE and hazards..."):
                try:
                    if live:
                        st.session_state["ppe"] = gemini_json(PPE_PROMPT, img, api_key, model_name)
                        st.session_state["ppe_demo"] = False
                    else:
                        st.session_state["ppe"] = DEMO_PPE
                        st.session_state["ppe_demo"] = True
                except Exception as e:
                    st.error(f"Analysis failed: {e}. Check your API key and model name, or try a different image.")
    else:
        st.info("Upload a JPG or PNG to start the audit.")

    r = st.session_state.get("ppe")
    if r:
        if st.session_state.get("ppe_demo"):
            st.markdown('<div class="demo-banner">Demonstration data. Add a Gemini key for a real audit of your photo.</div>', unsafe_allow_html=True)
        score = r.get("safety_score", 0)
        status = r.get("overall_status", "Needs Attention")
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric("Safety score", f"{score}/100"), unsafe_allow_html=True)
        c2.markdown(metric("Workers detected", str(r.get("workers_detected", "-"))), unsafe_allow_html=True)
        c3.markdown(metric("Overall status", pill(status)), unsafe_allow_html=True)
        st.progress(min(max(int(score), 0), 100) / 100)
        st.write(r.get("summary", ""))

        st.markdown("#### PPE checklist")
        labels = {"helmet": "Helmet", "high_vis_vest": "High-vis vest",
                  "steel_toe_boots": "Steel-toe boots", "safety_harness": "Safety harness"}
        cols = st.columns(4)
        for col, (key, label) in zip(cols, labels.items()):
            item = r.get("ppe_detection", {}).get(key, {"status": "Not Visible", "note": ""})
            col.markdown(f"**{label}**<br>{pill(item.get('status', 'Not Visible'))}<br><small>{item.get('note', '')}</small>",
                         unsafe_allow_html=True)

        st.markdown("#### Identified hazards")
        for h in r.get("hazards", []):
            sev = h.get("severity", "Medium")
            st.markdown(f"{pill(sev)} **{h.get('hazard', '')}**<br><small>Remedy: {h.get('remedy', '')}</small>",
                        unsafe_allow_html=True)
            st.write("")
        st.download_button("Download JSON report", json.dumps(r, indent=2),
                           file_name=f"ppe_report_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")

# --------------------------------------------------------------------------
# Tab 2 - Crack diagnostics
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    up2 = st.file_uploader("Upload a concrete or wall surface photo", type=["jpg", "jpeg", "png"], key="crack_up")
    if up2:
        img2 = load_image(up2)
        col_img, col_btn = st.columns([2, 1])
        col_img.image(img2, caption="Uploaded surface", use_container_width=True)
        if col_btn.button("Diagnose defect", type="primary", key="crack_run"):
            with st.spinner("Classifying defect..."):
                try:
                    if live:
                        st.session_state["crack"] = gemini_json(CRACK_PROMPT, img2, api_key, model_name)
                        st.session_state["crack_demo"] = False
                    else:
                        st.session_state["crack"] = DEMO_CRACK
                        st.session_state["crack_demo"] = True
                except Exception as e:
                    st.error(f"Diagnosis failed: {e}. Check your API key and model name, or try a different image.")
    else:
        st.info("Upload a close-up photo of the crack or surface.")

    c = st.session_state.get("crack")
    if c:
        if st.session_state.get("crack_demo"):
            st.markdown('<div class="demo-banner">Demonstration data. Add a Gemini key to diagnose your photo.</div>', unsafe_allow_html=True)
        a, b, d, e = st.columns(4)
        a.markdown(metric("Defect type", c.get("defect_type", "-")), unsafe_allow_html=True)
        b.markdown(metric("Est. width", f"{c.get('estimated_width_mm', '-')} mm"), unsafe_allow_html=True)
        d.markdown(metric("Severity score", f"{c.get('severity_score', '-')}/100"), unsafe_allow_html=True)
        e.markdown(metric("Urgency", pill(c.get("urgency", "Monitor"))), unsafe_allow_html=True)
        st.caption(f"Model confidence: {c.get('confidence', '-')}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Probable root causes")
            for cause in c.get("root_causes", []):
                st.markdown(f"- {cause}")
        with right:
            st.markdown("#### Recommended repairs")
            for rep in c.get("recommended_repairs", []):
                st.markdown(f"- **{rep.get('method', '')}**: {rep.get('detail', '')}")
        st.markdown(f"**Monitoring:** {c.get('monitoring_advice', '')}")
        st.caption(c.get("disclaimer", ""))
        st.download_button("Download JSON report", json.dumps(c, indent=2),
                           file_name=f"crack_report_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")

# --------------------------------------------------------------------------
# Tab 3 - Planning & valuation
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Project Planning & Real Estate Valuation")
    i1, i2, i3 = st.columns(3)
    structure = i1.selectbox("Structure type", list(STRUCTURES.keys()))
    sqft = i2.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000, value=2500, step=100)
    tier_name = i3.selectbox("Location tier", TIERS)
    completion = st.slider("Completion (%)", 0, 100, 40)

    tier_idx = TIERS.index(tier_name)
    rates, steel_per_sqft, cement_per_sqft = STRUCTURES[structure]
    with st.expander("Rate assumptions (edit to match your market)"):
        cost_rate = st.number_input("Construction cost (₹/sq ft)", value=int(rates[tier_idx]), step=100)
        mkt_rate = st.number_input("Completed market value (₹/sq ft)", value=int(MARKET_RATE[tier_idx]), step=250)
        land_share = st.slider("Land share of market value", 0.0, 0.6, LAND_SHARE, 0.05)

    budget = sqft * cost_rate
    capex = budget * completion / 100
    full_value = sqft * mkt_rate
    current_value = full_value * land_share + full_value * (1 - land_share) * completion / 100
    steel_total = sqft * steel_per_sqft / 1000  # tonnes
    cement_total = sqft * cement_per_sqft       # bags

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric("Total project budget", inr(budget)), unsafe_allow_html=True)
    m2.markdown(metric("Capex incurred", inr(capex)), unsafe_allow_html=True)
    m3.markdown(metric("Current asset value", inr(current_value)), unsafe_allow_html=True)
    m4.markdown(metric("Value on completion", inr(full_value)), unsafe_allow_html=True)

    st.write("")
    s1, s2 = st.columns(2)
    s1.markdown(metric("Steel required", f"{steel_total:,.1f} t (used {steel_total * completion / 100:,.1f} t)"), unsafe_allow_html=True)
    s2.markdown(metric("Cement required", f"{cement_total:,.0f} bags (used {cement_total * completion / 100:,.0f})"), unsafe_allow_html=True)

    st.markdown("#### Milestone progress")
    st.progress(completion / 100)
    prev = 0
    for name, upto in MILESTONES:
        if completion >= upto:
            state = "✅ Done"
        elif completion > prev:
            state = f"🔧 In progress ({(completion - prev) / (upto - prev) * 100:.0f}%)"
        else:
            state = "⏳ Upcoming"
        st.markdown(f"- **{name}** ({prev}-{upto}%): {state}")
        prev = upto

    st.markdown("#### Conceptual 3D rendering")
    style = st.selectbox("Rendering style", ["Photorealistic architectural render", "Blueprint isometric", "Modern minimalist", "Indian contemporary"])
    prompt = (f"{style} of a {structure.lower()} building, {sqft} sq ft, {tier_name} Indian city, "
              f"{'under construction, scaffolding and cranes' if completion < 100 else 'completed'}, golden hour, 3D, high detail")
    st.text_area("Generated prompt", prompt, height=70)
    if st.button("Generate concept image"):
        url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) + "?width=1024&height=640&model=flux&nologo=true"
        st.image(url, caption="Concept rendering via Pollinations (Flux). Illustrative only.", use_container_width=True)
        st.caption("If the image does not load, open this link directly: " + url)
    st.caption("Estimates use editable benchmark rates and are indicative only. Get a quantity surveyor or registered valuer for real decisions.")

# --------------------------------------------------------------------------
# Tab 4 - Weather advisory and chatbot
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Today's site condition", list(WEATHER.keys()), horizontal=True)
    for a in WEATHER[weather]["alerts"]:
        st.markdown(f'<div class="alert {WEATHER[weather]["tone"]}">{a}</div>', unsafe_allow_html=True)

    st.markdown("#### Ask the site engineer")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                                      "content": "Hi, I'm your site engineering assistant. Ask about safety, curing, mix design, cracks, or today's work plan."}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_msg := st.chat_input("Ask a construction question..."):
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if live:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT.format(weather=weather))
                        history = [
                            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                            for m in st.session_state.messages[1:-1]  # skip greeting and the new message
                        ]
                        reply = model.start_chat(history=history).send_message(user_msg).text
                    else:
                        reply = demo_chat_reply(user_msg, weather)
                except Exception as e:
                    reply = f"Sorry, I couldn't get a response ({e}). Check your API key and model name."
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()
