import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image

company_logo = Image.open("Images/Green and Black Simple Clean Vegan Food Logo.png")

load_dotenv()

st.set_page_config(
    page_title="ShelfSense",
    page_icon=company_logo,
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F5F0;
}

.block-container {
    max-width: 760px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.shelfsense-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #1C1C1A;
    margin: 0;
    line-height: 1;
}

.shelfsense-subtitle {
    font-size: 13px;
    color: #8A8478;
    margin: 3px 0 0 0;
    font-weight: 300;
    letter-spacing: 0.02em;
}

.chat-bubble-user {
    background: #1C1C1A;
    color: #F7F5F0;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
    max-width: 85%;
    margin-left: auto;
    display: block;
}

.chat-bubble-assistant {
    background: #FFFFFF;
    color: #1C1C1A;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
    border: 1px solid #E2DDD5;
    max-width: 90%;
}

.chat-bubble-assistant pre {
    background: #F7F5F0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    overflow-x: auto;
    border: 1px solid #E2DDD5;
}

.role-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8A8478;
    margin-bottom: 4px;
}

.role-label-user {
    text-align: right;
}

.welcome-card {
    background: #FFFFFF;
    border: 1px solid #E2DDD5;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.welcome-card p {
    font-size: 14px;
    color: #5C5850;
    line-height: 1.7;
    margin: 0 0 1rem 0;
}

.suggestion-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.suggestion-chip {
    background: #F7F5F0;
    border: 1px solid #E2DDD5;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #5C5850;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
}

.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #E2DDD5 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1C1C1A !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1C1C1A !important;
    box-shadow: none !important;
}

.stButton > button {
    background: #1C1C1A !important;
    color: #F7F5F0 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    height: auto !important;
    transition: opacity 0.15s ease !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.clear-btn > button {
    background: transparent !important;
    color: #8A8478 !important;
    border: 1px solid #E2DDD5 !important;
    font-size: 12px !important;
    padding: 6px 14px !important;
}

div[data-testid="stVerticalBlock"] > div:has(.stTextInput) {
    padding-top: 0.5rem;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# RAG LOADER
# Loads the ShelfSense knowledge base and injects it into
# the system prompt so the agent has industry-grounded
# knowledge for every conversation.
# ============================================================

def load_rag_document(filepath: str = "tools/shelfsense_rag.txt") -> str:
    """Loads the RAG knowledge base from the tools folder."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

rag_document = load_rag_document()

rag_notice = "(Knowledge base loaded ✓)" if rag_document else "(Knowledge base not found — running without RAG)"


# ============================================================
# SYSTEM PROMPT — injected with RAG document
# ============================================================

system_prompt = f'''
You are ShelfSense, an expert AI agent for small and mid-sized restaurant owners and kitchen managers. Your job is to help them forecast customer demand, manage inventory levels, and generate smart ingredient shopping lists — so they never over-order perishables or run out of popular menu items mid-service.

You have access to the following tools:
- search_local_factors: Searches for local events, weather forecasts, holidays, and other factors that could impact restaurant traffic in the user's area. Use this FIRST after collecting sales and inventory data.
- forecast_demand: Takes historical sales data and applies event/weather/holiday multipliers to project demand for each menu item and ingredient over the next 7 days. Use AFTER search_local_factors.
- generate_shopping_list: Cross-references forecasted demand with current inventory levels to produce a prioritized, quantified ingredient shopping list. Use AFTER forecast_demand.

---

SHELFSENSE KNOWLEDGE BASE
Use this reference data to give accurate, industry-grounded advice. It contains
standard portion ratios, PAR level thresholds, demand multipliers, waste benchmarks,
shelf life data, and restaurant industry terminology.

{rag_document if rag_document else "Knowledge base unavailable — use general restaurant industry knowledge."}

---

BEHAVIOR GUIDELINES

Always begin by collecting the following information if not already provided:
1. The restaurant's location (city/neighborhood)
2. Recent sales data (e.g. how many of each dish was sold last week)
3. Current inventory levels (e.g. how much of each ingredient is on hand)
4. Any known upcoming events or closures

If any of this information is missing, ask clarifying questions before proceeding.
Do not generate a forecast or shopping list without sufficient data.

Once you have enough information, follow this exact sequence:
1. Use search_local_factors to find local events, weather, and holidays for the upcoming week.
2. Use forecast_demand to project demand for each menu item and ingredient, using the multipliers from the knowledge base.
3. Use generate_shopping_list to translate forecasted demand into specific ingredient quantities, applying the PAR level thresholds and safety buffers from the knowledge base.

---

OUTPUT FORMAT

Structure every final response into three clearly labeled sections:

DEMAND FORECAST — Next 7 Days
List each menu item with projected demand and a percentage change vs. last week.
Note the reason for any significant changes (e.g. local event, holiday, weather).

RECOMMENDED SHOPPING LIST
List each ingredient with the exact quantity to order and its priority level
(CRITICAL / HIGH / NORMAL / SKIP). Flag any items at risk of stockout or spoilage.

FLAGS & INSIGHTS
Highlight anything the restaurant owner should pay attention to — unexpected demand
spikes, waste risks, slow-moving items, shelf life warnings, or ordering tips from
the knowledge base.

---

TONE & STYLE

- Be direct, practical, and concise. Restaurant owners are busy — get to the point.
- Use plain language and standard restaurant industry terminology where appropriate.
- Be confident in your recommendations but always flag uncertainty when data is limited.
- If the user provides vague or incomplete data, ask one focused follow-up question
  at a time rather than overwhelming them with a list of questions.

---

CONSTRAINTS

- Never make up sales data or inventory numbers. Only work with what the user provides.
- Always use the portion ratios, PAR thresholds, and multipliers from the knowledge base
  rather than guessing. If a ratio isn't in the knowledge base, state your assumption.
- Always prioritize reducing food waste and preventing stockouts as the two core goals.
- Do not provide legal, financial, or health/safety compliance advice.
'''


# ============================================================
# STREAMLIT APP
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Header
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("Images/Green and Black Simple Clean Vegan Food Logo.png", width=52)
    except:
        st.markdown("🥬")
with col2:
    st.markdown(f"""
        <div style="padding-top: 4px;">
            <p class="shelfsense-title">ShelfSense</p>
            <p class="shelfsense-subtitle">Restaurant inventory & demand intelligence &nbsp;·&nbsp; {rag_notice}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='border-bottom: 1px solid #E2DDD5; margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# Welcome card
visible_messages = [m for m in st.session_state.messages if m["role"] != "system"]
if len(visible_messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <p>Hi! I'm ShelfSense — your inventory and demand forecasting assistant. Tell me about your restaurant's recent sales and current stock, and I'll help you figure out exactly what to order for the week ahead.</p>
        <p style="margin-bottom: 0.5rem; font-size: 12px; color: #8A8478; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;">Try asking</p>
        <div class="suggestion-row">
            <span class="suggestion-chip">📊 Forecast demand for next week</span>
            <span class="suggestion-chip">🛒 Generate my shopping list</span>
            <span class="suggestion-chip">⚠️ What's at risk of running out?</span>
            <span class="suggestion-chip">📦 What are my PAR levels?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for msg in visible_messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="role-label role-label-user">You</div><div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="role-label">ShelfSense</div><div class="chat-bubble-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="message",
        placeholder="Describe your sales, inventory, or ask a question...",
        label_visibility="collapsed",
        key="user_input"
    )
with col_send:
    send = st.button("Send", use_container_width=True)

# Clear button
if len(visible_messages) > 0:
    st.markdown("<div class='clear-btn'>", unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Handle send
if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    with st.spinner("Thinking..."):
        response = st.session_state.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
