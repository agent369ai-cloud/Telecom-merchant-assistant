import streamlit as st
import requests

API_URL = "http://localhost:8000/v1/agent/chat"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Merchant Support",
    page_icon="📡",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean chat bubbles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.user-bubble {
    background: #0078D4;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 60px;
    font-size: 15px;
}
.agent-bubble {
    background: #F0F2F6;
    color: #1a1a1a;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 60px 6px 0;
    font-size: 15px;
}
.blocked-bubble {
    background: #FFF0F0;
    color: #CC0000;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 60px 6px 0;
    font-size: 15px;
    border-left: 4px solid #CC0000;
}
.sidebar-label { font-size: 13px; color: #666; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — merchant identity
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Merchant Portal")
    st.divider()

    merchant_id = st.text_input("Merchant ID", value="shop-9921")

    tier = st.selectbox(
        "Merchant Tier",
        options=["Gold", "Platinum", "Silver", "Standard"],
        index=0,
    )

    st.divider()
    st.markdown("**Connected to**")
    st.markdown("🤖 `llama-3.3-70b` via Groq")
    st.markdown("🗄️ ChromaDB (Vector Search)")
    st.markdown("🛡️ Guardrails: Active")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Telecom Merchant Support Engine v1.0")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Telecom Merchant Support")
st.caption(f"Logged in as **{merchant_id}** · Tier: **{tier}**")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "agent",
        "content": f"👋 Welcome! I'm your Telecom Merchant Support assistant. How can I help you today, **{merchant_id}**?",
    })

# Render existing messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "agent":
        st.markdown(f'<div class="agent-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "blocked":
        st.markdown(f'<div class="blocked-bubble">🚫 {msg["content"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input box
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about policies, returns, inventory, super sale...")

if user_input and user_input.strip():
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    # Call FastAPI backend
    with st.spinner("Agent is thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "merchant_id": merchant_id,
                    "tier": tier,
                    "message": user_input,
                },
                timeout=30,
            )
            data = response.json()

            if response.status_code == 200:
                agent_reply = data.get("agent_response", "No response.")

                # Detect blocked/security messages
                if agent_reply.startswith("Security Alert"):
                    st.session_state.messages.append({"role": "blocked", "content": agent_reply})
                    st.markdown(f'<div class="blocked-bubble">🚫 {agent_reply}</div>', unsafe_allow_html=True)
                else:
                    st.session_state.messages.append({"role": "agent", "content": agent_reply})
                    st.markdown(f'<div class="agent-bubble">{agent_reply}</div>', unsafe_allow_html=True)
            else:
                err = data.get("detail", "Unknown error")
                st.error(f"API Error {response.status_code}: {err}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
