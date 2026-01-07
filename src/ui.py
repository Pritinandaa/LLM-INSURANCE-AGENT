import streamlit as st
import requests
import json
import time

# Page config
st.set_page_config(
    page_title="AI Underwriting Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .agent-message {
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #4F46E5;
        background-color: #EEF2FF;
        border-radius: 4px;
        color: #1f2937; /* Force dark text for readability on light bg */
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Underwriting AI")
    st.markdown("---")
    st.markdown("### 🤖 Active Agents")
    st.markdown("- 📧 Email Parser")
    st.markdown("- 🏭 Industry Classifier")
    st.markdown("- 💰 Rate Discovery")
    st.markdown("- ⚖️ Risk Assessment")
    st.markdown("- 📝 Quote Generator")
    st.markdown("---")
    st.info("Powered by Microsoft AutoGen & Google Vertex AI")

# Main Content
st.title("🛡️ Intelligent Insurance Underwriting")
st.markdown("Generate comprehensive insurance quotes in seconds using our multi-agent autonomous system.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Broker Email Request")
    default_email = """Subject: Quote Request

Hi,

I would like to request a quote for TechNova Solutions, a software development company based in San Francisco, CA.
We have an annual revenue of approximately $2,000,000 and 15 full-time employees.
We are looking for General Liability and Cyber Insurance coverage.

Thanks,
John Doe"""
    
    email_input = st.text_area("Paste email content here", value=default_email, height=300)
    
    if st.button("Generate Quote 🚀"):
        if not email_input:
            st.error("Please enter email content.")
        else:
            with st.spinner("🤖 AutoGen Agents are collaborating..."):
                start_time = time.time()
                try:
                    # Call FastAPI backend
                    response = requests.post(
                        "http://localhost:8000/api/quotes/process",
                        json={"email_content": email_input},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state['quote_data'] = data
                        st.session_state['processing_time'] = time.time() - start_time
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}. Is the backend running?")

with col2:
    if 'quote_data' in st.session_state:
        data = st.session_state['quote_data']
        
        st.subheader("✅ Generated Quote")
        
        # Top level metrics
        m1, m2, m3 = st.columns(3)
        
        premium = data.get('total_premium')
        if premium is None: premium = 0
            
        with m1:
            st.metric("Total Premium", f"${premium:,}")
        with m2:
            st.metric("Risk Score", f"{data.get('risk_score') or 'N/A'}/100")
        with m3:
            st.metric("Risk Level", data.get('risk_level', 'Unknown'))
            
        st.markdown("---")
        
        # Activity Log (Simulated visuals for agents)
        with st.expander("Show Agent Activity", expanded=True):
            if data.get('processing_mode') == 'microsoft_autogen':
                client_name = data.get('client_name', 'Client')
                risk_level = data.get('risk_level', 'Calculated')
                
                st.markdown(f'<div class="agent-message">📧 <b>Email Parser</b>: Extracted client details for {client_name}.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-message">🏭 <b>Industry Classifier</b>: Analyzed business description and assigned industry codes.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-message">💰 <b>Rate Discovery</b>: Retrieved current market rates for {risk_level} risk profile.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-message">⚖️ <b>Risk Assessment</b>: Determine {risk_level} risk score based on revenue/employee data.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-message">📝 <b>Quote Generator</b>: Finalized premium at ${premium:,}.</div>', unsafe_allow_html=True)

        st.markdown("### 📋 Premium Breakdown")
        breakdown = data.get('premium_breakdown', [])
        if breakdown:
            st.table(breakdown)
        else:
            st.info("No breakdown details available.")
            
        with st.expander("View Raw JSON Response"):
            st.json(data)

    else:
        st.info("👈 Enter email details and click Generate to see the agents in action.")
        st.image("https://raw.githubusercontent.com/microsoft/autogen/main/website/static/img/autogen_agentchat.png", caption="Multi-Agent Workflow", use_container_width=True)
