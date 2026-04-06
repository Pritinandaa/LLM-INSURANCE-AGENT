import streamlit as st
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
 
# LangChain & AI Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from docx import Document
 
# --- INITIALIZE ENVIRONMENT ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
 
# --- PAGE CONFIG ---
st.set_page_config(page_title="Acuity AI", layout="wide", initial_sidebar_state="expanded")
 
# --- CUSTOM CSS (Corporate Dark Theme) ---
st.markdown("""
<style>
    .stApp { background-color: #0D1117; color: #E6E6E6; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* Upload Box Styling */
    .upload-container {
        border: 2px dashed #30363D;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        margin-top: 50px;
    }
 
    /* KPI Header */
    .kpi-container {
        display: flex;
        justify-content: space-around;
        padding: 15px;
        background: rgba(0, 150, 199, 0.1);
        border-radius: 12px;
        border: 1px solid #0096C7;
        margin-bottom: 25px;
    }
    .kpi-box { text-align: center; }
    .kpi-value { font-size: 1.8rem; font-weight: bold; color: #0096C7; }
</style>
""", unsafe_allow_html=True)
 
# --- REPORT EXPORT ---
def build_report(messages):
    doc = Document()
    doc.add_heading("Acuity AI — Portfolio Intelligence Report", level=0)
    doc.add_paragraph(f"Generated from {len(messages)} conversation turns.")
    doc.add_paragraph("")
    for m in messages:
        role = "User" if m["role"] == "user" else "Acuity AI"
        doc.add_heading(role, level=2)
        doc.add_paragraph(m["content"])
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# --- SIDEBAR (Control Panel Only) ---
with st.sidebar:
    st.markdown("<h1 style='color: #0096C7;'>🛡️ Acuity AI</h1>", unsafe_allow_html=True)
    st.caption("Strategic Control Panel")
    st.divider()
 
    st.markdown("### 🖥️ Display Controls")
    show_heatmap = st.toggle("Risk Heatmap", value=False)
    show_gauge = st.toggle("Health Gauge", value=False)
    show_table = st.toggle("Detailed Logs", value=False)
 
    st.divider()
    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
# --- MAIN SCREEN LOGIC ---
 
# 1. Check for API Key first
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please ensure your .env file is configured correctly.")
    st.stop()
 
# 2. Main Title
st.markdown("<h1 style='text-align: center; color: #0096C7;'>Enterprise Portfolio Intelligence</h1>", unsafe_allow_html=True)
 
# 3. File Upload (Conditional Rendering)
uploaded_pdf = st.file_uploader("Upload Portfolio PDF", type="pdf", label_visibility="collapsed")
 
if not uploaded_pdf:
    # Display Welcome Screen if no file is present
    st.markdown("""
    <div class="upload-container">
        <h2 style='color: #E6E6E6;'>Welcome to Acuity AI</h2>
        <p style='color: #8B949E;'>To begin your intelligence analysis, please upload your project portfolio or executive PDF report above.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # --- PROCESSING LOGIC (Runs once per file) ---
    @st.cache_resource
    def process_data(file):
        with open("active_portfolio.pdf", "wb") as f:
            f.write(file.getbuffer())
        
        loader = PyPDFLoader("active_portfolio.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        
        # Simple Risk Scoring logic
        RISK_MAP = {"delay": 2, "over budget": 3, "critical path": 2, "risk": 1, "slippage": 2}
        risk_list = []
        for i, d in enumerate(docs):
            s = min(sum(v for k, v in RISK_MAP.items() if k in d.page_content.lower()), 10)
            risk_list.append({"Chunk": i, "Score": s, "Content": d.page_content[:150]})
        
        df = pd.DataFrame(risk_list)
        
        # LLM & Vector Store
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector = FAISS.from_documents(docs, embeddings)
        
        return df, vector.as_retriever(), llm
 
    with st.spinner("Analyzing Portfolio Architecture..."):
        risk_df, retriever, llm = process_data(uploaded_pdf)
    
    # Calculate Metrics
    avg_score = risk_df["Score"].mean()
    health_idx = max(0, 100 - (avg_score * 12))
 
    # --- ACTIVE DASHBOARD UI ---
    
    # KPI Bar
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box">Portfolio Health<br><span class="kpi-value">{health_idx:.0f}%</span></div>
        <div class="kpi-box">Avg Risk Score<br><span class="kpi-value">{avg_score:.1f}</span></div>
        <div class="kpi-box">Analysis Chunks<br><span class="kpi-value">{len(risk_df)}</span></div>
    </div>
    """, unsafe_allow_html=True)
 
    # Layout: Split vs Single
    any_viz = any([show_heatmap, show_gauge, show_table])
    
    if any_viz:
        col_viz, col_chat = st.columns([1, 1], gap="large")
    else:
        col_chat = st.container()
 
    # VISUALS COLUMN
    if any_viz:
        with col_viz:
            st.subheader("📊 Analytical Overlays")
            if show_heatmap:
                fig = px.imshow(np.array([risk_df["Score"]]), color_continuous_scale="Reds")
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            if show_gauge:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=health_idx,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#0096C7"}}))
                gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(gauge, use_container_width=True)
            
            if show_table:
                st.dataframe(risk_df, hide_index=True, use_container_width=True)
 
    # CHAT COLUMN
    with (col_chat if any_viz else st.container()):
        st.subheader("💬 Acuity AI Analyst")
        
        chat_container = st.container(height=550 if any_viz else 650)
        with chat_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
 
        if prompt := st.chat_input("Analyze specific project risks or financial dependencies..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    # RAG Context Retrieval
                    context_docs = retriever.invoke(prompt)
                    context_text = "\n".join([d.page_content for d in context_docs])
                    
                    full_p = f"System: You are Acuity AI, a Portfolio Analyst. Use context.\nContext: {context_text}\nUser: {prompt}"
                    response = llm.invoke(full_p)
                    st.markdown(response.content)
            
            st.session_state.messages.append({"role": "assistant", "content": response.content})

        # Download Report Button
        if st.session_state.messages:
            st.download_button(
                label="📥 Download Report",
                data=build_report(st.session_state.messages),
                file_name="AcuityAI_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
 