# app.py — Agentic Insurance Policy Q&A (LangGraph Enhanced)
import time, re, json
import streamlit as st
from typing import TypedDict, Annotated, List, Dict, Any, Union
import operator

# LangGraph & LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Local modules
from retriever import get_retriever, numeric_intent, load_vectorstore
from llm_client import LLMClient, safe_json_loads

# === Page Config ===
st.set_page_config(page_title="InsuranceAI Pro (Agentic)", layout="wide")

# (Keep your existing CSS & Hero Section code here...)
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .chat-message { padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .chat-user { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .chat-ai { background-color: #ffffff; border-left: 5px solid #4caf50; }
</style>
""", unsafe_allow_html=True)

# =========================
# 1. DEFINE STATE (The Brain)
# =========================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: Dict[str, Any]      # {age, location, requirements}
    selected_filters: Dict[str, str]  # {type, company, policy}
    next_step: str
    final_output: str                 # To hold the text to display

# =========================
# 2. DEFINE NODES (The Agents)
# =========================
llm_client = LLMClient(model="gemini-1.5-flash", temperature=0.2)

def router_node(state: AgentState):
    """Decides the user's intent based on the last message."""
    messages = state["messages"]
    last_msg = messages[-1].content
    
    system_prompt = (
        "Classify the user's intent into exactly one of these categories:\n"
        "- 'RAG': Questions about policy details, coverage, exclusions, or rules.\n"
        "- 'SUMMARIZE': Requests to summarize, explain, or shorten a policy document.\n"
        "- 'RECOMMEND': User describes needs/requirements and wants a suggestion or ranking.\n"
        "- 'CHAT': Greetings, general chit-chat, or unclear queries.\n\n"
        "Return ONLY the keyword."
    )
    
    intent = llm_client.chat(system=system_prompt, user=last_msg).strip().upper()
    
    # Fallback safety
    valid = ["RAG", "SUMMARIZE", "RECOMMEND", "CHAT"]
    if intent not in valid:
        intent = "RAG"
        
    return {"next_step": intent}

def retrieval_node(state: AgentState):
    """Handles specific questions using Vector Search."""
    query = state["messages"][-1].content
    filters = state["selected_filters"]
    
    # Build filter dict from state
    db_filter = {}
    if filters["type"] != "(All)": db_filter["type"] = filters["type"]
    if filters["company"] != "(All)": db_filter["company"] = filters["company"]
    if filters["policy"] != "(All)": db_filter["policy_name"] = filters["policy"]
    
    # Retrieve
    retriever = get_retriever(k=6, search_type="mmr", filter_dict=db_filter)
    docs = retriever.invoke(query)
    
    if not docs:
        return {"messages": [AIMessage(content="I checked the documents but couldn't find information matching your specific filters.")]}

    # Synthesize
    context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Doc')}] {d.page_content[:800]}" for d in docs])
    
    sys_prompt = "You are an insurance expert. Answer the question using ONLY the context provided. Cite sources."
    user_prompt = f"Question: {query}\n\nContext:\n{context_text}"
    
    response = llm_client.chat(system=sys_prompt, user=user_prompt)
    return {"messages": [AIMessage(content=response)]}

def summarize_node(state: AgentState):
    """Handles full policy summarization requests."""
    filters = state["selected_filters"]
    policy_name = filters.get("policy", "(All)")
    
    if policy_name == "(All)":
        return {"messages": [AIMessage(content="Please select a specific **Policy** from the sidebar to generate a summary.")]}
        
    # Reuse your existing retrieval logic here, simplified
    docs = get_retriever(k=10, filter_dict={"policy_name": policy_name}).invoke("policy benefits coverage exclusions")
    context_text = "\n".join([d.page_content[:1000] for d in docs])
    
    sys_prompt = "Summarize the key benefits, exclusions, and waiting periods of this insurance policy."
    response = llm_client.chat(system=sys_prompt, user=f"Policy: {policy_name}\n\nContext:\n{context_text}")
    
    return {"messages": [AIMessage(content=response)]}

def recommend_node(state: AgentState):
    """Ranks policies based on user needs."""
    user_reqs = state["messages"][-1].content
    filters = state["selected_filters"]
    
    # Fetch candidates (broad search based on type/company)
    db_filter = {}
    if filters["type"] != "(All)": db_filter["type"] = filters["type"]
    if filters["company"] != "(All)": db_filter["company"] = filters["company"]
    
    # We retrieve broadly to find candidates
    docs = get_retriever(k=15, search_type="similarity", filter_dict=db_filter).invoke(user_reqs)
    
    # Group by policy to rank
    candidates = {}
    for d in docs:
        pol = d.metadata.get("policy_name", "Unknown")
        candidates.setdefault(pol, []).append(d.page_content)
    
    ranking_prompt = (
        f"User Requirements: {user_reqs}\n"
        "Rank the following policies based on how well they meet the requirements. "
        "Provide a short reason for the top choice.\n\n"
    )
    
    for pol, texts in list(candidates.items())[:4]: # Limit to 4 policies for context window
        ranking_prompt += f"POLICY: {pol}\nDETAILS: {' '.join(texts[:2])[:500]}\n\n"
        
    response = llm_client.chat(system="You are a helpful recommendation assistant.", user=ranking_prompt)
    return {"messages": [AIMessage(content=response)]}

def chat_node(state: AgentState):
    """General chatter."""
    msg = state["messages"][-1].content
    response = llm_client.chat(system="You are a helpful assistant.", user=msg)
    return {"messages": [AIMessage(content=response)]}

# =========================
# 3. BUILD GRAPH
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag", retrieval_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("recommend", recommend_node)
workflow.add_node("chat", chat_node)

workflow.set_entry_point("router")

def route_logic(state):
    return state["next_step"].lower()

workflow.add_conditional_edges(
    "router",
    route_logic,
    {
        "rag": "rag",
        "summarize": "summarize",
        "recommend": "recommend",
        "chat": "chat"
    }
)

workflow.add_edge("rag", END)
workflow.add_edge("summarize", END)
workflow.add_edge("recommend", END)
workflow.add_edge("chat", END)

app_graph = workflow.compile()

# =========================
# 4. STREAMLIT UI
# =========================

# --- Sidebar ---
st.sidebar.title("Configuration")
# (Load catalog logic kept from your original code)
vs = load_vectorstore()
# ... [Your existing catalog loading logic here to populate lists] ...
# For brevity, I'll use placeholders, please uncomment your original catalog code
types_list = ["(All)", "Health", "Motor"] 
companies_list = ["(All)", "HDFC ERGO", "Tata AIG"] # Replace with your dynamic lists
# ...

selected_type = st.sidebar.selectbox("Insurance Type", types_list)
selected_company = st.sidebar.selectbox("Company", companies_list)
selected_policy = st.sidebar.text_input("Policy Name (Optional)", "")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I am your AI Agent. Ask me about policies, get a summary, or ask for recommendations.")]

# Display History
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run Graph
    inputs = {
        "messages": st.session_state.messages,
        "selected_filters": {
            "type": selected_type,
            "company": selected_company,
            "policy": selected_policy
        },
        "user_profile": {} 
    }
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            final_res = ""
            # Stream the graph updates
            for event in app_graph.stream(inputs):
                for node_name, value in event.items():
                    if "messages" in value:
                        final_res = value["messages"][-1].content
            
            st.markdown(final_res)
            st.session_state.messages.append(AIMessage(content=final_res))
