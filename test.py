def ask_chat_question(question):
    """Process a chat question and get AI response"""
    if 'last_report' not in st.session_state:
        st.error("No analysis report available. Please run an analysis first.")
        return
    
    try:
        # Initialize Gemini AI
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create context-aware prompt
        chat_prompt = f"""You are a financial advisor AI assistant. A user has received the following investment analysis report:

{st.session_state['last_report']}

The user is asking: {question}

Provide a clear, concise answer based on the analysis report. Be specific and reference details from the report."""
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = llm.invoke(chat_prompt)
        
        # Store in chat history ONLY - don't display here
        st.session_state.chat_history.append({
            'question': question,
            'answer': response.content
        })
        
    except Exception as e:
        st.error(f"Chat error: {str(e)}")
