def display_chat_assistant():
    st.markdown("---")
    st.markdown('<h3 class="section-header" style="font-size: 1.5rem;">💬 AI Chat Assistant</h3>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <p style="color: #667eea; font-weight: 600; margin: 0;">
                Ask me anything about the analysis report! I can explain recommendations, risks, or any other details.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Quick question buttons
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ Why this recommendation?", use_container_width=True):
            ask_chat_question(f"Why is {st.session_state.get('last_stock', 'this stock')} rated with this recommendation?")
    
    with col2:
        if st.button("⚠️ What are the main risks?", use_container_width=True):
            ask_chat_question(f"What are the main risks for {st.session_state.get('last_stock', 'this stock')}?")
    
    with col3:
        if st.button("🎯 What's the price target?", use_container_width=True):
            ask_chat_question(f"What is the price target for {st.session_state.get('last_stock', 'this stock')} and how was it calculated?")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("**💬 Chat History:**")
        for i, chat in enumerate(st.session_state.chat_history):
            # User question - styled like a message bubble
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.2rem; 
                            border-radius: 15px; 
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                    <strong style="color: white; font-size: 1.1rem;">👤 You:</strong><br>
                    <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1rem; line-height: 1.6;">{chat['question']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # AI response - styled like a card
            st.markdown(f"""
                <div style="background: white; 
                            padding: 1.5rem; 
                            border-radius: 15px; 
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                            border-left: 5px solid #667eea;">
                    <strong style="color: #764ba2; font-size: 1.1rem;">🤖 AI Assistant:</strong><br>
                    <div style="color: #333; margin: 0.8rem 0 0 0; font-size: 1rem; line-height: 1.8;">{chat['answer']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Custom question input
    user_question = st.text_input(
        "Ask your question:",
        placeholder="e.g., What factors influenced the recommendation?",
        key="chat_input"
    )
    
    if st.button("📤 Send", use_container_width=True):
        if user_question:
            ask_chat_question(user_question)
            # Don't use st.rerun() - just let the response display
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

