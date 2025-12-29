def stock_analysis_section():
    st.markdown('<h2 class="section-header">📈 AI-Powered Stock Analysis</h2>', unsafe_allow_html=True)
    
    # Sidebar Inputs
    st.sidebar.markdown("<h3 style='color: white;'>⚙️ Analysis Settings</h3>", unsafe_allow_html=True)
    stock_symbol = st.sidebar.text_input("🎯 Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL, MSFT)")
    time_period = st.sidebar.selectbox("📅 Time Period", ['3mo', '6mo', '1y', '2y', '5y'], index=2)
    indicators = st.sidebar.multiselect(
        "📊 Technical Indicators", 
        ['Moving Averages', 'Volume', 'RSI', 'MACD'],
        default=['Volume']
    )
    
    analyze_button = st.sidebar.button("🚀 Analyze Stock", use_container_width=True)
    
    # FIX: Display previous analysis if it exists in session state
    if 'last_report' in st.session_state and 'last_stock_data' in st.session_state:
        # Show previous stock data and chart
        stock_data = st.session_state['last_stock_data']
        
        col1, col2, col3, col4 = st.columns(4)
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
        price_change_pct = (price_change / stock_data['Close'].iloc[0]) * 100
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${current_price:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Change</div>
                    <div class="metric-value">{price_change_pct:+.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High</div>
                    <div class="metric-value">${stock_data['High'].max():.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Low</div>
                    <div class="metric-value">${stock_data['Low'].min():.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Show chart
        st.plotly_chart(plot_stock_chart(stock_data, indicators), use_container_width=True)
        
        # Show report
        st.write(st.session_state['last_report'])
        
        st.markdown("""
            <div class="analysis-card">
                <h3 style="margin-top: 0;">✅ Analysis Complete!</h3>
                <p>Our AI agents have successfully analyzed {stock_symbol} across multiple dimensions.</p>
            </div>
        """.format(stock_symbol=st.session_state.get('last_stock', stock_symbol)), unsafe_allow_html=True)
        
        # Show chat assistant
        display_chat_assistant()
    
    if analyze_button:
        # Fetch stock data
        with st.spinner(f"🔍 Fetching data for {stock_symbol}..."):
            stock_data = yf.Ticker(stock_symbol).history(period=time_period, interval="1d")
        
        if not stock_data.empty:
            # Display stock metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = stock_data['Close'].iloc[-1]
            price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
            price_change_pct = (price_change / stock_data['Close'].iloc[0]) * 100
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${current_price:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Change</div>
                        <div class="metric-value">{price_change_pct:+.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">High</div>
                        <div class="metric-value">${stock_data['High'].max():.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Low</div>
                        <div class="metric-value">${stock_data['Low'].min():.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Chart
            st.plotly_chart(plot_stock_chart(stock_data, indicators), use_container_width=True)
            
            # AI Analysis
            analysis = perform_crew_analysis(stock_symbol)
            
            # FIX: Store stock data in session state
            st.session_state['last_stock_data'] = stock_data
            
            if analysis:
                st.markdown("""
                    <div class="analysis-card">
                        <h3 style="margin-top: 0;">✅ Analysis Complete!</h3>
                        <p>Our AI agents have successfully analyzed {stock_symbol} across multiple dimensions.</p>
                    </div>
                """.format(stock_symbol=stock_symbol), unsafe_allow_html=True)
        else:
            st.error("❌ Unable to fetch stock data. Please check the symbol and try again.")
 
