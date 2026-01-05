
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from crew_simple import run_analysis
import json
# ============================================================================
# AI CHAT ASSISTANT FEATURE - NEW IMPORT
# Added: Import for Vertex AI chat functionality
# ============================================================================
from langchain_google_vertexai import ChatVertexAI
import os
from dotenv import load_dotenv
# ============================================================================
# PDF REPORT GENERATION - NEW IMPORTS
# Added: Imports for PDF generation functionality
# ============================================================================
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import io
import base64

load_dotenv()

# Vertex AI Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =r"C:\Users\pnanda\Downloads\my-project-28112025-479604-7513c3845ed2.json"  # <-- CHANGE THIS
VERTEX_PROJECT_ID = "my-project-28112025-479604"
VERTEX_LOCATION = "us-central1"
VERTEX_MODEL_NAME = "gemini-2.5-flash"

# Page Configuration
st.set_page_config(
    layout="wide", 
    page_title="Financial Advisor",
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
def add_custom_css():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            
            .stApp {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
                font-family: 'Roboto', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 16px;
                line-height: 1.6;
            }}
            
            * {{
                font-family: 'Roboto', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            }}
            
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            }}
            
            [data-testid="stSidebar"] .stMarkdown {{
                color: white;
                font-size: 16px;
                line-height: 1.6;
            }}
            
            [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
                font-weight: 600 !important;
                letter-spacing: -0.025em;
                color: white !important;
            }}
            
            .main-header {{
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
                padding: 2.5rem;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
            }}
            
            .main-title {{
                font-size: 3.5rem;
                font-weight: 700;
                color: #ffffff;
                margin: 0;
                letter-spacing: -0.05em;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }}
            
            .main-subtitle {{
                font-size: 1.25rem;
                color: #e2e8f0;
                margin-top: 0.75rem;
                font-weight: 400;
                letter-spacing: 0.025em;
            }}
            
            .info-card {{
                background: #ffffff;
                border-radius: 12px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                border: 1px solid #e2e8f0;
            }}
            
            .info-card h3 {{
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                margin-bottom: 1rem !important;
                letter-spacing: -0.025em;
            }}
            
            .info-card p {{
                font-size: 1.1rem !important;
                line-height: 1.7 !important;
                color: #374151 !important;
                margin-bottom: 0.75rem !important;
            }}
            
            .analysis-card {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                border-radius: 12px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
                color: white;
            }}
            
            .analysis-card h3 {{
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                letter-spacing: -0.025em;
            }}
            
            .analysis-card p {{
                font-size: 1.1rem !important;
                line-height: 1.7 !important;
            }}
            
            .report-card {{
                background: white;
                border-radius: 12px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                border-left: 5px solid #6366f1;
            }}
            
            .stButton>button {{
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.875rem 1.75rem;
                font-size: 1.05rem !important;
                font-weight: 500 !important;
                transition: all 0.3s ease;
                box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
                letter-spacing: 0.025em;
            }}
            
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            }}
            
            .stTextInput>div>div>input {{
                border-radius: 8px;
                border: 2px solid #d1d5db;
                padding: 0.875rem;
                font-size: 1rem !important;
                font-weight: 400;
            }}
            
            .stSelectbox>div>div>select {{
                border-radius: 8px;
                border: 2px solid #d1d5db;
                font-size: 1rem !important;
                font-weight: 400;
            }}
            
            .stNumberInput>div>div>input {{
                border-radius: 8px;
                border: 2px solid #d1d5db;
                padding: 0.875rem;
                font-size: 1rem !important;
                font-weight: 400;
            }}
            
            .metric-card {{
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                color: white;
                box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
                transition: transform 0.2s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-2px);
            }}
            
            .metric-value {{
                font-size: 2.25rem !important;
                font-weight: 700 !important;
                margin: 0.75rem 0 !important;
                letter-spacing: -0.025em;
            }}
            
            .metric-label {{
                font-size: 0.95rem !important;
                opacity: 0.9;
                font-weight: 500 !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            .stSuccess {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                border-radius: 8px;
                padding: 1.25rem;
                font-size: 1.05rem !important;
                font-weight: 500;
            }}
            
            .stError {{
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                border-radius: 8px;
                padding: 1.25rem;
                font-size: 1.05rem !important;
                font-weight: 500;
            }}
            
            .section-header {{
                font-size: 2.25rem !important;
                font-weight: 700 !important;
                color: white;
                margin: 2.5rem 0 1.5rem 0 !important;
                padding-bottom: 0.75rem;
                border-bottom: 3px solid white;
                letter-spacing: -0.05em;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }}
            
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
                font-weight: 600 !important;
                letter-spacing: -0.025em !important;
            }}
            
            .stMarkdown p {{
                font-size: 1.05rem !important;
                line-height: 1.7 !important;
                color: #374151;
            }}
            
            .stRadio > div {{
                font-size: 1.05rem !important;
            }}
            
            .stRadio label {{
                font-weight: 500 !important;
                color: white !important;
            }}
            
            .stMultiSelect label {{
                font-size: 1.05rem !important;
                font-weight: 500 !important;
                color: white !important;
            }}
            
            .stSpinner > div {{
                font-size: 1.1rem !important;
                font-weight: 500 !important;
            }}
            
            .chat-message {{
                font-size: 1.05rem !important;
                line-height: 1.7 !important;
            }}
            
            @media (max-width: 768px) {{
                .main-title {{
                    font-size: 2.5rem !important;
                }}
                
                .main-subtitle {{
                    font-size: 1.1rem !important;
                }}
                
                .section-header {{
                    font-size: 1.75rem !important;
                }}
                
                .metric-value {{
                    font-size: 1.75rem !important;
                }}
            }}
        </style>
    """, unsafe_allow_html=True)
# Main Function
def main():
    add_custom_css()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">WealthRa</h1>
            <p class="main-subtitle">AI-Powered Financial Advisory Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.markdown("<h2 style='color: white; text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    
    options = st.sidebar.radio(
        "Navigation Menu",
        ["Stock Analysis", "AI Chat Assistant", "Budget Planning"],
        label_visibility="collapsed"
    )

    if options == "Stock Analysis":
        stock_analysis_section()
    elif options == "AI Chat Assistant":
        ai_chat_section()
    elif options == "Budget Planning":
        budgeting_section()

def stock_analysis_section():
    st.markdown('<h2 class="section-header">Stock Analysis</h2>', unsafe_allow_html=True)
    
    # Sidebar Inputs
    st.sidebar.markdown("<h3 style='color: white;'>Analysis Settings</h3>", unsafe_allow_html=True)
    stock_symbol = st.sidebar.text_input("Stock Symbol", help="Enter stock ticker (e.g., AAPL, GOOGL, MSFT)")
    time_period = st.sidebar.selectbox("Time Period", ['3mo', '6mo', '1y', '2y', '5y'], index=2)
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
        
        # ============================================================================
        # PDF DOWNLOAD BUTTON
        # Purpose: Allow users to download the analysis report as PDF
        # ============================================================================
        st.markdown("""
            <div class="report-card">
                <h3 style="margin-top: 0; color: #6366f1;">📄 Download Report</h3>
                <p style="margin-bottom: 1.5rem;">Get a professional PDF copy of your analysis report for offline viewing or sharing.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate PDF and create download button
        try:
            pdf_buffer = generate_pdf_report(
                st.session_state.get('last_stock', 'UNKNOWN'),
                st.session_state['last_report'],
                stock_data
            )
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=f"{st.session_state.get('last_stock', 'stock')}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
        
        st.markdown("""
            <div class="analysis-card">
                <h3 style="margin-top: 0;">✅ Analysis Complete!</h3>
                <p>Our AI agents have successfully analyzed {stock_symbol} across multiple dimensions.</p>
            </div>
        """.format(stock_symbol=st.session_state.get('last_stock', stock_symbol)), unsafe_allow_html=True)
        
        # Show chat assistant notification
        if 'last_report' in st.session_state:
            st.markdown("""
                <div class="info-card">
                    <p style="color: #6366f1; font-weight: 600; margin: 0; text-align: center;">
                        💬 Use the AI Chat Assistant in the navigation to ask questions about your analysis!
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
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
                # ============================================================================
                # PDF DOWNLOAD BUTTON FOR NEW ANALYSIS
                # Purpose: Allow users to download the fresh analysis report as PDF
                # ============================================================================
                st.markdown("""
                    <div class="report-card">
                        <h3 style="margin-top: 0; color: #6366f1;">📄 Download Report</h3>
                        <p style="margin-bottom: 1.5rem;">Get a professional PDF copy of your analysis report for offline viewing or sharing.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Generate PDF and create download button
                try:
                    pdf_buffer = generate_pdf_report(
                        stock_symbol,
                        st.session_state['last_report'],
                        stock_data
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"{stock_symbol}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                
                st.markdown("""
                    <div class="analysis-card">
                        <h3 style="margin-top: 0;">✅ Analysis Complete!</h3>
                        <p>Our AI agents have successfully analyzed {stock_symbol} across multiple dimensions.</p>
                    </div>
                """.format(stock_symbol=stock_symbol), unsafe_allow_html=True)
        else:
            st.error("❌ Unable to fetch stock data. Please check the symbol and try again.")
        
# Fetch Stock Data
def get_stock_data(stock_symbol, period='1y'):
    try:
        return yf.download(stock_symbol, period=period)
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Plot Stock Chart
def plot_stock_chart(stock_data, indicators):
    if stock_data.empty or stock_data.isnull().any().any():
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#11998e',
            decreasing_line_color='#eb3349'
        ),
        row=1, col=1
    )

    # Volume chart
    if 'Volume' in indicators:
        colors = ['#11998e' if stock_data['Close'].iloc[i] >= stock_data['Open'].iloc[i] 
                  else '#eb3349' for i in range(len(stock_data))]
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker=dict(color=colors)
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=700,
        title=dict(
            text="Stock Price Chart",
            font=dict(size=24, color='white')
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        yaxis=dict(
            title="Price ($)",
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis2=dict(
            title="Volume",
            gridcolor='rgba(255,255,255,0.1)'
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)'
        ),
        xaxis2=dict(
            gridcolor='rgba(255,255,255,0.1)'
        ),
        hovermode='x unified'
    )

    return fig


def perform_crew_analysis(stock_symbol):
    with st.spinner("🤖 AI Agents analyzing... This may take a moment..."):
        try:
            analysis_result = run_analysis(stock_symbol)
            st.write(analysis_result['report'])
            
            # ============================================================================
            # AI CHAT ASSISTANT FEATURE - STORE REPORT IN SESSION STATE
            # Purpose: Save report and stock symbol for chat context
            # ============================================================================
            st.session_state['last_report'] = analysis_result['report']
            st.session_state['last_stock'] = stock_symbol
            
            return analysis_result

        except Exception as e:
            st.error(f"⚠️ Analysis failed: {str(e)}")
            return None


# ============================================================================
# PDF REPORT GENERATION FUNCTION
# Purpose: Generate downloadable PDF reports for stock analysis
# ============================================================================
def generate_pdf_report(stock_symbol, report_content, stock_data):
    """Generate a PDF report for stock analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#6366f1'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#374151')
    )
    
    # Title
    story.append(Paragraph(f"Stock Analysis Report: {stock_symbol.upper()}", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Paragraph("Generated by: WealthRa AI Financial Advisor", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Stock metrics table
    if stock_data is not None and not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
        price_change_pct = (price_change / stock_data['Close'].iloc[0]) * 100
        high_price = stock_data['High'].max()
        low_price = stock_data['Low'].min()
        
        story.append(Paragraph("Key Metrics", heading_style))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Current Price', f'${current_price:.2f}'],
            ['Price Change', f'{price_change_pct:+.2f}%'],
            ['Period High', f'${high_price:.2f}'],
            ['Period Low', f'${low_price:.2f}']
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 30))
    
    # Analysis report
    story.append(Paragraph("AI Analysis Report", heading_style))
    story.append(Spacer(1, 12))
    
    # Split report content into paragraphs
    report_paragraphs = report_content.split('\n\n')
    for paragraph in report_paragraphs:
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 50))
    story.append(Paragraph("Disclaimer: This report is for informational purposes only and should not be considered as financial advice. Please consult with a qualified financial advisor before making investment decisions.", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def budgeting_section():
    st.markdown('<h2 class="section-header">💳 Smart Budgeting Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; margin-top: 0;">📊 Track Your Finances</h3>
            <p style="font-size: 1.1rem; color: #333;">
                Calculate your monthly savings and get personalized financial advice.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("💵 Monthly Income ($)", min_value=0.0, step=100.0, value=5000.0)
    
    with col2:
        expenses = st.number_input("💸 Monthly Expenses ($)", min_value=0.0, step=100.0, value=3000.0)

    if st.button("📊 Calculate Savings", use_container_width=True):
        savings = income - expenses
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        if savings < 0:
            st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid #eb3349;">
                    <h3 style="color: #eb3349;">⚠️ Budget Deficit</h3>
                    <p style="font-size: 1.5rem; color: #eb3349; font-weight: bold;">${-savings:.2f}</p>
                    <p style="color: #666;">Your expenses exceed your income. Consider reducing expenses or increasing income.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid #11998e;">
                    <h3 style="color: #11998e;">✅ Monthly Savings</h3>
                    <p style="font-size: 2rem; color: #11998e; font-weight: bold;">${savings:.2f}</p>
                    <p style="font-size: 1.2rem; color: #667eea;">Savings Rate: {savings_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            if savings_rate < 10:
                advice = "⚠️ Try to save at least 10-20% of your income for financial security."
                color = "#f45c43"
            elif savings_rate < 20:
                advice = "👍 Good start! Aim for 20% or more for optimal financial health."
                color = "#f39c12"
            else:
                advice = "🎉 Excellent! You're on track for strong financial health."
                color = "#11998e"
            
            st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid {color};">
                    <p style="font-size: 1.1rem; color: #333;">{advice}</p>
                </div>
            """, unsafe_allow_html=True)


def ai_chat_section():
    st.markdown('<h2 class="section-header">💬 AI Chat Assistant</h2>', unsafe_allow_html=True)
    
    if 'last_report' not in st.session_state:
        st.markdown("""
            <div class="info-card">
                <h3 style="color: #ef4444; margin-top: 0;">⚠️ No Analysis Available</h3>
                <p style="font-size: 1.1rem; color: #333;">
                    Please run a stock analysis first to enable the AI Chat Assistant.
                    Go to "Stock Analysis" and analyze a stock to get started!
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
        <div class="info-card">
            <h3 style="color: #6366f1; margin-top: 0;">🤖 Chat about {st.session_state.get('last_stock', 'Your Analysis')}</h3>
            <p style="font-size: 1.1rem; color: #333;">
                Ask me anything about the analysis report! I can explain recommendations, risks, market conditions, or any other details.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
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
    
    # Custom question input
    user_question = st.text_input(
        "Ask your question:",
        placeholder="e.g., What factors influenced the recommendation?",
        key="main_chat_input"
    )
    
    if st.button("📤 Send Question", use_container_width=True):
        if user_question:
            ask_chat_question(user_question)
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💬 Chat History:**")
        
        for i, chat in enumerate(st.session_state.chat_history):
            # User question
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                            padding: 1.2rem; 
                            border-radius: 15px; 
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);">
                    <strong style="color: white; font-size: 1.1rem;">👤 You:</strong><br>
                    <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1rem; line-height: 1.6;">{chat['question']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # AI response
            st.markdown(f"""
                <div style="background: white; 
                            padding: 1.5rem; 
                            border-radius: 15px; 
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                            border-left: 5px solid #6366f1;">
                    <strong style="color: #6366f1; font-size: 1.1rem;">🤖 AI Assistant:</strong><br>
                    <div style="color: #333; margin: 0.8rem 0 0 0; font-size: 1rem; line-height: 1.8;">{chat['answer']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# AI CHAT ASSISTANT FEATURE - QUESTION HANDLER
# Purpose: Process user questions and generate AI responses using Vertex AI
# ============================================================================
def ask_chat_question(question):
    """Process a chat question and get AI response"""
    if 'last_report' not in st.session_state:
        st.error("No analysis report available. Please run an analysis first.")
        return
    
    try:
        # Initialize Vertex AI Gemini
        llm = ChatVertexAI(
            model_name=VERTEX_MODEL_NAME,
            project=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION,
            temperature=0.7,
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


if __name__ == "__main__":
    main()
