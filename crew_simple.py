
from langchain_google_vertexai import ChatVertexAI
from tools.search_tool import SearchInternetTool, SearchNewsTool
from tools.yf_tech_analysis import YFinanceTechnicalAnalysisTool
from tools.yf_fundamental_analysis import YFinanceFundamentalAnalysisTool
# from tools.sentiment_analysis import RedditSentimentAnalysisTool  # Temporarily disabled
from dotenv import load_dotenv
from pathlib import Path
import os

os.environ.setdefault("CREWAI_TELEMETRY", "false")
load_dotenv()

# --- Vertex AI config (update to your creds) ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\pnanda\Downloads\my-project-28112025-479604-7513c3845ed2.json"  # <-- CHANGE THIS
VERTEX_PROJECT_ID = "my-project-28112025-479604"  # <-- CHANGE THIS
VERTEX_LOCATION = "us-central1"                   # e.g., "us-central1" <-- CHANGE THIS
VERTEX_MODEL_NAME = "gemini-2.5-flash"

# ---- Simple logger helper so you see exactly which source was used ----
def log_source(tool_obj, default_name: str):
    """
    Print the data source/provider name for the given tool.
    If the tool exposes 'provider' or 'source_name', print that; else print default_name.
    """
    provider = getattr(tool_obj, "provider", None) or getattr(tool_obj, "source_name", None) or default_name
    print(f"[SOURCE] Using {provider}")

def run_analysis(stock_symbol: str):
    """Run financial analysis using Vertex AI Gemini with visible source logging."""

    # LangChain chat model for Vertex AI
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL_NAME,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
        temperature=0.7,
    )

    # Initialize tools
    search_tool = SearchInternetTool()          # typically backed by Serper/Bing
    news_tool = SearchNewsTool()                # typically backed by Serper/News API
    yf_tech_tool = YFinanceTechnicalAnalysisTool()         # Yahoo Finance (yfinance)
    yf_fundamental_tool = YFinanceFundamentalAnalysisTool()# Yahoo Finance (yfinance)
    # reddit_tool = RedditSentimentAnalysisTool()          # Temporarily disabled

    print(f"Running analysis for stock: {stock_symbol}")

    # ---------------------- Research Analysis ----------------------
    print("\n=== Research Phase ===")
    log_source(search_tool, "Serper (Web Search)")
    try:
        # Expecting the tool to return text or a structured dict; adjust as needed
        web_research = search_tool._run(stock_symbol)
    except Exception as e:
        print(f"[ERROR] SearchInternetTool failed: {e}")
        web_research = "Web search unavailable."

    log_source(news_tool, "Serper/News API (Top News)")
    try:
        recent_news = news_tool._run(stock_symbol)
    except Exception as e:
        print(f"[ERROR] SearchNewsTool failed: {e}")
        recent_news = "News search unavailable."

    research_prompt = f"""You are a financial research assistant.
Synthesize the following inputs for {stock_symbol}:

Web Research:
{web_research}

Recent News:
{recent_news}

Provide a concise but comprehensive research summary with sources mentioned where available."""
    research_result = llm.invoke(research_prompt)

    # ---------------------- Technical Analysis ----------------------
    print("\n=== Technical Analysis Phase ===")
    log_source(yf_tech_tool, "Yahoo Finance (yfinance)")
    try:
        tech_data = yf_tech_tool._run(stock_symbol)
    except Exception as e:
        print(f"[ERROR] YFinanceTechnicalAnalysisTool failed: {e}")
        tech_data = "Technical data unavailable."

    tech_prompt = f"""Based on this technical data for {stock_symbol}:
{tech_data}

Provide a technical analysis summary with buy/sell/hold recommendation."""
    tech_result = llm.invoke(tech_prompt)

    # ---------------------- Fundamental Analysis ----------------------
    print("\n=== Fundamental Analysis Phase ===")
    log_source(yf_fundamental_tool, "Yahoo Finance (yfinance)")
    try:
        fundamental_data = yf_fundamental_tool._run(stock_symbol)
    except Exception as e:
        print(f"[ERROR] YFinanceFundamentalAnalysisTool failed: {e}")
        fundamental_data = "Fundamental data unavailable."

    fundamental_prompt = f"""Based on this fundamental data for {stock_symbol}:
{fundamental_data}

Provide a fundamental analysis summary with valuation assessment."""
    fundamental_result = llm.invoke(fundamental_prompt)

    # ---------------------- Sentiment Analysis ----------------------
    print("\n=== Sentiment Analysis Phase ===")
    print("[SOURCE] Reddit (Pushshift/API/Serper) — temporarily disabled")
    sentiment_summary = "Reddit sentiment analysis temporarily disabled"

    # ---------------------- Final Report ----------------------
    print("\n=== Generating Final Report ===")
    final_prompt = f"""Create a comprehensive investment report for {stock_symbol}.

Research Findings:
{research_result.content}

Technical Analysis:
{tech_result.content}

Fundamental Analysis:
{fundamental_result.content}

Sentiment Analysis:
{sentiment_summary}

Provide a final investment recommendation with:
1. Executive Summary
2. Key Findings
3. Investment Recommendation (Buy/Hold/Sell)
4. Price Target (12-month)
5. Key Risks
6. Sources used (list the sources you relied on)."""
    final_report = llm.invoke(final_prompt)

    return {
        'report': final_report.content
    }

if __name__ == "__main__":
    result = run_analysis('AAPL')
    print("\n=== FINAL REPORT ===")
    print(result['report'])
