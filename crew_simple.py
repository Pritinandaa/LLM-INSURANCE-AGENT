
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\pnanda\Downloads\my-project-28112025-479604-418004fc57ae.json"  # <-- CHANGE THIS

# 2) Your GCP project and region
VERTEX_PROJECT_ID = "my-project-28112025-479604"     # <-- CHANGE THIS
VERTEX_LOCATION   = "us-central1"     # e.g., "us-central1"  <-- CHANGE THIS

VERTEX_MODEL_NAME = "gemini-2.5-flash"

def run_analysis(stock_symbol):
    """Run financial analysis using Vertex AI Gemini"""

    # LangChain chat model for Vertex AI (drop-in compatible with .invoke)
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL_NAME,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
        temperature=0.7,
    )

    # Initialize tools (unchanged)
    search_tool = SearchInternetTool()
    news_tool = SearchNewsTool()
    yf_tech_tool = YFinanceTechnicalAnalysisTool()
    yf_fundamental_tool = YFinanceFundamentalAnalysisTool()
    # reddit_tool = RedditSentimentAnalysisTool()  # Temporarily disabled

    print(f"Running analysis for stock: {stock_symbol}")

    # ---------------- Research Analysis ----------------
    print("\n=== Research Phase ===")
    research_prompt = f"""Conduct research on {stock_symbol}. Search for recent news, analyst ratings, and market sentiment.
Provide a comprehensive summary of your findings."""
    research_result = llm.invoke(research_prompt)

    # ---------------- Technical Analysis ----------------
    print("\n=== Technical Analysis Phase ===")
    tech_data = yf_tech_tool._run(stock_symbol)
    tech_prompt = f"""Based on this technical data for {stock_symbol}:
{tech_data}
Provide a technical analysis summary with buy/sell/hold recommendation."""
    tech_result = llm.invoke(tech_prompt)

    # ---------------- Fundamental Analysis ----------------
    print("\n=== Fundamental Analysis Phase ===")
    fundamental_data = yf_fundamental_tool._run(stock_symbol)
    fundamental_prompt = f"""Based on this fundamental data for {stock_symbol}:
{fundamental_data}
Provide a fundamental analysis summary with valuation assessment."""
    fundamental_result = llm.invoke(fundamental_prompt)

    # ---------------- Sentiment Analysis ----------------
    print("\n=== Sentiment Analysis Phase ===")
    sentiment_summary = "Reddit sentiment analysis temporarily disabled"

    # ---------------- Final Report ----------------
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
"""
    final_report = llm.invoke(final_prompt)

    return {
        'report': final_report.content
    }

if __name__ == "__main__":
    result = run_analysis('AAPL')
    print("\n=== FINAL REPORT ===")
    print(result['report'])
