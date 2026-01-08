import asyncio
import os
from dotenv import load_dotenv
from src.agents.underwriting_agents import UnderwritingAgentOrchestrator

load_dotenv()

async def test_agents():
    print("Testing Underwriting Agents...")
    email_content = """
    Subject: Quote Request - Blue Sky Construction
    Hi, I need a quote for Blue Sky Construction. They are a residential builder in us-central1 with $5M revenue and 10 employees. 
    They have had no losses in 5 years. They need $1M General Liability.
    """
    
    orchestrator = UnderwritingAgentOrchestrator()
    results = await orchestrator.run_workflow(email_content)
    
    for res in results:
        print(f"\n--- {res['agent']} ---")
        print(res['content'])

if __name__ == "__main__":
    asyncio.run(test_agents())
