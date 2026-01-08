"""
Test script to verify the full pipeline is working.
Usage: python scripts/test_full_pipeline.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.pipeline.orchestrator import UnderwritingPipeline


def test_pipeline():
    """Run a test quote through the pipeline."""
    print("=" * 60)
    print("  Insurance Underwriting Pipeline Test")
    print("=" * 60)

    # Sample email for testing
    test_email = """
Subject: Quote Request - ABC Construction Corp

Hi there,

I need a quote for ABC Construction Corp. They're a mid-size commercial
construction company based in Texas, doing about $15M in annual revenue.
They need General Liability coverage with $2M/$4M limits, plus Auto
Liability for their fleet of 25 vehicles.

The company has been in business for 12 years, 85 employees. They've had
two small workers comp claims in the past 3 years but nothing major.
They're looking for coverage to start March 1st.

Thanks,
Sarah Johnson
ABC Insurance Brokerage
"""

    print("\nTest Email:")
    print("-" * 40)
    print(test_email[:200] + "...")
    print("-" * 40)

    # Progress callback
    def progress(step: int, name: str, status: str):
        icon = "..." if status == "running" else "done" if status == "complete" else "!"
        print(f"  [{step:2d}/10] {name:<25} [{icon}]")

    print("\nProcessing...")
    print("-" * 40)

    try:
        pipeline = UnderwritingPipeline(progress_callback=progress)
        result = pipeline.process(test_email)

        print("-" * 40)

        if result.success:
            print("\nSUCCESS!")
            print(f"\nQuote ID: {result.quote_id}")
            print(f"Client: {result.extracted_email.client_name}")
            print(f"Industry: {result.industry_classification.industry_name}")
            print(f"Total Premium: ${result.modifier_result.adjusted_premium:,.2f}")
            print(f"Risk Level: {result.risk_assessment.overall_risk_level}")
            print(f"Processing Time: {result.metrics.total_duration_seconds:.1f}s")

            if result.warnings:
                print(f"\nWarnings:")
                for w in result.warnings:
                    print(f"  - {w}")

            print("\nPipeline test passed!")
            return True
        else:
            print("\nFAILED!")
            print(f"Errors: {result.errors}")
            return False

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
