"""
CLI tool for processing insurance quote requests.
Usage: python -m cli.process_quote <email_file_or_text>
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import UnderwritingPipeline
from src.config import get_settings


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a header with formatting."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")


def print_step(step_num: int, name: str, status: str):
    """Print step progress."""
    if status == "running":
        icon = "..."
        color = Colors.YELLOW
    elif status == "complete":
        icon = "done"
        color = Colors.GREEN
    else:
        icon = "!"
        color = Colors.RED

    print(f"  [{step_num:2d}/10] {name:<25} {color}{icon}{Colors.ENDC}")


def print_result(result):
    """Print the pipeline result in a formatted way."""
    if not result.success:
        print(f"\n{Colors.RED}Quote generation failed!{Colors.ENDC}")
        for error in result.errors:
            print(f"  Error: {error}")
        return

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}        QUOTE GENERATED SUCCESSFULLY{Colors.ENDC}")
    print("=" * 70)

    # Client info
    if result.extracted_email:
        print(f"\n{Colors.BOLD}Client:{Colors.ENDC} {result.extracted_email.client_name}")
        if result.industry_classification:
            print(f"{Colors.BOLD}Industry:{Colors.ENDC} {result.industry_classification.industry_name} "
                  f"(BIC: {result.industry_classification.bic_code})")

    # Premium summary
    if result.modifier_result:
        print(f"\n{Colors.BOLD}Annual Premium:{Colors.ENDC} ${result.modifier_result.adjusted_premium:,.2f}")

        if result.premium_calculation:
            print(f"\n{Colors.BOLD}Premium Breakdown:{Colors.ENDC}")
            for item in result.premium_calculation.line_items:
                print(f"  - {item.coverage_type.replace('_', ' ').title()}: ${item.base_premium:,.2f}")

            if result.modifier_result.modifiers_applied:
                print(f"\n{Colors.BOLD}Modifiers Applied:{Colors.ENDC}")
                for mod in result.modifier_result.modifiers_applied:
                    sign = "+" if mod.premium_impact > 0 else ""
                    print(f"  - {mod.modifier_name}: {sign}${mod.premium_impact:,.2f} ({mod.modifier_value:+.0%})")

    # Risk assessment
    if result.risk_assessment:
        print(f"\n{Colors.BOLD}Risk Assessment:{Colors.ENDC}")
        print(f"  Level: {result.risk_assessment.overall_risk_level}")
        print(f"  Score: {result.risk_assessment.risk_score:.0f}/100")
        print(f"  Recommendation: {result.risk_assessment.recommendation}")

    # Authority
    if result.authority_check:
        print(f"\n{Colors.BOLD}Authority:{Colors.ENDC}")
        print(f"  Level: {result.authority_check.authority_level}")
        if result.authority_check.requires_approval:
            print(f"  {Colors.YELLOW}Requires approval: {result.authority_check.approval_reason}{Colors.ENDC}")
        else:
            print(f"  {Colors.GREEN}Auto-bind eligible{Colors.ENDC}")

    # Warnings
    if result.warnings:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Warnings:{Colors.ENDC}")
        for warning in result.warnings:
            print(f"  - {warning}")

    # Quote ID and file locations
    print(f"\n{Colors.BOLD}Quote ID:{Colors.ENDC} {result.quote_id}")

    # Save outputs
    if result.generated_quote:
        settings = get_settings()
        quotes_dir = settings.quotes_dir

        # Save JSON
        json_path = quotes_dir / f"{result.quote_id}.json"
        with open(json_path, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        print(f"\n{Colors.CYAN}Quote data saved to:{Colors.ENDC} {json_path}")

        # Save quote letter
        letter_path = quotes_dir / f"{result.quote_id}_letter.txt"
        with open(letter_path, 'w') as f:
            f.write(result.generated_quote.quote_letter)
        print(f"{Colors.CYAN}Quote letter saved to:{Colors.ENDC} {letter_path}")

    # Processing time
    if result.metrics:
        print(f"\n{Colors.BOLD}Processing time:{Colors.ENDC} {result.metrics.total_duration_seconds:.1f} seconds")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process insurance quote requests from broker emails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.process_quote data/sample_emails/construction_company.txt
  python -m cli.process_quote --text "Hi, I need a quote for ABC Corp..."
  python -m cli.process_quote data/sample_emails/restaurant.txt --json
        """
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Path to email file (.txt or .eml)"
    )
    parser.add_argument(
        "--text", "-t",
        help="Email content as text (alternative to file)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON only"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Get email content
    email_content = None

    if args.text:
        email_content = args.text
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"{Colors.RED}Error: File not found: {args.file}{Colors.ENDC}")
            sys.exit(1)
        email_content = file_path.read_text()
    else:
        # Try reading from stdin
        if not sys.stdin.isatty():
            email_content = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    if not email_content or len(email_content.strip()) < 50:
        print(f"{Colors.RED}Error: Email content too short or empty{Colors.ENDC}")
        sys.exit(1)

    # Setup progress callback
    def progress_callback(step: int, name: str, status: str):
        if not args.quiet and not args.json:
            print_step(step, name, status)

    # Print header
    if not args.quiet and not args.json:
        print(f"\n{Colors.BOLD}Insurance Quote Processing{Colors.ENDC}")
        print("-" * 40)
        print(f"Processing quote request...")

    # Process quote
    try:
        pipeline = UnderwritingPipeline(progress_callback=progress_callback)
        result = pipeline.process(email_content)

        if args.json:
            # JSON output only
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            # Formatted output
            print_result(result)

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
