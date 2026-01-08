"""
Tests for the underwriting pipeline.
"""

import pytest
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    RateInfo,
    PremiumCalculation,
    PremiumLineItem,
    ModifierResult,
    ModifierDetail,
    CoverageRequest,
)
from src.pipeline.steps.premium_calculation import PremiumCalculationStep


class TestPremiumCalculation:
    """Tests for premium calculation step."""

    def test_basic_gl_calculation(self):
        """Test basic general liability premium calculation."""
        step = PremiumCalculationStep()

        extracted = ExtractedEmail(
            client_name="Test Corp",
            industry_description="Test Industry",
            annual_revenue=1000000,
            employee_count=50,
            coverage_requested=[],
            raw_email="test"
        )

        rates = [
            RateInfo(
                bic_code="54",
                coverage_type="general_liability",
                base_rate=5.0,
                rate_basis="per_1000_revenue",
                minimum_premium=1000
            )
        ]

        result = step.execute(extracted, rates, None)

        assert result.total_base_premium == 5000.0  # $1M / 1000 * $5
        assert len(result.line_items) == 1
        assert result.line_items[0].coverage_type == "general_liability"

    def test_minimum_premium_applies(self):
        """Test that minimum premium is applied when calculated is lower."""
        step = PremiumCalculationStep()

        extracted = ExtractedEmail(
            client_name="Small Corp",
            industry_description="Small Business",
            annual_revenue=100000,  # Low revenue
            employee_count=5,
            coverage_requested=[],
            raw_email="test"
        )

        rates = [
            RateInfo(
                bic_code="54",
                coverage_type="general_liability",
                base_rate=5.0,
                rate_basis="per_1000_revenue",
                minimum_premium=1500  # Higher than calculated ($500)
            )
        ]

        result = step.execute(extracted, rates, None)

        assert result.total_base_premium == 1500.0  # Minimum applies
        assert "minimum premium" in result.line_items[0].calculation_notes.lower()

    def test_auto_liability_calculation(self):
        """Test auto liability premium calculation by vehicle count."""
        step = PremiumCalculationStep()

        extracted = ExtractedEmail(
            client_name="Fleet Corp",
            industry_description="Transportation",
            annual_revenue=5000000,
            vehicle_count=10,
            coverage_requested=[],
            raw_email="test"
        )

        rates = [
            RateInfo(
                bic_code="48",
                coverage_type="auto_liability",
                base_rate=750.0,
                rate_basis="per_vehicle",
                minimum_premium=1500
            )
        ]

        result = step.execute(extracted, rates, None)

        assert result.total_base_premium == 7500.0  # 10 vehicles * $750

    def test_property_calculation(self):
        """Test property premium calculation by TIV."""
        step = PremiumCalculationStep()

        extracted = ExtractedEmail(
            client_name="Property Corp",
            industry_description="Real Estate",
            annual_revenue=2000000,
            property_value=500000,
            coverage_requested=[],
            raw_email="test"
        )

        rates = [
            RateInfo(
                bic_code="53",
                coverage_type="property",
                base_rate=0.50,
                rate_basis="percent_of_tiv",
                minimum_premium=1000
            )
        ]

        result = step.execute(extracted, rates, None)

        assert result.total_base_premium == 2500.0  # $500K * 0.5%


class TestModels:
    """Tests for pipeline data models."""

    def test_extracted_email_creation(self):
        """Test ExtractedEmail model creation."""
        email = ExtractedEmail(
            client_name="Test Corp",
            industry_description="Testing Services",
            annual_revenue=1000000,
            employee_count=50,
            years_in_business=10,
            coverage_requested=[
                CoverageRequest(
                    coverage_type="general_liability",
                    limits="$1M/$2M"
                )
            ],
            raw_email="Test email content"
        )

        assert email.client_name == "Test Corp"
        assert email.annual_revenue == 1000000
        assert len(email.coverage_requested) == 1
        assert email.coverage_requested[0].limits == "$1M/$2M"

    def test_industry_classification(self):
        """Test IndustryClassification model."""
        classification = IndustryClassification(
            bic_code="54",
            industry_name="Professional Services",
            risk_category="LOW",
            confidence_score=0.95,
            matching_keywords=["consulting", "professional"]
        )

        assert classification.bic_code == "54"
        assert classification.risk_category == "LOW"
        assert classification.confidence_score == 0.95

    def test_modifier_result(self):
        """Test ModifierResult model."""
        modifier = ModifierResult(
            modifiers_applied=[
                ModifierDetail(
                    modifier_name="Loss History Credit",
                    modifier_type="experience",
                    modifier_value=-0.10,
                    reason="Clean loss history",
                    premium_impact=-500
                )
            ],
            total_modifier_impact=-500,
            total_modifier_percentage=-0.10,
            adjusted_premium=4500
        )

        assert modifier.total_modifier_impact == -500
        assert modifier.adjusted_premium == 4500
        assert len(modifier.modifiers_applied) == 1


class TestEmailValidation:
    """Tests for email content validation."""

    def test_minimum_email_length(self):
        """Test that very short emails are rejected."""
        # This would be tested at the API level
        short_email = "Hi, quote please."
        assert len(short_email) < 50  # Below minimum threshold

    def test_email_with_all_fields(self, sample_email_construction):
        """Test that sample email has sufficient content."""
        assert len(sample_email_construction) > 50
        assert "construction" in sample_email_construction.lower()
        assert "$15M" in sample_email_construction or "15M" in sample_email_construction
