"""
Step 5: Premium Calculation
Calculates base premiums for each coverage line.
"""

from datetime import datetime
from typing import List, Optional
from src.pipeline.models import (
    ExtractedEmail,
    RateInfo,
    RevenueEstimate,
    PremiumCalculation,
    PremiumLineItem,
)


class PremiumCalculationStep:
    """
    Calculates base premiums by applying rates to exposure values.
    Handles different rate bases (revenue, property value, vehicle count, etc.)
    """

    def execute(
        self,
        extracted_email: ExtractedEmail,
        rates: List[RateInfo],
        revenue_estimate: Optional[RevenueEstimate] = None
    ) -> PremiumCalculation:
        """
        Calculate base premium for each coverage.

        Args:
            extracted_email: Parsed email data
            rates: List of applicable rates
            revenue_estimate: Revenue estimate if original wasn't provided

        Returns:
            PremiumCalculation with all line items
        """
        # Determine revenue to use
        revenue = extracted_email.annual_revenue
        if revenue is None and revenue_estimate:
            revenue = revenue_estimate.estimated_revenue
        if revenue is None:
            revenue = 1000000  # Default fallback

        # Estimate payroll if not available (roughly 30-40% of revenue)
        estimated_payroll = revenue * 0.35

        # Get exposure values
        exposures = {
            "revenue": revenue,
            "payroll": estimated_payroll,
            "property_value": extracted_email.property_value or 0,
            "vehicle_count": extracted_email.vehicle_count or 0,
        }

        line_items = []
        for rate in rates:
            premium = self._calculate_line_item(rate, exposures)
            line_items.append(premium)

        total_base = sum(item.base_premium for item in line_items)

        return PremiumCalculation(
            line_items=line_items,
            total_base_premium=total_base,
            calculation_timestamp=datetime.utcnow(),
        )

    def _calculate_line_item(
        self,
        rate: RateInfo,
        exposures: dict
    ) -> PremiumLineItem:
        """Calculate premium for a single coverage line."""
        rate_basis = rate.rate_basis.lower()

        # Map rate basis to exposure value
        if "revenue" in rate_basis or "1000" in rate_basis:
            exposure_value = exposures["revenue"]
            multiplier = exposure_value / 1000
            calculation_notes = f"${exposure_value:,.0f} revenue / 1000 * ${rate.base_rate}"
        elif "payroll" in rate_basis or "100" in rate_basis:
            exposure_value = exposures["payroll"]
            multiplier = exposure_value / 100
            calculation_notes = f"${exposure_value:,.0f} payroll / 100 * ${rate.base_rate}"
        elif "vehicle" in rate_basis:
            exposure_value = exposures["vehicle_count"]
            multiplier = exposure_value
            calculation_notes = f"{int(exposure_value)} vehicles * ${rate.base_rate} per vehicle"
        elif "tiv" in rate_basis or "percent" in rate_basis or "property" in rate_basis:
            exposure_value = exposures["property_value"]
            multiplier = exposure_value * (rate.base_rate / 100)
            calculation_notes = f"${exposure_value:,.0f} property value * {rate.base_rate}%"
        else:
            # Default to revenue-based
            exposure_value = exposures["revenue"]
            multiplier = exposure_value / 1000
            calculation_notes = f"Default revenue-based calculation"

        # Calculate premium
        calculated_premium = rate.base_rate * multiplier if "percent" not in rate_basis else multiplier
        if "percent" in rate_basis:
            calculated_premium = exposures["property_value"] * (rate.base_rate / 100)

        # Apply minimum premium
        final_premium = max(calculated_premium, rate.minimum_premium)

        if final_premium == rate.minimum_premium and calculated_premium < rate.minimum_premium:
            calculation_notes += f" (minimum premium applied)"

        return PremiumLineItem(
            coverage_type=rate.coverage_type,
            base_premium=round(final_premium, 2),
            rate_used=rate.base_rate,
            rate_basis=rate.rate_basis,
            exposure_value=exposure_value,
            calculation_notes=calculation_notes,
        )
