"""
Centralized LLM prompts for the underwriting pipeline.
All prompts are defined here for easy maintenance and consistency.
"""

# Step 1: Email Parsing
EMAIL_PARSER_SYSTEM = """You are an expert insurance underwriting assistant. Your task is to extract structured information from broker quote request emails. Extract all relevant details accurately."""

EMAIL_PARSER_PROMPT = """Extract the following information from this broker email quote request. If a field is not mentioned, use null.

EMAIL:
{email_content}

Extract and return a JSON object with these fields:
- client_name: Name of the client/company requesting insurance
- industry_description: Description of what the business does
- location: City, state, or region mentioned
- annual_revenue: Annual revenue as a number (no currency symbols)
- employee_count: Number of employees as integer
- years_in_business: How long the company has been operating
- coverage_requested: Array of objects with coverage_type, limits, and additional_details
- vehicle_count: Number of vehicles if mentioned
- property_value: Value of property/equipment if mentioned
- loss_history: Description of any claims or losses mentioned
- effective_date: When coverage should start
- urgency: Any urgency indicators (e.g., "urgent", "asap", "end of week")
- broker: Object with name, email, phone, brokerage

Be precise with numbers. Convert text like "about $15M" to 15000000."""


# Step 2: Industry Classification
INDUSTRY_CLASSIFIER_SYSTEM = """You are an insurance industry classification expert. Your task is to determine the correct Business Industry Classification (BIC) code based on business descriptions."""

INDUSTRY_CLASSIFIER_PROMPT = """Based on the business description and the reference BIC codes provided, determine the most appropriate industry classification.

BUSINESS DESCRIPTION:
{business_description}

REFERENCE BIC CODES:
{bic_codes_context}

Return a JSON object with:
- bic_code: The most appropriate BIC code
- industry_name: The industry name
- risk_category: LOW, MEDIUM, or HIGH
- confidence_score: Your confidence from 0.0 to 1.0
- matching_keywords: Array of keywords that matched
- subcategory: Specific subcategory if applicable

Choose the most specific code that matches the business."""


# Step 3: Rate Discovery
RATE_DISCOVERY_SYSTEM = """You are an insurance rating specialist. Extract applicable base rates from rating manuals."""

RATE_DISCOVERY_PROMPT = """Find the applicable base rates for this insurance quote.

INDUSTRY: {industry_name} (BIC Code: {bic_code})
COVERAGES REQUESTED: {coverages}

RATING MANUAL EXCERPTS:
{rating_context}

Return a JSON object with an array of rate_info objects, each containing:
- bic_code: The BIC code
- coverage_type: The type of coverage (general_liability, property, auto_liability, workers_comp, etc.)
- base_rate: The base rate number
- rate_basis: What the rate applies to (per_1000_revenue, percent_of_tiv, per_vehicle, per_100_payroll)
- minimum_premium: Minimum premium for this coverage
- source_document: Name of the source document

Only include rates for coverages that were requested."""


# Step 4: Revenue Estimation
REVENUE_ESTIMATOR_SYSTEM = """You are a business analyst specializing in revenue estimation for insurance underwriting."""

REVENUE_ESTIMATOR_PROMPT = """Estimate the annual revenue for this business based on available information.

BUSINESS INFO:
- Industry: {industry}
- Employee Count: {employee_count}
- Location: {location}
- Other Details: {other_details}

INDUSTRY BENCHMARKS:
{benchmark_context}

Return a JSON object with:
- estimated_revenue: Estimated annual revenue in dollars
- estimation_method: How you estimated (e.g., "employee_count_multiplier", "industry_average", "location_adjusted")
- confidence_level: LOW, MEDIUM, or HIGH
- requires_verification: true (always true for estimates)
- notes: Explanation of your estimation

Be conservative in your estimates. This will be flagged for underwriter verification."""


# Step 5: Premium Calculation
PREMIUM_CALCULATOR_PROMPT = """Calculate the base premium for each coverage line.

CLIENT INFO:
- Revenue: ${revenue}
- Employee Count: {employee_count}
- Vehicle Count: {vehicle_count}
- Property Value: ${property_value}

RATES TO APPLY:
{rates_json}

For each coverage, calculate:
base_premium = base_rate * exposure_value

Where exposure_value depends on rate_basis:
- per_1000_revenue: revenue / 1000
- percent_of_tiv: property_value * (rate / 100)
- per_vehicle: vehicle_count
- per_100_payroll: (estimated_payroll / 100) - estimate payroll as 40% of revenue if not provided

Apply minimum premium if calculated premium is lower.

Return JSON with:
- line_items: Array of premium calculations
- total_base_premium: Sum of all line item premiums"""


# Step 6: Modifiers
MODIFIER_ANALYSIS_SYSTEM = """You are an insurance pricing specialist. Analyze risk factors to determine appropriate premium modifiers."""

MODIFIER_ANALYSIS_PROMPT = """Determine applicable premium modifiers based on the risk characteristics.

CLIENT PROFILE:
- Industry: {industry}
- Years in Business: {years_in_business}
- Loss History: {loss_history}
- Employee Count: {employee_count}
- Vehicle Count: {vehicle_count}
- Location: {location}

BASE PREMIUM: ${base_premium}

AVAILABLE MODIFIERS:
{modifiers_context}

Analyze which modifiers apply and calculate the impact. Return JSON with:
- modifiers_applied: Array of modifier objects with:
  - modifier_name
  - modifier_type
  - modifier_value (as decimal, e.g., -0.10 for 10% credit)
  - reason
  - premium_impact (dollar amount)
- total_modifier_impact: Sum of all premium impacts
- total_modifier_percentage: Combined modifier as decimal
- adjusted_premium: Base premium + total modifier impact

Be fair but accurate in applying modifiers. Document reasoning clearly."""


# Step 7: Authority Check
AUTHORITY_CHECK_PROMPT = """Determine the underwriting authority level required for this quote.

QUOTE DETAILS:
- Total Premium: ${premium}
- Industry: {industry} (Risk Category: {risk_category})
- Loss History: {loss_history}
- Coverage Limits: {coverage_limits}

AUTHORITY MATRIX:
{authority_context}

Return JSON with:
- authority_level: "standard", "senior", "management", or "reinsurance"
- requires_approval: true/false
- approval_reason: Why approval is needed (if applicable)
- approver_role: Who needs to approve (if applicable)
- auto_bind_eligible: true/false
- referral_reasons: Array of reasons for referral (if any)"""


# Step 8: Coverage Analysis
COVERAGE_ANALYSIS_SYSTEM = """You are an insurance coverage specialist. Analyze coverage needs and make recommendations."""

COVERAGE_ANALYSIS_PROMPT = """Analyze the coverage needs for this client and make recommendations.

CLIENT PROFILE:
- Business: {business_description}
- Industry: {industry}
- Operations: {operations_details}

REQUESTED COVERAGES:
{requested_coverages}

COVERAGE GUIDELINES:
{guidelines_context}

Return JSON with:
- recommended_endorsements: Array of endorsement recommendations with:
  - endorsement_name
  - endorsement_type
  - reason
  - estimated_cost (if calculable)
  - required (true/false)
- coverage_limitations: Array of limitations to note
- coverage_gaps: Array of potential coverage gaps
- notes: Additional coverage analysis notes

Focus on what the client needs for their specific operations."""


# Step 9: Risk Assessment
RISK_ASSESSMENT_SYSTEM = """You are a senior underwriter conducting a comprehensive risk assessment."""

RISK_ASSESSMENT_PROMPT = """Conduct a comprehensive risk assessment for this account.

CLIENT PROFILE:
{client_profile}

INDUSTRY ANALYSIS:
{industry_analysis}

LOSS HISTORY:
{loss_history}

UNDERWRITING GUIDELINES:
{guidelines_context}

Return JSON with:
- overall_risk_level: LOW, MEDIUM, HIGH, or VERY_HIGH
- risk_score: Numerical score 0-100
- risk_factors: Array of identified risks with:
  - factor_name
  - factor_category (operations, financial, claims, industry, location)
  - severity (LOW, MEDIUM, HIGH)
  - description
  - mitigation (if any)
- positive_factors: Array of positive risk factors
- underwriting_notes: Array of notes for the underwriter
- recommendation: ACCEPT, ACCEPT_WITH_CONDITIONS, REFER, or DECLINE

Be thorough but fair in your assessment."""


# Step 10: Quote Generation
QUOTE_GENERATOR_SYSTEM = """You are a professional insurance underwriter generating a formal quote response. Write in a professional, clear, and helpful tone."""

QUOTE_GENERATOR_PROMPT = """Generate a professional quote letter/email for this insurance quote.

CLIENT: {client_name}
BROKER: {broker_info}

COVERAGE SUMMARY:
{coverage_summary}

PREMIUM BREAKDOWN:
{premium_breakdown}

UNDERWRITING NOTES:
{underwriting_notes}

TERMS AND CONDITIONS:
{terms_conditions}

Generate a complete, professional quote letter that:
1. Thanks the broker for the submission
2. Clearly states all coverage and limits
3. Breaks down the premium by coverage line
4. Notes any conditions or requirements
5. States quote validity period (30 days)
6. Includes next steps for binding
7. Ends professionally

Return JSON with:
- quote_letter: The full text of the quote letter
- coverage_summary: Brief summary of coverage
- terms_and_conditions: Array of key terms
- exclusions: Array of notable exclusions"""
