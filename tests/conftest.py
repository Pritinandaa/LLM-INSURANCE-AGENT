"""
Pytest configuration and fixtures.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_email_construction():
    """Sample construction company quote request email."""
    return """
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


@pytest.fixture
def sample_email_restaurant():
    """Sample restaurant quote request email."""
    return """
Subject: Insurance Quote - Mario's Italian Kitchen

Hello,

I'm reaching out on behalf of Mario's Italian Kitchen, a family-owned
restaurant in downtown Chicago. They've been serving the community for
8 years and are looking for a comprehensive insurance package.

Details:
- Full-service Italian restaurant
- Annual revenue: approximately $2.2 million
- 22 employees (mix of full-time and part-time)
- They have a full bar with liquor license (about 30% of revenue from alcohol)
- Clean claims history

They need:
- General Liability ($1M/$2M)
- Property coverage for equipment and contents
- Liquor Liability
- Workers Compensation

Best regards,
Michael Torres
Torres Insurance Agency
"""


@pytest.fixture
def sample_email_tech():
    """Sample tech company quote request email."""
    return """
Subject: Coverage Request - CloudSync Technologies

Hi Insurance Team,

We're a SaaS company called CloudSync Technologies looking for our first
commercial insurance package.

Company Info:
- Founded 3 years ago in Austin, Texas
- We develop cloud-based project management software
- Current ARR (annual recurring revenue): $4.5 million
- 35 employees, all W-2, mostly remote with a small office

What we're looking for:
1. General Liability - standard limits
2. Professional Liability / E&O
3. Cyber Liability - at least $1M

Thanks,
Jennifer Chen
COO, CloudSync Technologies
"""


@pytest.fixture
def mock_mongodb(mocker):
    """Mock MongoDB client for unit tests."""
    mock_client = mocker.MagicMock()
    mock_db = mocker.MagicMock()
    mock_collection = mocker.MagicMock()

    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.aggregate.return_value = []
    mock_collection.find_one.return_value = None

    mocker.patch('src.core.mongodb_client.get_mongodb_client', return_value=mock_client)
    mocker.patch('src.core.mongodb_client.get_database', return_value=mock_db)
    mocker.patch('src.core.mongodb_client.get_collection', return_value=mock_collection)

    return mock_collection


@pytest.fixture
def mock_fireworks(mocker):
    """Mock Fireworks client for unit tests."""
    mock_response = {
        "client_name": "Test Company",
        "industry_description": "Test Industry",
        "annual_revenue": 1000000,
        "employee_count": 50,
    }

    mock_client = mocker.MagicMock()
    mock_client.generate_json.return_value = mock_response
    mock_client.generate_embedding.return_value = [0.1] * 768
    mock_client.generate_embeddings.return_value = [[0.1] * 768]

    mocker.patch('src.core.fireworks_client.get_fireworks_client', return_value=mock_client)

    return mock_client
