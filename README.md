# AI-Powered Insurance Underwriting System

Automate insurance underwriting with AI. Process broker emails and generate professional quotes in minutes instead of days.

## Features

- **10-Step AI Pipeline**: Email parsing, industry classification, rate discovery, premium calculation, risk assessment, and quote generation
- **RAG-Powered**: Retrieval-Augmented Generation ensures decisions are grounded in your underwriting documents
- **Fast Processing**: Generate quotes in under 60 seconds
- **Multiple Interfaces**: REST API, CLI tool, or file upload

## Tech Stack

- **Fireworks AI** (Llama 3.3 70B) - Natural language processing and structured output
- **MongoDB Atlas** - Document storage with vector search
- **FastAPI** - High-performance REST API
- **Python 3.9+** - Cross-platform support

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd insurance-underwriting-ai

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
# macOS/Linux:
cp .env.example .env
# Windows:
copy .env.example .env

# Edit .env and add your API keys (see Configuration section below)
```

### 3. Set Up Database

```bash
# Run setup script (creates collections and seeds data)
python scripts/setup_and_seed.py
```

### 4. Process Your First Quote

**Using CLI:**
```bash
python -m cli.process_quote data/sample_emails/construction_company.txt
```

**Using API:**
```bash
# Start the server
python -m uvicorn src.main:app --reload

# Send a quote request
curl -X POST http://localhost:8000/api/quotes/process \
  -H "Content-Type: application/json" \
  -d '{"email_content": "Hi, I need a quote for ABC Corp..."}'
```

## Configuration

### Fireworks AI Setup

1. Go to https://fireworks.ai
2. Create a free account (includes $1 credit)
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env`: `FIREWORKS_API_KEY=your_key_here`

### MongoDB Atlas Setup

1. Go to https://www.mongodb.com/cloud/atlas
2. Create a free account
3. Create a new cluster (M0 Free Tier)
4. Create a database user (Database Access)
5. Add your IP to whitelist (Network Access → Add IP → Allow from anywhere)
6. Get connection string (Connect → Drivers)
7. Add to `.env`: `MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/`

### Vector Search Index (Required for Full Functionality)

After seeding data, create vector search indexes in MongoDB Atlas:

1. Go to your cluster → Atlas Search → Create Search Index
2. Choose JSON Editor
3. Create an index for each collection with this definition:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 768,
        "similarity": "cosine"
      }
    }
  }
}
```

Collections to index:
- `bic_codes`
- `rating_manuals`
- `underwriting_guidelines`
- `modifiers`

## Usage

### CLI Tool

```bash
# Process an email file
python -m cli.process_quote data/sample_emails/construction_company.txt

# Process with JSON output
python -m cli.process_quote data/sample_emails/restaurant.txt --json

# Process inline text
python -m cli.process_quote --text "Hi, I need a quote for ABC Corp..."
```

### REST API

```bash
# Start the server
python -m uvicorn src.main:app --reload

# API Documentation
open http://localhost:8000/docs
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/quotes/process` | Submit email text for quote |
| POST | `/api/quotes/process-file` | Upload email file |
| GET | `/api/quotes/{id}` | Get quote details |
| GET | `/api/quotes` | List recent quotes |
| GET | `/health` | Health check |

### Sample Request

```bash
curl -X POST http://localhost:8000/api/quotes/process \
  -H "Content-Type: application/json" \
  -d '{
    "email_content": "Subject: Quote Request - ABC Construction\n\nHi, I need a quote for ABC Construction Corp. They are a commercial construction company in Texas with $15M revenue and 85 employees. They need General Liability $2M/$4M and Auto for 25 vehicles.\n\nThanks,\nSarah"
  }'
```

## Project Structure

```
insurance-underwriting-ai/
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── prompts.py           # LLM prompts
│   ├── core/                # Core services
│   │   ├── fireworks_client.py
│   │   ├── mongodb_client.py
│   │   ├── embedding_service.py
│   │   └── vector_search.py
│   ├── pipeline/            # 10-step pipeline
│   │   ├── orchestrator.py
│   │   ├── models.py
│   │   └── steps/
│   └── api/                 # API routes
├── cli/                     # CLI tool
├── data/                    # Seed data
├── scripts/                 # Setup scripts
└── tests/                   # Test suite
```

## The 10-Step Pipeline

1. **Email Parser** - Extract structured data from broker emails
2. **Industry Classifier** - Determine BIC code using vector search
3. **Rate Discovery** - Find applicable base rates from rating manuals
4. **Revenue Estimation** - Estimate revenue if not provided
5. **Premium Calculation** - Calculate base premium by coverage
6. **Modifiers** - Apply credits/debits based on risk factors
7. **Authority Check** - Determine approval requirements
8. **Coverage Analysis** - Recommend endorsements and identify gaps
9. **Risk Assessment** - Comprehensive risk evaluation
10. **Quote Generation** - Generate professional quote letter

## Sample Output

```
Insurance Quote Processing
----------------------------------------
Processing quote request...
  [ 1/10] Email Parser              [done]
  [ 2/10] Industry Classifier       [done]
  [ 3/10] Rate Discovery            [done]
  [ 4/10] Revenue Estimation        [done]
  [ 5/10] Premium Calculation       [done]
  [ 6/10] Modifiers                 [done]
  [ 7/10] Authority Check           [done]
  [ 8/10] Coverage Analysis         [done]
  [ 9/10] Risk Assessment           [done]
  [10/10] Quote Generation          [done]

======================================================================
        QUOTE GENERATED SUCCESSFULLY
======================================================================

Client: ABC Construction Corp
Industry: Construction - Commercial/Industrial (BIC: 44)

Annual Premium: $165,375.00

Premium Breakdown:
  - General Liability: $127,500.00
  - Auto Liability: $18,750.00

Modifiers Applied:
  - Loss History: +$19,125.00 (+15%)

Risk Assessment:
  Level: MEDIUM
  Score: 45/100
  Recommendation: ACCEPT

Quote ID: Q-20240301-ABC12345
Processing time: 47.3 seconds
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific tests
pytest tests/test_pipeline.py
```

## Docker

```bash
# Build image
docker build -t insurance-underwriting-ai .

# Run container
docker run -p 8000:8000 --env-file .env insurance-underwriting-ai

# Or use Docker Compose (includes local MongoDB)
docker-compose up -d
```

## Troubleshooting

### "MongoDB connection failed"
- Check your `MONGODB_URI` in `.env`
- Ensure your IP is whitelisted in MongoDB Atlas
- Verify database user credentials

### "Fireworks API key not configured"
- Add `FIREWORKS_API_KEY` to your `.env` file
- Get a key from https://fireworks.ai

### "Vector search not working"
- Create vector search indexes in MongoDB Atlas (see Configuration section)
- Ensure you're using MongoDB Atlas (local MongoDB doesn't support vector search)

## License

MIT License - see LICENSE file for details.
