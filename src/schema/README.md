# Synthetic Data Platform

Enterprise-grade synthetic data generation system for analytics, ML, and privacy-safe testing.

## Features

- 🔍 Automatic schema inference with confidence scoring
- 🔢 Numeric data generation with correlations
- 📝 LLM-powered text generation
- 🔒 Privacy-safe PII generation
- ✅ Multi-layer validation (quality + privacy)
- 🎯 Domain-specific rules (fashion, finance, healthcare)
- 🔁 Deterministic & reproducible (seeded)
- 📊 Multiple interfaces: CLI, API, Web UI

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd synthetic-data-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your settings (especially OPENAI_API_KEY if using LLM)
```

### Basic Usage

#### CLI

```bash
# Generate synthetic data from input CSV
python -m src.cli \
  --input data/sample_input.csv \
  --output data/output/synthetic.csv \
  --config config/default.yaml \
  --validate

# Use a different preset
python -m src.cli \
  --input data/sample_input.csv \
  --output data/output/synthetic.csv \
  --config config/analytics.yaml
```

#### API

```bash
# Start the API server
python -m src.api

# In another terminal, test the API
curl -X POST "http://localhost:8000/generate" \
  -F "file=@data/sample_input.csv" \
  -F "config_path=config/default.yaml"
```

#### Web UI

```bash
# Start Streamlit app
streamlit run src/streamlit_app.py
```

---

## Project Structure

```
synthetic-data-platform/
├── src/                      # Source code
│   ├── core/                 # Core orchestration
│   │   ├── config.py         # Configuration management
│   │   ├── orchestrator.py   # Main generation controller
│   │   └── router.py         # Pipeline routing
│   ├── schema/               # Schema inference
│   │   ├── profiler.py       # Column type inference
│   │   ├── types.py          # Column type definitions
│   │   └── pii_detector.py   # PII detection
│   ├── generators/           # Data generators
│   │   ├── numeric.py        # Numeric generation
│   │   ├── text.py           # Text generation (LLM)
│   │   └── pii.py            # PII generation
│   ├── validation/           # Quality & privacy checks
│   │   ├── quality.py        # Quality metrics
│   │   └── privacy.py        # Privacy risk assessment
│   ├── cli.py                # Command-line interface
│   ├── api.py                # REST API
│   └── streamlit_app.py      # Web UI
├── config/                   # Configuration presets
│   ├── default.yaml
│   ├── analytics.yaml
│   └── survey.yaml
├── data/
│   ├── knowledge/            # Domain knowledge files
│   └── sample_input.csv      # Example input
├── tests/                    # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

The platform uses YAML configuration files with multiple presets:

- **default.yaml**: General-purpose, balanced settings
- **analytics.yaml**: Analytics/ML optimized (strict validation, numeric focus)
- **survey.yaml**: Survey responses (expressive text, conversational tone)

### Key Configuration Sections

```yaml
generation:
  rows: 1000              # Number of synthetic rows
  seed: 42                # Random seed for reproducibility

numeric:
  min: 0
  max: 100
  precision: 2
  allow_correlations: true

text:
  enabled: true
  model: gpt-4-turbo
  temperature: 0.7
  max_tokens: 256

pii:
  anonymize: true
  ensure_uniqueness: true

validation:
  quality:
    min_variance: 0.01
    min_unique_ratio: 0.05
  privacy:
    min_k_anonymity: 5
    max_uniqueness_ratio: 0.05
```

---

## Domain Knowledge

The platform supports domain-specific rules stored in `data/knowledge/`:

- **fashion/**: Product pricing, inventory rules, text guidelines
- **finance/**: Account balances, interest rates, transaction rules
- **healthcare/**: Vital signs, lab values, physiological limits

---

## Examples

### Generate Analytics Dataset

```bash
python -m src.cli \
  --input data/ecommerce_schema.csv \
  --output data/synthetic_analytics.csv \
  --config config/analytics.yaml \
  --validate
```

### Generate Survey Responses

```bash
python -m src.cli \
  --input data/survey_template.csv \
  --output data/synthetic_survey.csv \
  --config config/survey.yaml \
  --rows 5000
```

---

## API Reference

### POST /generate

Generate synthetic data from uploaded CSV.

**Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "file=@input.csv" \
  -F "config_path=config/default.yaml" \
  -F "validate=true"
```

**Response:**
```json
{
  "data": [...],
  "validation": {
    "passed": true,
    "issues": []
  }
}
```

### POST /profile

Profile an input dataset (schema inference only).

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_generators.py
```

---

## Environment Variables

```bash
# Required for text generation
OPENAI_API_KEY=sk-...

# Optional
LOG_LEVEL=INFO
OUTPUT_PATH=data/output
VALIDATION_STRICT=true
```

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## Support

For issues and questions, please open a GitHub issue.
