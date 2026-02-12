# 🔬 Synthetic Data Generator

A comprehensive, production-ready synthetic data generation system with schema analysis, multi-pipeline generation, quality validation, and privacy guarantees.

## ✨ Features

### 🎯 Smart Generation
- **Numeric Data**: Distribution fitting, correlation preservation, outlier handling
- **Text Data**: LLM-powered generation (Claude, GPT) with template fallback
- **PII Data**: Realistic names, emails, phones, addresses with full anonymization
- **Temporal Data**: Pattern preservation (weekdays, seasonality, business days)
- **Domain Knowledge**: RAG-based generation with business rules and constraints

### ✅ Quality Validation
- **Distribution Similarity**: KS test, Jensen-Shannon divergence
- **Statistical Properties**: Mean, std, quantile matching
- **Correlation Preservation**: Matrix comparison
- **Text Quality**: Vocabulary overlap, length distribution, character statistics
- **PII Validity**: Format checking, diversity analysis

### 🔒 Privacy Protection
- **K-Anonymity**: Configurable group size requirements
- **Re-identification Risk**: Comprehensive risk assessment
- **Uniqueness Analysis**: Rare record detection
- **Data Leakage**: Exact match detection
- **Privacy Recommendations**: Automated improvement suggestions

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd synthetic-data-generator

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

## 🚀 Quick Start

### 1. Streamlit Web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 2. REST API

```bash
uvicorn api:app --reload
```

API docs available at http://localhost:8000/docs

### 3. Command Line

```bash
# Generate synthetic data
python cli.py generate input.csv output.csv --rows 1000

# Validate synthetic data
python cli.py validate reference.csv synthetic.csv --quality --privacy

# Analyze data schema
python cli.py analyze data.csv --detailed
```

### 4. Python API

```python
from src.config import get_default_config
from src.orchestrator import DataOrchestrator
import pandas as pd

# Load reference data
reference = pd.read_csv("data.csv")

# Configure generator
config = get_default_config()
config.generation.num_rows = 1000
config.generation.seed = 42

# Generate synthetic data
orchestrator = DataOrchestrator(config)
result = orchestrator.generate(reference_data=reference)

# Access generated data
synthetic_data = result.data
print(f"Generated {len(synthetic_data)} rows")
```

## 📚 Usage Examples

### Generate Analytics Data

```python
from src.config import create_analytics_preset
from src.orchestrator import DataOrchestrator

config = create_analytics_preset()
config.generation.num_rows = 5000

orchestrator = DataOrchestrator(config)
result = orchestrator.generate(reference_data=df)
```

### Generate with Privacy Guarantees

```python
from src.config import create_healthcare_preset

config = create_healthcare_preset()
config.pii.anonymization_level = "full"
config.validation.k_anonymity = 10

orchestrator = DataOrchestrator(config)
result = orchestrator.generate(reference_data=df)

# Validate privacy
from src.validation import PrivacyValidator

validator = PrivacyValidator(config)
report = validator.validate(
    synthetic=result.data,
    reference=df
)

print(f"Privacy Risk: {report.overall_risk}")
print(f"K-Anonymity: {report.k_anonymity_score}")
```

### Custom Column Configuration

```python
config = get_default_config()

# Override specific columns
config.column_configs = {
    'age': {
        'type': 'numeric',
        'distribution': 'normal',
        'min': 18,
        'max': 100
    },
    'email': {
        'type': 'pii_email',
        'domains': ['company.com', 'example.org']
    }
}

orchestrator = DataOrchestrator(config)
result = orchestrator.generate(reference_data=df)
```

## 🎨 Streamlit UI

The Streamlit app provides a complete workflow:

1. **Home**: Overview and quick start
2. **Upload**: Upload reference data (CSV, Excel, JSON)
3. **Configure**: Set generation parameters and presets
4. **Generate**: Generate synthetic data with progress tracking
5. **Validate**: Run quality and privacy checks
6. **Export**: Download results in multiple formats

### Features:
- Real-time progress tracking
- Interactive schema analysis
- Visual quality metrics
- Privacy risk dashboard
- Multi-format export (CSV, Excel, JSON, Parquet)

## 🌐 REST API

### Endpoints

```
POST   /upload              Upload reference data
GET    /analyze/{data_id}   Analyze schema
POST   /generate            Generate synthetic data
POST   /validate/{data_id}  Validate data
GET    /download/{data_id}  Download data
GET    /jobs/{job_id}       Get job status
GET    /presets             List presets
```

### Example: Generate via API

```bash
# Upload reference data
curl -X POST "http://localhost:8000/upload" \
  -F "file=@reference.csv"

# Generate synthetic data
curl -X POST "http://localhost:8000/generate?data_id=<data_id>" \
  -H "Content-Type: application/json" \
  -d '{
    "num_rows": 1000,
    "seed": 42,
    "preserve_correlations": true
  }'

# Download result
curl -X GET "http://localhost:8000/download/<data_id>?format=csv" \
  -o synthetic.csv
```

## 🖥️ CLI Commands

### Generate

```bash
# Basic generation
python cli.py generate input.csv output.csv --rows 1000

# With preset
python cli.py generate input.csv output.csv --preset analytics --rows 5000

# With custom config
python cli.py generate input.csv output.csv --config my_config.yaml

# With seed for reproducibility
python cli.py generate input.csv output.csv --seed 42
```

### Validate

```bash
# Quality validation only
python cli.py validate reference.csv synthetic.csv --quality

# Privacy validation only
python cli.py validate reference.csv synthetic.csv --privacy

# Both with custom thresholds
python cli.py validate reference.csv synthetic.csv \
  --quality --privacy \
  --threshold 0.85 \
  --k-anonymity 10

# Save report
python cli.py validate reference.csv synthetic.csv \
  --quality --privacy \
  --output validation_report.json
```

### Analyze

```bash
# Basic analysis
python cli.py analyze data.csv

# Detailed with PII detection
python cli.py analyze data.csv --detailed

# Save to JSON
python cli.py analyze data.csv --detailed --output schema.json
```

### Config

```bash
# List presets
python cli.py config list

# Show preset details
python cli.py config show analytics

# Create custom config
python cli.py config create my_config.yaml
```

## 📋 Configuration

### Preset Configurations

- **default**: Balanced settings for general use
- **analytics**: Correlation preservation, high quality threshold
- **survey**: Template-based text, moderate privacy
- **healthcare**: Full anonymization, strict privacy (k=10)
- **finance**: Precision control, strict validation

### Custom Configuration

Create a YAML configuration file:

```yaml
generation:
  num_rows: 1000
  seed: 42
  batch_size: 100
  enable_parallel: true

numeric:
  preserve_correlations: true
  distribution_fitting: auto
  range_enforcement: true
  decimal_places: 2

text:
  llm_provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0.7
  use_templates: false

pii:
  locale: en_US
  anonymization_level: full
  
validation:
  enable_quality_checks: true
  enable_privacy_checks: true
  quality_threshold: 0.8
  k_anonymity: 5
```

## 🏗️ Architecture

```
synthetic-data-generator/
├── src/
│   ├── config.py           # Configuration management
│   ├── orchestrator.py     # Main orchestrator
│   ├── schema/             # Schema analysis
│   ├── generators/         # Data generators
│   ├── validation/         # Quality & privacy
│   └── utils.py            # Utilities
├── app.py                  # Streamlit UI
├── api.py                  # FastAPI server
└── cli.py                  # CLI interface
```

### Key Components

- **Orchestrator**: Coordinates generation workflow
- **Schema Analyzer**: Detects types, patterns, PII
- **Generators**: Specialized per data type
- **Validators**: Quality and privacy checks
- **Config System**: Flexible configuration management

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
flake8 src/
```

### Building Documentation

```bash
cd docs
make html
```

## 📊 Supported Data Types

- ✅ Numeric (int, float) with correlations
- ✅ Text (short, long, structured)
- ✅ Categorical (low/high cardinality)
- ✅ PII (names, emails, phones, addresses, IDs)
- ✅ Temporal (dates, times, datetimes)
- ✅ Boolean
- ✅ JSON/nested structures

## 📈 Performance

- **Small datasets** (<10K rows): Seconds
- **Medium datasets** (10K-100K rows): Minutes
- **Large datasets** (100K-1M rows): 10-30 minutes
- **Parallel processing**: 2-4x speedup on multi-core systems
- **Memory efficient**: Batch processing for large datasets

## 🔐 Privacy & Security

- K-anonymity validation
- Re-identification risk assessment
- No exact record duplication
- PII anonymization
- Quasi-identifier detection
- Privacy recommendations

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with pandas, numpy, scipy
- LLM integration via Anthropic Claude API
- UI powered by Streamlit
- API powered by FastAPI
- CLI using Rich for beautiful terminal output

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs`
- **Examples**: `/examples`

---

**Made with ❤️ for privacy-preserving data science**
