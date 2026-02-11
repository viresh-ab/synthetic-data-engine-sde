# Synthetic Data Generator - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Generation Pipeline](#data-generation-pipeline)
4. [Configuration System](#configuration-system)
5. [Quality Validation](#quality-validation)
6. [Privacy Protection](#privacy-protection)
7. [API Reference](#api-reference)
8. [Development Guide](#development-guide)

---

## Architecture Overview

The Synthetic Data Generator follows a modular, pipeline-based architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐      │
│  │ Streamlit│  │  FastAPI │  │  CLI (Rich)      │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────────────┘      │
└───────┼─────────────┼─────────────┼────────────────────┘
        │             │             │
┌───────▼─────────────▼─────────────▼────────────────────┐
│              Orchestration Layer                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          DataOrchestrator                        │  │
│  │  - Schema Analysis                               │  │
│  │  - Pipeline Routing                              │  │
│  │  - Batch Processing                              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│              Generation Pipelines                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Numeric  │  │   Text   │  │   PII    │             │
│  │ Pipeline │  │ Pipeline │  │ Pipeline │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│  ┌──────────┐  ┌──────────────────────────────┐       │
│  │Temporal  │  │   Knowledge (RAG)            │       │
│  │ Pipeline │  │   Pipeline                   │       │
│  └──────────┘  └──────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│              Validation Layer                            │
│  ┌─────────────────────┐  ┌─────────────────────┐      │
│  │  Quality Validator  │  │  Privacy Validator  │      │
│  │  - Distribution     │  │  - K-Anonymity      │      │
│  │  - Statistics       │  │  - Re-ID Risk       │      │
│  │  - Correlations     │  │  - Uniqueness       │      │
│  └─────────────────────┘  └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration Management (`src/config.py`)

The configuration system uses dataclasses for type safety and validation:

```python
@dataclass
class Config:
    generation: GenerationConfig
    numeric: NumericConfig
    text: TextConfig
    pii: PIIConfig
    temporal: TemporalConfig
    validation: ValidationConfig
    knowledge: KnowledgeConfig
```

**Key Features:**
- Preset configurations (default, analytics, survey, healthcare, finance)
- Column-level overrides
- YAML/JSON serialization
- Configuration validation
- Merge and inheritance support

### 2. Schema Analysis (`src/schema/analyzer.py`)

Automatic data type detection and profiling:

```python
class SchemaAnalyzer:
    - analyze_dataframe()     # Main analysis entry point
    - detect_pii()            # PII detection
    - profile_statistics()    # Statistical profiling
    - infer_types()          # Type inference
```

**Detection Capabilities:**
- 11 PII types (email, phone, SSN, etc.)
- Numeric vs categorical distinction
- Temporal format inference
- Correlation analysis
- Quality metrics (completeness, uniqueness)

### 3. Data Orchestrator (`src/orchestrator.py`)

Central coordinator for the generation workflow:

```python
class DataOrchestrator:
    def __init__(self, config: Config)
    
    def analyze_schema(self, data: DataFrame) -> DatasetSchema
    def generate(self, reference_data: DataFrame) -> GenerationResult
    def validate(self, synthetic: DataFrame, reference: DataFrame) -> ValidationReport
```

**Workflow:**
1. Schema analysis
2. Pipeline selection
3. Batch processing
4. Result aggregation
5. Validation

---

## Data Generation Pipeline

### Numeric Pipeline

**Distribution Fitting:**
- Normal (Gaussian)
- Uniform
- Exponential
- Lognormal
- Gamma
- Auto-detection (KS test)

**Correlation Preservation:**
- Cholesky decomposition
- Copula-based generation
- Covariance matrix reconstruction

**Features:**
```python
generator = NumericGenerator(config)
synthetic = generator.generate(
    reference_data=df[numeric_cols],
    num_rows=1000,
    preserve_correlations=True,
    seed=42
)
```

### Text Pipeline

**LLM Integration:**
- Anthropic Claude API
- OpenAI GPT API
- HuggingFace models
- Template fallback

**Template System:**
```json
{
  "product_review": {
    "templates": ["This {adjective} product {verb}..."],
    "variables": {...}
  }
}
```

**Features:**
- Length preservation
- Style consistency
- Vocabulary matching
- Sentiment control

### PII Pipeline

**Synthetic PII Generation:**

| Type | Format | Example |
|------|--------|---------|
| Names | Full, First, Last | "John Smith" |
| Emails | Multiple formats | "john.smith@example.com" |
| Phones | US/International | "555-123-4567" |
| Addresses | Full/Partial | "123 Main St, NYC, NY 10001" |
| SSN | Realistic pattern | "123-45-6789" |
| UUID | v4 standard | "550e8400-e29b-41d4-a716-..." |

**Privacy Guarantee:**
- Zero real PII leakage
- Consistency checks (email ↔ name)
- Format validation
- Diversity enforcement

### Temporal Pipeline

**Pattern Detection:**
- Weekday vs weekend
- Business days only
- Seasonality
- Time-of-day patterns

**Generation Modes:**
```python
# Pattern-preserving
generator.generate(reference, preserve_patterns=True)

# Regular series
generator.generate_time_series(start, end, freq='D')

# Seasonal
generator.generate_with_seasonality(peak_months=[6,7,8])

# Recurring
generator.generate_recurring_dates(interval_days=7)
```

### Knowledge Pipeline (RAG)

**Domain Knowledge Integration:**
```yaml
# knowledge/healthcare.yaml
constraints:
  age:
    - type: range
      parameters: {min: 0, max: 120}
  
  bmi:
    - type: formula
      parameters:
        formula: "(weight_lbs / (height_inches^2)) * 703"

business_rules:
  - rule_id: pediatric_routing
    condition: "age < 18 SUGGESTS department = 'Pediatrics'"
```

**Constraint Enforcement:**
- Range validation
- Category restrictions
- Dependency rules
- Formula calculations

---

## Configuration System

### Preset Configurations

**Default:**
```yaml
generation:
  num_rows: 1000
  batch_size: 100
validation:
  quality_threshold: 0.8
  k_anonymity: 5
```

**Analytics:**
```yaml
generation:
  num_rows: 10000
  seed: 42
numeric:
  preserve_correlations: true
validation:
  quality_threshold: 0.9
```

**Healthcare:**
```yaml
pii:
  anonymization_level: full
validation:
  k_anonymity: 10
  privacy_threshold: 0.8
```

### Column-Level Overrides

```python
config.column_configs = {
    'age': {
        'type': 'numeric',
        'distribution': 'normal',
        'min': 18,
        'max': 100
    },
    'email': {
        'type': 'pii_email',
        'domains': ['company.com']
    }
}
```

---

## Quality Validation

### Metrics by Data Type

**Numeric:**
- KS test (distribution similarity)
- Statistical properties (mean, std, quantiles)
- Correlation preservation
- Range consistency

**Text:**
- Length distribution
- Vocabulary overlap (Jaccard similarity)
- Character statistics
- Uniqueness ratio

**PII:**
- Format validity (regex patterns)
- No data leakage
- Diversity check

**Temporal:**
- Date range consistency
- Weekday distribution (Jensen-Shannon)
- Monthly distribution (seasonality)

### Quality Report Structure

```python
QualityReport:
  - overall_score: float (0-1)
  - passed: bool
  - metrics: List[QualityMetric]
  - column_scores: Dict[str, float]
  - summary: Dict[str, Any]
```

---

## Privacy Protection

### K-Anonymity

Ensures each combination of quasi-identifiers appears ≥k times:

```python
checker = KAnonymity(k=5)
is_anonymous, actual_k, details = checker.check(
    data,
    quasi_identifiers=['age', 'zipcode', 'gender']
)
```

**Auto-Detection:**
- Cardinality analysis
- Uniqueness ratio
- Quasi-identifier suggestion

### Re-identification Risk

Multi-factor risk assessment:

```python
assessor = ReIdentificationRisk()
risk_score, risk_level, details = assessor.assess(
    data,
    quasi_identifiers=['age', 'zipcode']
)
```

**Risk Components:**
- Uniqueness (ratio of unique records)
- Distinctiveness (entropy analysis)
- Combinatorial risk (total combinations)

**Risk Levels:**
- Low: score < 0.25
- Medium: 0.25 ≤ score < 0.5
- High: 0.5 ≤ score < 0.75
- Critical: score ≥ 0.75

### Privacy Report

```python
PrivacyReport:
  - overall_risk: str (low/medium/high/critical)
  - k_anonymity_score: int
  - reid_risk_score: float
  - recommendations: List[str]
```

---

## API Reference

### Python API

```python
from src.config import get_default_config
from src.orchestrator import DataOrchestrator
import pandas as pd

# Load configuration
config = get_default_config()
config.generation.num_rows = 1000

# Initialize orchestrator
orchestrator = DataOrchestrator(config)

# Generate synthetic data
result = orchestrator.generate(reference_data=df)

# Access results
synthetic_data = result.data
quality_report = result.quality_report
privacy_report = result.privacy_report
```

### REST API

```bash
# Upload reference data
curl -X POST http://localhost:8000/upload \
  -F "file=@data.csv"

# Generate synthetic data
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"data_id": "<id>", "num_rows": 1000}'

# Download results
curl http://localhost:8000/download/<data_id>?format=csv \
  -o synthetic.csv
```

### CLI

```bash
# Generate
python cli.py generate input.csv output.csv --rows 1000 --seed 42

# Validate
python cli.py validate reference.csv synthetic.csv --quality --privacy

# Analyze
python cli.py analyze data.csv --detailed --output schema.json
```

---

## Development Guide

### Setup Development Environment

```bash
# Clone repository
git clone <repo-url>
cd synthetic-data-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_generators.py

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
flake8 src/
pylint src/

# Type checking
mypy src/
```

### Adding New Generators

1. Create generator class in `src/generators/`
2. Implement required methods:
   - `__init__(config)`
   - `generate(reference_data, num_rows, **kwargs)`
3. Add to `src/generators/__init__.py`
4. Add tests in `tests/test_generators.py`
5. Update documentation

### Adding Domain Knowledge

1. Create YAML file in `data/knowledge/`
2. Define constraints, relationships, terminology
3. Add business rules
4. Test with `KnowledgeGenerator`

---

## Performance Optimization

### Batch Processing

```python
config.generation.batch_size = 500  # Larger batches
config.generation.enable_parallel = True
config.generation.max_workers = 8
```

### Memory Management

- Use chunked processing for large datasets
- Enable garbage collection
- Stream results for very large outputs

### Benchmarks

| Dataset Size | Time (Sequential) | Time (Parallel) |
|-------------|-------------------|-----------------|
| 1K rows | ~5 sec | ~3 sec |
| 10K rows | ~30 sec | ~15 sec |
| 100K rows | ~5 min | ~2 min |
| 1M rows | ~45 min | ~20 min |

---

## Troubleshooting

### Common Issues

**Issue: OOM (Out of Memory)**
```python
# Solution: Reduce batch size
config.generation.batch_size = 100
```

**Issue: Low quality scores**
```python
# Solution: Increase sample size, check correlations
config.generation.num_rows = 10000
config.numeric.preserve_correlations = True
```

**Issue: Privacy validation fails**
```python
# Solution: Increase k-anonymity, reduce quasi-identifiers
config.validation.k_anonymity = 10
```

---

## License

MIT License - See LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: <repo-url>/issues
- Documentation: /docs
- Examples: /examples
