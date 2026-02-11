# Synthetic Data Generator - User Guide

A comprehensive guide for users of the Synthetic Data Generator.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Choosing an Interface](#choosing-an-interface)
3. [Step-by-Step Tutorials](#step-by-step-tutorials)
4. [Best Practices](#best-practices)
5. [Common Use Cases](#common-use-cases)
6. [FAQ](#faq)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# Install from source
git clone <repository-url>
cd synthetic-data-generator
pip install -r requirements.txt
```

### Quick Start (5 minutes)

1. **Prepare your data** - CSV, Excel, or JSON file
2. **Launch Streamlit app** - `streamlit run app.py`
3. **Upload your data** - Drag and drop in the UI
4. **Generate** - Click "Generate Synthetic Data"
5. **Download** - Export in your preferred format

---

## Choosing an Interface

### When to Use Each Interface

| Interface | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Streamlit UI** | Non-technical users, exploration | Visual, intuitive, no coding | Limited automation |
| **REST API** | Web apps, microservices | Scalable, language-agnostic | Requires HTTP knowledge |
| **CLI** | Automation, scripts, pipelines | Fast, scriptable | Command-line only |
| **Python API** | Custom applications, Jupyter | Full control, flexibility | Requires Python knowledge |

### Quick Comparison

**Streamlit UI:**
```
✓ Perfect for: Data analysts, business users
✓ No coding required
✓ Visual feedback
✗ Not for automation
```

**REST API:**
```
✓ Perfect for: Web developers, system integrators
✓ Any programming language
✓ Scalable
✗ Requires server setup
```

**CLI:**
```
✓ Perfect for: Data engineers, DevOps
✓ Bash scripts, cron jobs
✓ Fast iteration
✗ Terminal-based only
```

**Python API:**
```
✓ Perfect for: Data scientists, ML engineers
✓ Jupyter notebooks
✓ Maximum flexibility
✗ Python knowledge required
```

---

## Step-by-Step Tutorials

### Tutorial 1: First Synthetic Dataset (Streamlit)

**Scenario:** Generate synthetic customer data

**Steps:**

1. **Launch the app**
   ```bash
   streamlit run app.py
   ```

2. **Upload your data**
   - Navigate to "📤 Upload" page
   - Click "Choose a file"
   - Select your customer data CSV
   - Click "🔍 Analyze Schema"

3. **Review the analysis**
   - Check data types are correct
   - Note any PII detected
   - Review column statistics

4. **Configure generation**
   - Go to "⚙️ Configure" page
   - Set number of rows (e.g., 1000)
   - Choose preset: "Analytics"
   - Set random seed for reproducibility (e.g., 42)

5. **Generate data**
   - Go to "🎲 Generate" page
   - Click "🚀 Generate Synthetic Data"
   - Wait for progress bar to complete

6. **Validate quality**
   - Go to "✅ Validate" page
   - Check both "Quality" and "Privacy"
   - Click "🔍 Run Validation"
   - Review scores and metrics

7. **Export results**
   - Go to "💾 Export" page
   - Choose format (CSV, Excel, JSON)
   - Click "📥 Generate Download"
   - Save your file

**Expected Time:** 10-15 minutes

---

### Tutorial 2: API-Based Generation (REST API)

**Scenario:** Integrate synthetic data into your web application

**Steps:**

1. **Start the API server**
   ```bash
   uvicorn api:app --reload
   ```

2. **Upload reference data**
   ```bash
   curl -X POST http://localhost:8000/upload \
     -F "file=@customers.csv" \
     > upload_response.json
   
   # Extract data_id
   DATA_ID=$(cat upload_response.json | jq -r '.data_id')
   echo $DATA_ID
   ```

3. **Analyze schema (optional)**
   ```bash
   curl http://localhost:8000/analyze/$DATA_ID \
     | jq '.'
   ```

4. **Generate synthetic data**
   ```bash
   curl -X POST "http://localhost:8000/generate?data_id=$DATA_ID" \
     -H "Content-Type: application/json" \
     -d '{
       "num_rows": 1000,
       "seed": 42,
       "preserve_correlations": true
     }' \
     > generate_response.json
   
   SYNTHETIC_ID=$(cat generate_response.json | jq -r '.data_id')
   ```

5. **Validate**
   ```bash
   curl -X POST "http://localhost:8000/validate/$SYNTHETIC_ID?reference_id=$DATA_ID" \
     -H "Content-Type: application/json" \
     -d '{
       "enable_quality_checks": true,
       "enable_privacy_checks": true,
       "quality_threshold": 0.8,
       "k_anonymity": 5
     }' \
     | jq '.'
   ```

6. **Download results**
   ```bash
   curl "http://localhost:8000/download/$SYNTHETIC_ID?format=csv" \
     -o synthetic_customers.csv
   ```

**Integration Example (Python):**
```python
import requests

# Upload
files = {'file': open('customers.csv', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
data_id = response.json()['data_id']

# Generate
config = {'num_rows': 1000, 'seed': 42}
response = requests.post(
    f'http://localhost:8000/generate?data_id={data_id}',
    json=config
)
synthetic_id = response.json()['data_id']

# Download
response = requests.get(
    f'http://localhost:8000/download/{synthetic_id}?format=csv'
)
with open('synthetic.csv', 'wb') as f:
    f.write(response.content)
```

---

### Tutorial 3: Automated Pipeline (CLI)

**Scenario:** Daily synthetic data generation for testing

**Steps:**

1. **Create a generation script** (`generate_daily.sh`)
   ```bash
   #!/bin/bash
   
   DATE=$(date +%Y%m%d)
   INPUT="data/production_sample_${DATE}.csv"
   OUTPUT="data/synthetic_${DATE}.csv"
   REPORT="reports/validation_${DATE}.json"
   
   # Generate
   python cli.py generate $INPUT $OUTPUT \
     --preset analytics \
     --rows 10000 \
     --seed 42
   
   # Validate
   python cli.py validate $INPUT $OUTPUT \
     --quality \
     --privacy \
     --output $REPORT
   
   # Check if validation passed
   PASSED=$(cat $REPORT | jq -r '.quality.passed')
   
   if [ "$PASSED" = "true" ]; then
     echo "✓ Validation passed - synthetic data ready"
     # Upload to S3, database, etc.
   else
     echo "✗ Validation failed - check report"
     exit 1
   fi
   ```

2. **Make script executable**
   ```bash
   chmod +x generate_daily.sh
   ```

3. **Run manually**
   ```bash
   ./generate_daily.sh
   ```

4. **Schedule with cron** (daily at 2 AM)
   ```bash
   crontab -e
   # Add:
   0 2 * * * /path/to/generate_daily.sh >> /var/log/synthetic_gen.log 2>&1
   ```

**Expected Outcome:**
- Daily synthetic datasets
- Validation reports
- Automated quality checks
- Integration with CI/CD

---

### Tutorial 4: Python API for Data Science

**Scenario:** Generate synthetic data for model training

```python
import pandas as pd
from src.config import get_default_config
from src.orchestrator import DataOrchestrator

# Load real customer data (small sample)
real_data = pd.read_csv('customers_sample.csv')

# Configure generation
config = get_default_config()
config.generation.num_rows = 100000  # Large training set
config.generation.seed = 42  # Reproducibility
config.numeric.preserve_correlations = True  # ML feature relationships
config.validation.quality_threshold = 0.9  # High quality for training

# Initialize orchestrator
orchestrator = DataOrchestrator(config)

# Generate synthetic training data
result = orchestrator.generate(reference_data=real_data)

# Check quality
if result.quality_report.passed:
    print(f"✓ Quality score: {result.quality_report.overall_score:.3f}")
    
    # Save for training
    result.data.to_csv('synthetic_training_data.csv', index=False)
    
    # Train your model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X = result.data.drop('target', axis=1)
    y = result.data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
else:
    print("✗ Quality validation failed")
    print(result.quality_report.get_failed_metrics())
```

---

## Best Practices

### Data Preparation

**✓ DO:**
- Provide clean, representative sample data
- Include at least 100 rows (preferably 1000+)
- Use recent data for current patterns
- Document data dictionary

**✗ DON'T:**
- Use data with excessive missing values (>30%)
- Mix unrelated datasets
- Include highly seasonal data without annotation
- Use data with known quality issues

### Configuration

**✓ DO:**
- Use presets as starting points
- Set random seeds for reproducibility
- Test with small datasets first
- Document your configuration

**✗ DON'T:**
- Generate more rows than needed (wastes time)
- Skip validation checks
- Use default config for specialized domains
- Ignore column-level configurations

### Validation

**✓ DO:**
- Always validate before using synthetic data
- Review failed metrics carefully
- Check privacy metrics for PII data
- Save validation reports

**✗ DON'T:**
- Skip privacy checks for sensitive data
- Ignore low quality scores
- Use synthetic data with k-anonymity < 5
- Share data without validation

### Privacy

**✓ DO:**
- Use k-anonymity ≥ 5 for general use
- Use k-anonymity ≥ 10 for healthcare/finance
- Review re-identification risk
- Apply domain knowledge constraints

**✗ DON'T:**
- Share synthetic data without privacy validation
- Use real PII in synthetic data
- Ignore uniqueness warnings
- Mix real and synthetic data

---

## Common Use Cases

### Use Case 1: Software Testing

**Requirement:** Test database with realistic data

**Solution:**
```python
config = get_default_config()
config.generation.num_rows = 10000
config.pii.anonymization_level = 'full'

# Generate
synthetic_data = generate(reference_sample, config)

# Load into test database
load_to_database(synthetic_data, 'test_db')
```

**Benefits:**
- No real customer data exposure
- Unlimited test data
- Reproducible test scenarios

---

### Use Case 2: Machine Learning Training

**Requirement:** Augment small training dataset

**Solution:**
```python
config = get_default_config()
config.generation.num_rows = 50000
config.numeric.preserve_correlations = True
config.validation.quality_threshold = 0.9

# Generate high-quality training data
synthetic = generate(small_real_dataset, config)

# Combine with real data
combined = pd.concat([real_data, synthetic])

# Train model
model.fit(combined)
```

**Benefits:**
- Larger training sets
- Balanced classes
- Preserved feature relationships

---

### Use Case 3: Data Sharing

**Requirement:** Share data with partners while protecting privacy

**Solution:**
```python
config = get_default_config()
config.pii.anonymization_level = 'full'
config.validation.k_anonymity = 10
config.validation.privacy_threshold = 0.8

# Generate privacy-preserving data
synthetic = generate(real_data, config)

# Validate privacy
privacy_report = validate_privacy(synthetic)

if privacy_report.overall_risk == 'low':
    # Safe to share
    share_with_partner(synthetic)
```

**Benefits:**
- Protect customer privacy
- Compliance with regulations (GDPR, HIPAA)
- Enable data collaboration

---

### Use Case 4: Analytics & BI

**Requirement:** Demo dashboards with realistic data

**Solution:**
```python
config = create_analytics_preset()
config.generation.num_rows = 100000

synthetic = generate(real_sample, config)

# Load into BI tool
load_to_tableau(synthetic)
load_to_powerbi(synthetic)
```

**Benefits:**
- Realistic demos
- No data exposure
- Unlimited data volume

---

## FAQ

### General Questions

**Q: How much reference data do I need?**
A: Minimum 100 rows, but 1000+ rows provide better statistical properties. For complex distributions, more is better.

**Q: Can I generate more rows than my reference data?**
A: Yes! That's the point. You can generate 100,000 synthetic rows from 1,000 reference rows.

**Q: Is synthetic data as good as real data?**
A: For most purposes (testing, demos, training), yes. But it's not identical - some rare patterns may be missed.

**Q: How long does generation take?**
A: Depends on size. Typically:
- 1K rows: ~5 seconds
- 10K rows: ~30 seconds
- 100K rows: ~5 minutes
- 1M rows: ~45 minutes

### Privacy Questions

**Q: Can synthetic data be re-identified?**
A: If properly validated with k-anonymity ≥ 5, re-identification risk is very low. Always run privacy validation.

**Q: Is synthetic data GDPR/HIPAA compliant?**
A: Properly anonymized synthetic data is generally compliant, but consult your legal team. Use k-anonymity ≥ 10 for healthcare.

**Q: Can I use synthetic data in production?**
A: For testing, analytics, and ML training: yes. For operational decisions affecting real people: consult stakeholders.

### Technical Questions

**Q: What file formats are supported?**
A: Input: CSV, Excel, JSON. Output: CSV, Excel, JSON, Parquet.

**Q: Can I use my own LLM?**
A: Yes! The text generator supports Claude, GPT, and HuggingFace models. Or use templates.

**Q: How do I handle missing values?**
A: The generator preserves null patterns from reference data. Or use config to specify handling.

**Q: Can I generate data without reference data?**
A: No, the system learns from reference data. But you can use very small samples (100 rows).

---

## Troubleshooting

### Issue: Low Quality Scores

**Symptoms:**
- Quality score < 0.7
- Many failed metrics
- Poor distribution matching

**Solutions:**
1. Increase reference data size
2. Enable correlation preservation
3. Use appropriate presets
4. Check for data quality issues in reference

```python
config.numeric.preserve_correlations = True
config.generation.num_rows = 10000  # Larger dataset
```

---

### Issue: Privacy Validation Fails

**Symptoms:**
- K-anonymity < required threshold
- High re-identification risk
- Many unique records

**Solutions:**
1. Increase k-anonymity threshold
2. Reduce quasi-identifiers
3. Generalize attribute values
4. Generate more records

```python
config.validation.k_anonymity = 10
# Remove overly-specific columns
```

---

### Issue: Slow Generation

**Symptoms:**
- Takes longer than expected
- High CPU/memory usage

**Solutions:**
1. Enable parallel processing
2. Reduce batch size if OOM
3. Disable unnecessary features
4. Use templates instead of LLM for text

```python
config.generation.enable_parallel = True
config.generation.batch_size = 500
config.text.use_templates = True
```

---

### Issue: Out of Memory (OOM)

**Symptoms:**
- Process killed
- Memory error

**Solutions:**
1. Reduce batch size
2. Disable parallel processing
3. Generate in chunks
4. Use streaming for large outputs

```python
config.generation.batch_size = 50
config.generation.enable_parallel = False
```

---

### Issue: LLM API Errors

**Symptoms:**
- "API key not found"
- "Rate limit exceeded"
- "Connection timeout"

**Solutions:**
1. Set API key in environment
2. Use templates as fallback
3. Reduce generation rate
4. Check API quota

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

```python
config.text.use_templates = True  # Fallback
```

---

## Getting Help

### Support Channels

1. **Documentation**: `/docs` directory
2. **Examples**: `/examples` directory
3. **GitHub Issues**: Report bugs, request features
4. **Community**: Discussions, Q&A

### Reporting Issues

When reporting issues, include:
1. Your configuration (YAML)
2. Sample data (anonymized)
3. Error messages
4. Steps to reproduce
5. Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please:
1. Search existing requests
2. Describe use case
3. Explain expected behavior
4. Provide examples

---

## Next Steps

### Learning Path

1. ✅ Complete Quick Start
2. ✅ Try a tutorial
3. ✅ Read best practices
4. ⏭️ Explore advanced features
5. ⏭️ Integrate into your workflow
6. ⏭️ Contribute improvements

### Advanced Topics

- Custom generators
- Domain knowledge creation
- Multi-table generation
- Time series synthesis
- Custom validation rules

### Resources

- Technical docs: `docs/README.md`
- API reference: `docs/API.md`
- Examples: `examples/`
- Tests: `tests/`

---

**Happy Generating! 🎉**
