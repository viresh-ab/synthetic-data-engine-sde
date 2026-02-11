# Quickstart Guide

Get started with the Synthetic Data Platform in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# Navigate to project directory
cd synthetic-data-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY if using text generation
```

## Quick Test

### Method 1: Command Line

```bash
# Generate 100 synthetic rows from sample data
python -m src.cli \
  --input data/sample_input.csv \
  --output data/output/synthetic.csv \
  --rows 100 \
  --seed 42 \
  --validate
```

### Method 2: Web UI

```bash
# Start the web interface
streamlit run src/streamlit_app.py

# Open your browser to http://localhost:8501
# Upload data/sample_input.csv and click Generate
```

### Method 3: API

```bash
# Terminal 1: Start API server
python -m src.api

# Terminal 2: Test the API
curl -X POST "http://localhost:8000/generate" \
  -F "file=@data/sample_input.csv" \
  -F "config_path=config/default.yaml" \
  -F "validate=true"
```

## Using Different Presets

### Analytics (strict validation, numeric focus)

```bash
python -m src.cli \
  --input data/sample_input.csv \
  --output data/output/analytics.csv \
  --config config/analytics.yaml
```

### Survey (expressive text, conversational)

```bash
python -m src.cli \
  --input data/sample_input.csv \
  --output data/output/survey.csv \
  --config config/survey.yaml
```

## Your Own Data

```bash
# Use your own CSV file
python -m src.cli \
  --input /path/to/your/data.csv \
  --output /path/to/output.csv \
  --rows 5000 \
  --seed 123 \
  --validate
```

## Programmatic Usage

```python
from src.core import Config, Orchestrator
import pandas as pd

# Load your data
input_df = pd.read_csv('your_data.csv')

# Configure
config = Config('config/default.yaml')
config.config['generation']['rows'] = 5000

# Generate
orchestrator = Orchestrator(config)
synthetic_df = orchestrator.generate(input_df)

# Save
synthetic_df.to_csv('output.csv', index=False)
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore configuration options in `config/` directory
- Review sample data in `data/sample_input.csv`
- Run tests: `pytest tests/`

## Troubleshooting

### Text generation not working

Make sure you have set `OPENAI_API_KEY` in your `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### Import errors

Make sure you're running commands from the project root directory and the virtual environment is activated.

### Low quality output

Try adjusting configuration:
- Increase `generation.rows` for more variety
- Change `text.temperature` (0.4 = consistent, 0.9 = creative)
- Use `config/analytics.yaml` for stricter validation

## Support

For issues and questions, please check:
- [README.md](README.md) - Full documentation
- [tests/](tests/) - Example usage
- GitHub Issues - Report bugs
