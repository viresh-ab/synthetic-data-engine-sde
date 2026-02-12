"""
Streamlit Application for Synthetic Data Generator

Single-file app with multiple pages:
- Home: Overview and quick start
- Upload: Data upload and preview
- Configure: Generation settings
- Generate: Run generation
- Validate: Quality and privacy checks
- Export: Download results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import json
import io

# Import our modules
from src.config import Config, ConfigLoader, get_default_config
from src.orchestrator import DataOrchestrator, DatasetSchema, PipelineType
from src.schema import SchemaAnalyzer
from src.validation import QualityValidator, PrivacyValidator
from src.generators import NumericGenerator, TextGenerator, PIIGenerator, TemporalGenerator, CategoricalGenerator
from src.utils import FileHandler, setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'reference_data' not in st.session_state:
        st.session_state.reference_data = None
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'schema' not in st.session_state:
        st.session_state.schema = None
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = None
    if 'privacy_report' not in st.session_state:
        st.session_state.privacy_report = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'


# Page: Home
def page_home():
    """Home page with overview and quick start"""
    st.markdown('<p class="main-header">🔬 Synthetic Data Generator</p>',
                unsafe_allow_html=True)

    st.markdown("""
    Generate high-quality synthetic data that preserves statistical properties 
    and privacy guarantees.
    """)

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Smart Generation")
        st.markdown("""
        - Numeric with correlations
        - LLM-powered text
        - Realistic PII
        - Temporal patterns
        - Domain knowledge
        """)

    with col2:
        st.markdown("### ✅ Quality Validation")
        st.markdown("""
        - Distribution matching
        - Statistical similarity
        - Correlation preservation
        - Format validation
        - Pattern consistency
        """)

    with col3:
        st.markdown("### 🔒 Privacy Checks")
        st.markdown("""
        - K-anonymity
        - Re-identification risk
        - Uniqueness analysis
        - Data leakage detection
        - Privacy recommendations
        """)

    # Quick Start
    st.markdown('<p class="sub-header">Quick Start</p>',
                unsafe_allow_html=True)

    st.markdown("""
    1. **Upload** your reference data (CSV, Excel, or JSON)
    2. **Configure** generation settings and parameters
    3. **Generate** synthetic data with one click
    4. **Validate** quality and privacy metrics
    5. **Export** your synthetic dataset
    """)

# Page: Upload
def page_upload():
    """Data upload and preview page"""
    st.markdown('<p class="main-header">📤 Upload Reference Data</p>',
                unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload your reference dataset"
    )

    if uploaded_file is not None:
        try:
            # Read file
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                data = pd.read_json(uploaded_file)

            st.session_state.reference_data = data

            st.markdown('<div class="success-box">✅ File uploaded successfully!</div>',
                        unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error reading file: {str(e)}</div>',
                        unsafe_allow_html=True)
            return

    # Display data if available
    if st.session_state.reference_data is not None:
        data = st.session_state.reference_data

        # Data overview
        st.markdown('<p class="sub-header">Data Overview</p>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(data):,}")
        col2.metric("Columns", len(data.columns))
        col3.metric(
            "Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        col4.metric("Missing Values", f"{data.isna().sum().sum():,}")

        # Data preview
        st.markdown('<p class="sub-header">Data Preview</p>',
                    unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)

        # Column information
        st.markdown('<p class="sub-header">Column Information</p>',
                    unsafe_allow_html=True)

        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(data[col].dtype) for col in data.columns],
            'Non-Null': [data[col].notna().sum() for col in data.columns],
            'Null %': [(data[col].isna().sum() / len(data) * 100).round(2) for col in data.columns],
            'Unique': [data[col].nunique() for col in data.columns],
        })

        st.dataframe(col_info, use_container_width=True)

        # Schema analysis
        if st.button("🔍 Analyze Schema", use_container_width=True):
            with st.spinner("Analyzing schema..."):
                analyzer = SchemaAnalyzer()
                profiles = analyzer.analyze_dataframe(data)

                # Store in session state
                schema = DatasetSchema()
                schema.num_rows = len(data)

                # Display analysis results
                st.markdown(
                    '<p class="sub-header">Schema Analysis</p>', unsafe_allow_html=True)

                for col_name, profile in profiles.items():
                    with st.expander(f"📊 {col_name} - {profile.inferred_type}"):
                        col1, col2, col3 = st.columns(3)

                        col1.metric("Type", profile.inferred_type)
                        col2.metric("Completeness",
                                    f"{profile.completeness*100:.1f}%")
                        col3.metric(
                            "Uniqueness", f"{profile.uniqueness*100:.1f}%")

                        if profile.contains_pii:
                            st.warning(
                                f"⚠️ Contains PII: {profile.pii_type.value}")

                        if profile.inferred_type == 'numeric':
                            st.write(
                                f"Range: {profile.min_value:.2f} to {profile.max_value:.2f}")
                            st.write(
                                f"Mean: {profile.mean:.2f}, Std: {profile.std:.2f}")

                        if profile.categories:
                            st.write(f"Categories: {len(profile.categories)}")
                            st.write(
                                f"Top values: {list(profile.categories[:5])}")

                st.session_state.schema = schema
                st.success("✅ Schema analysis complete!")


# Page: Configure
def page_configure():
    """Configuration page"""
    st.markdown('<p class="main-header">⚙️ Configuration</p>',
                unsafe_allow_html=True)

    if st.session_state.reference_data is None:
        st.warning("⚠️ Please upload reference data first")
        return

    config = st.session_state.config

    # Preset selection
    st.markdown('<p class="sub-header">Presets</p>', unsafe_allow_html=True)

    preset = st.selectbox(
        "Choose a preset configuration",
        ["Custom", "Analytics", "Survey", "Healthcare", "Finance"],
        help="Pre-configured settings for common use cases"
    )

    if preset != "Custom":
        loader = ConfigLoader()
        if preset.lower() in loader.list_presets():
            config = loader.load_preset(preset.lower())
            st.session_state.config = config
            st.success(f"✅ Loaded {preset} preset")

    # Generation settings
    st.markdown('<p class="sub-header">Generation Settings</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        num_rows = st.number_input(
            "Number of rows",
            min_value=10,
            max_value=1000000,
            value=config.generation.num_rows,
            step=100
        )
        config.generation.num_rows = num_rows

        seed = st.number_input(
            "Random seed (optional)",
            min_value=0,
            max_value=2**31-1,
            value=config.generation.seed if config.generation.seed else 42,
            help="Set seed for reproducibility"
        )
        config.generation.seed = seed

    with col2:
        batch_size = st.number_input(
            "Batch size",
            min_value=10,
            max_value=10000,
            value=config.generation.batch_size,
            step=10
        )
        config.generation.batch_size = batch_size

        enable_parallel = st.checkbox(
            "Enable parallel processing",
            value=config.generation.enable_parallel
        )
        config.generation.enable_parallel = enable_parallel

    # Numeric settings
    st.markdown('<p class="sub-header">Numeric Data Settings</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        preserve_correlations = st.checkbox(
            "Preserve correlations",
            value=config.numeric.preserve_correlations,
            help="Maintain correlations between numeric columns"
        )
        config.numeric.preserve_correlations = preserve_correlations

        distribution = st.selectbox(
            "Distribution fitting",
            ["auto", "normal", "uniform", "exponential", "lognormal", "gamma"],
            index=0
        )
        config.numeric.distribution_fitting = distribution

    with col2:
        range_enforcement = st.checkbox(
            "Range enforcement",
            value=config.numeric.range_enforcement,
            help="Keep values within original min/max"
        )
        config.numeric.range_enforcement = range_enforcement

        decimal_places = st.number_input(
            "Decimal places (optional)",
            min_value=0,
            max_value=10,
            value=config.numeric.decimal_places if config.numeric.decimal_places else 2,
        )
        config.numeric.decimal_places = decimal_places

    # Validation settings
    st.markdown('<p class="sub-header">Validation Settings</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        enable_quality = st.checkbox(
            "Enable quality checks",
            value=config.validation.enable_quality_checks
        )
        config.validation.enable_quality_checks = enable_quality

        quality_threshold = st.slider(
            "Quality threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.validation.quality_threshold,
            step=0.05
        )
        config.validation.quality_threshold = quality_threshold

    with col2:
        enable_privacy = st.checkbox(
            "Enable privacy checks",
            value=config.validation.enable_privacy_checks
        )
        config.validation.enable_privacy_checks = enable_privacy

        k_anonymity = st.number_input(
            "K-anonymity",
            min_value=2,
            max_value=100,
            value=config.validation.k_anonymity,
            help="Minimum group size for quasi-identifiers"
        )
        config.validation.k_anonymity = k_anonymity

    # Save configuration
    if st.button("💾 Save Configuration", use_container_width=True):
        st.session_state.config = config
        st.success("✅ Configuration saved!")


# Page: Generate
def page_generate():
    """Data generation page"""
    st.markdown('<p class="main-header">🎲 Generate Synthetic Data</p>',
                unsafe_allow_html=True)

    if st.session_state.reference_data is None:
        st.warning("⚠️ Please upload reference data first")
        return

    # Display configuration summary
    config = st.session_state.config

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows to Generate", f"{config.generation.num_rows:,}")
    col2.metric("Random Seed", config.generation.seed or "Random")
    col3.metric("Batch Size", config.generation.batch_size)
    col4.metric("Parallel", "✓" if config.generation.enable_parallel else "✗")

    # Generate button
    if st.button("🚀 Generate Synthetic Data", use_container_width=True, type="primary"):
        try:
            reference_data = st.session_state.reference_data

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize orchestrator
            status_text.text("Initializing orchestrator...")
            progress_bar.progress(10)

            orchestrator = DataOrchestrator(config)

            # Analyze schema
            status_text.text("Analyzing schema...")
            progress_bar.progress(20)

            schema = orchestrator.analyze_schema(reference_data)

            # Generate data
            status_text.text("Generating synthetic data...")
            progress_bar.progress(40)

            def progress_callback(current, total):
                progress = 40 + int((current / total) * 40)
                progress_bar.progress(progress)
                status_text.text(f"Generating: {current}/{total} rows...")

            orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
            orchestrator.register_pipeline(PipelineType.TEXT, TextGenerator(config))
            orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
            orchestrator.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(config))
            orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

            result = orchestrator.generate(
                num_rows=config.generation.num_rows,
                reference_data=reference_data,
                schema=schema,
                progress_callback=progress_callback,
            )
            synthetic_data = result.data

            progress_bar.progress(80)

            # Store results
            st.session_state.synthetic_data = synthetic_data
            st.session_state.schema = schema

            progress_bar.progress(100)
            status_text.text("✅ Generation complete!")

            st.markdown('<div class="success-box">✅ Synthetic data generated successfully!</div>',
                        unsafe_allow_html=True)

            # Display preview
            st.markdown(
                '<p class="sub-header">Synthetic Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(synthetic_data.head(10), use_container_width=True)

            # Statistics comparison
            st.markdown(
                '<p class="sub-header">Statistics Comparison</p>', unsafe_allow_html=True)

            numeric_cols = reference_data.select_dtypes(
                include=[np.number]).columns

            if len(numeric_cols) > 0:
                comparison_data = []

                for col in numeric_cols[:5]:  # Show first 5 numeric columns
                    comparison_data.append({
                        'Column': col,
                        'Reference Mean': f"{reference_data[col].mean():.2f}",
                        'Synthetic Mean': f"{synthetic_data[col].mean():.2f}",
                        'Reference Std': f"{reference_data[col].std():.2f}",
                        'Synthetic Std': f"{synthetic_data[col].std():.2f}",
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error during generation: {str(e)}</div>',
                        unsafe_allow_html=True)
            logger.error(f"Generation error: {e}", exc_info=True)


# Page: Validate
def page_validate():
    """Validation page"""
    st.markdown('<p class="main-header">✅ Validate Synthetic Data</p>',
                unsafe_allow_html=True)

    if st.session_state.synthetic_data is None:
        st.warning("⚠️ Please generate synthetic data first")
        return

    reference = st.session_state.reference_data
    synthetic = st.session_state.synthetic_data
    config = st.session_state.config

    # Validation options
    col1, col2 = st.columns(2)

    with col1:
        run_quality = st.checkbox("Run Quality Validation", value=True)
    with col2:
        run_privacy = st.checkbox("Run Privacy Validation", value=True)

    if st.button("🔍 Run Validation", use_container_width=True, type="primary"):
        # Quality validation
        if run_quality:
            st.markdown('<p class="sub-header">Quality Validation</p>',
                        unsafe_allow_html=True)

            with st.spinner("Running quality checks..."):
                validator = QualityValidator(config)
                quality_report = validator.validate(reference, synthetic)
                st.session_state.quality_report = quality_report

            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Score", f"{quality_report.overall_score:.3f}")
            col2.metric(
                "Status", "✅ Pass" if quality_report.passed else "❌ Fail")
            col3.metric("Metrics Passed",
                        f"{quality_report.summary['passed_metrics']}/{quality_report.summary['total_metrics']}")

            # Column scores
            st.markdown("**Column Scores:**")
            for col, score in quality_report.column_scores.items():
                st.progress(score, text=f"{col}: {score:.3f}")

            # Failed metrics
            failed = quality_report.get_failed_metrics()
            if failed:
                st.markdown("**Failed Metrics:**")
                for metric in failed:
                    st.error(
                        f"{metric.name}: {metric.value:.3f} (threshold: {metric.threshold})")

        # Privacy validation
        if run_privacy:
            st.markdown('<p class="sub-header">Privacy Validation</p>',
                        unsafe_allow_html=True)

            with st.spinner("Running privacy checks..."):
                validator = PrivacyValidator(config)
                privacy_report = validator.validate(synthetic, reference)
                st.session_state.privacy_report = privacy_report

            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Level", privacy_report.overall_risk.upper())
            col2.metric("K-Anonymity",
                        privacy_report.k_anonymity_score or "N/A")
            col3.metric(
                "Re-ID Risk", f"{privacy_report.reid_risk_score:.3f}" if privacy_report.reid_risk_score else "N/A")

            # Risk classification
            if privacy_report.overall_risk == 'low':
                st.success("✅ Low privacy risk")
            elif privacy_report.overall_risk == 'medium':
                st.warning("⚠️ Medium privacy risk")
            else:
                st.error("❌ High/Critical privacy risk")

            # Recommendations
            if privacy_report.recommendations:
                st.markdown("**Recommendations:**")
                for i, rec in enumerate(privacy_report.recommendations, 1):
                    st.info(f"{i}. {rec}")


# Page: Export
def page_export():
    """Export page"""
    st.markdown('<p class="main-header">💾 Export Synthetic Data</p>',
                unsafe_allow_html=True)

    if st.session_state.synthetic_data is None:
        st.warning("⚠️ No synthetic data to export")
        return

    synthetic = st.session_state.synthetic_data

    # Export format
    st.markdown('<p class="sub-header">Export Format</p>',
                unsafe_allow_html=True)

    export_format = st.selectbox(
        "Choose format",
        ["CSV", "Excel", "JSON", "Parquet"]
    )

    # Generate download
    if st.button("📥 Generate Download", use_container_width=True):
        try:
            if export_format == "CSV":
                buffer = io.StringIO()
                synthetic.to_csv(buffer, index=False)
                data = buffer.getvalue()
                mime = "text/csv"
                filename = "synthetic_data.csv"

            elif export_format == "Excel":
                buffer = io.BytesIO()
                synthetic.to_excel(buffer, index=False)
                data = buffer.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = "synthetic_data.xlsx"

            elif export_format == "JSON":
                data = synthetic.to_json(orient='records', indent=2)
                mime = "application/json"
                filename = "synthetic_data.json"

            elif export_format == "Parquet":
                buffer = io.BytesIO()
                synthetic.to_parquet(buffer)
                data = buffer.getvalue()
                mime = "application/octet-stream"
                filename = "synthetic_data.parquet"

            st.download_button(
                label=f"⬇️ Download {export_format}",
                data=data,
                file_name=filename,
                mime=mime,
                use_container_width=True
            )

            st.success(f"✅ {export_format} file ready for download!")

        except Exception as e:
            st.error(f"❌ Error generating download: {str(e)}")

    # Export reports
    st.markdown('<p class="sub-header">Export Reports</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.quality_report:
            report_json = json.dumps(
                st.session_state.quality_report.to_dict(),
                indent=2
            )
            st.download_button(
                label="📊 Download Quality Report",
                data=report_json,
                file_name="quality_report.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        if st.session_state.privacy_report:
            report_json = json.dumps(
                st.session_state.privacy_report.to_dict(),
                indent=2
            )
            st.download_button(
                label="🔒 Download Privacy Report",
                data=report_json,
                file_name="privacy_report.json",
                mime="application/json",
                use_container_width=True
            )


# Main app
def main():
    """Main application"""
    init_session_state()

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": page_home,
        "📤 Upload": page_upload,
        "⚙️ Configure": page_configure,
        "🎲 Generate": page_generate,
        "✅ Validate": page_validate,
        "💾 Export": page_export,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display selected page
    pages[selection]()

    # Sidebar info
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("### About")
    # st.sidebar.info("""
    # **Synthetic Data Generator**
    
    # Generate high-quality synthetic data with:
    # - Statistical preservation
    # - Privacy guarantees
    # - Multiple data types
    # - Validation metrics
    # """)

    # Session state info
    # if st.sidebar.checkbox("Show Session State"):
    #     st.sidebar.json({
    #         'reference_data': st.session_state.reference_data is not None,
    #         'synthetic_data': st.session_state.synthetic_data is not None,
    #         'quality_report': st.session_state.quality_report is not None,
    #         'privacy_report': st.session_state.privacy_report is not None,
    #     })


if __name__ == "__main__":
    main()
