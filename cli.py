"""
Command-Line Interface for Synthetic Data Generator

Provides commands for:
- generate: Generate synthetic data
- validate: Validate synthetic data
- analyze: Analyze data schema
- config: Manage configurations
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich import print as rprint
import json

# Import our modules
from src.config import Config, ConfigLoader, get_default_config
from src.orchestrator import DataOrchestrator, PipelineType
from src.schema import SchemaAnalyzer
from src.validation import QualityValidator, PrivacyValidator
from src.generators import NumericGenerator, TextGenerator, PIIGenerator, TemporalGenerator, CategoricalGenerator
from src.utils import FileHandler, setup_logging, read_data, write_data

# Setup console
console = Console()


class CLI:
    """Main CLI class"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.config_loader = ConfigLoader()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Synthetic Data Generator CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate synthetic data
  python cli.py generate input.csv output.csv --rows 1000
  
  # Validate synthetic data
  python cli.py validate reference.csv synthetic.csv --quality --privacy
  
  # Analyze data schema
  python cli.py analyze data.csv --output schema.json
  
  # Use preset configuration
  python cli.py generate input.csv output.csv --preset analytics
            """
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Generate command
        generate_parser = subparsers.add_parser('generate', help='Generate synthetic data')
        generate_parser.add_argument('input', help='Input reference data file')
        generate_parser.add_argument('output', help='Output synthetic data file')
        generate_parser.add_argument('--rows', '-n', type=int, default=1000, help='Number of rows to generate')
        generate_parser.add_argument('--seed', '-s', type=int, help='Random seed for reproducibility')
        generate_parser.add_argument('--preset', '-p', help='Configuration preset')
        generate_parser.add_argument('--config', '-c', help='Custom configuration file')
        generate_parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size')
        generate_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate synthetic data')
        validate_parser.add_argument('reference', help='Reference data file')
        validate_parser.add_argument('synthetic', help='Synthetic data file')
        validate_parser.add_argument('--quality', action='store_true', help='Run quality validation')
        validate_parser.add_argument('--privacy', action='store_true', help='Run privacy validation')
        validate_parser.add_argument('--threshold', '-t', type=float, default=0.8, help='Quality threshold')
        validate_parser.add_argument('--k-anonymity', '-k', type=int, default=5, help='K-anonymity requirement')
        validate_parser.add_argument('--output', '-o', help='Output report file (JSON)')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze data schema')
        analyze_parser.add_argument('input', help='Input data file')
        analyze_parser.add_argument('--output', '-o', help='Output analysis file (JSON)')
        analyze_parser.add_argument('--detailed', action='store_true', help='Detailed analysis with PII detection')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Manage configurations')
        config_subparsers = config_parser.add_subparsers(dest='config_command')
        
        # Config list
        config_subparsers.add_parser('list', help='List available presets')
        
        # Config show
        show_parser = config_subparsers.add_parser('show', help='Show preset configuration')
        show_parser.add_argument('preset', help='Preset name')
        
        # Config create
        create_parser = config_subparsers.add_parser('create', help='Create custom configuration')
        create_parser.add_argument('output', help='Output configuration file')
        
        return parser
    
    def run(self, args=None):
        """Run CLI"""
        args = self.parser.parse_args(args)
        
        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=log_level)
        
        # Execute command
        if args.command == 'generate':
            self.cmd_generate(args)
        elif args.command == 'validate':
            self.cmd_validate(args)
        elif args.command == 'analyze':
            self.cmd_analyze(args)
        elif args.command == 'config':
            self.cmd_config(args)
        else:
            self.parser.print_help()
    
    def cmd_generate(self, args):
        """Generate synthetic data"""
        console.print(Panel.fit(
            "🎲 [bold]Synthetic Data Generation[/bold]",
            border_style="blue"
        ))
        
        try:
            # Load configuration
            if args.config:
                config = self.config_loader.load_from_file(args.config)
                console.print(f"✓ Loaded custom configuration: {args.config}")
            elif args.preset:
                config = self.config_loader.load_preset(args.preset)
                console.print(f"✓ Loaded preset: {args.preset}")
            else:
                config = get_default_config()
                console.print("✓ Using default configuration")
            
            # Override with command-line arguments
            config.generation.num_rows = args.rows
            if args.seed:
                config.generation.seed = args.seed
            config.generation.batch_size = args.batch_size
            config.generation.enable_parallel = not args.no_parallel
            
            # Load reference data
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading reference data...", total=None)
                reference_data = read_data(args.input)
                progress.update(task, completed=True)
            
            console.print(f"✓ Loaded {len(reference_data):,} rows, {len(reference_data.columns)} columns")
            
            # Initialize orchestrator
            orchestrator = DataOrchestrator(config)
            
            # Analyze schema
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing schema...", total=None)
                schema = orchestrator.analyze_schema(reference_data)
                progress.update(task, completed=True)
            
            console.print(f"✓ Schema analysis complete")
            
            # Generate synthetic data
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Generating synthetic data...", total=100)
                
                def progress_callback(current, total):
                    percentage = (current / total) * 100
                    progress.update(task, completed=percentage)
                
                orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
                orchestrator.register_pipeline(PipelineType.TEXT, TextGenerator(config))
                orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
                orchestrator.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(config))
                orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

                result = orchestrator.generate(
                    num_rows=args.rows,
                    reference_data=reference_data,
                    schema=schema,
                    progress_callback=progress_callback,
                )
                synthetic_data = result.data

                progress.update(task, completed=100)
            
            # Save synthetic data
            write_data(synthetic_data, args.output)
            console.print(f"✓ Saved synthetic data to: {args.output}")
            
            # Summary
            table = Table(title="Generation Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Rows Generated", f"{len(synthetic_data):,}")
            table.add_row("Columns", str(len(synthetic_data.columns)))
            table.add_row("Seed", str(config.generation.seed or "Random"))
            table.add_row("Output File", args.output)
            
            console.print(table)
            console.print("\n[bold green]✓ Generation complete![/bold green]")
        
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
            if args.verbose:
                console.print_exception()
            sys.exit(1)
    
    def cmd_validate(self, args):
        """Validate synthetic data"""
        console.print(Panel.fit(
            "✅ [bold]Data Validation[/bold]",
            border_style="green"
        ))
        
        try:
            # Load data
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task1 = progress.add_task("Loading reference data...", total=None)
                reference = read_data(args.reference)
                progress.update(task1, completed=True)
                
                task2 = progress.add_task("Loading synthetic data...", total=None)
                synthetic = read_data(args.synthetic)
                progress.update(task2, completed=True)
            
            console.print(f"✓ Loaded reference: {len(reference):,} rows")
            console.print(f"✓ Loaded synthetic: {len(synthetic):,} rows")
            
            # Create config
            config = get_default_config()
            config.validation.quality_threshold = args.threshold
            config.validation.k_anonymity = args.k_anonymity
            
            results = {}
            
            # Quality validation
            if args.quality or (not args.quality and not args.privacy):
                console.print("\n[bold]Running quality validation...[/bold]")
                
                validator = QualityValidator(config)
                quality_report = validator.validate(reference, synthetic)
                
                results['quality'] = quality_report.to_dict()
                
                # Display results
                table = Table(title="Quality Metrics", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                table.add_column("Status", style="green")
                
                table.add_row(
                    "Overall Score",
                    f"{quality_report.overall_score:.3f}",
                    "✓ Pass" if quality_report.passed else "✗ Fail"
                )
                table.add_row(
                    "Metrics Passed",
                    f"{quality_report.summary['passed_metrics']}/{quality_report.summary['total_metrics']}",
                    ""
                )
                
                console.print(table)
                
                # Column scores
                if quality_report.column_scores:
                    console.print("\n[bold]Column Scores:[/bold]")
                    for col, score in quality_report.column_scores.items():
                        status = "✓" if score >= args.threshold else "✗"
                        console.print(f"  {status} {col}: {score:.3f}")
            
            # Privacy validation
            if args.privacy or (not args.quality and not args.privacy):
                console.print("\n[bold]Running privacy validation...[/bold]")
                
                validator = PrivacyValidator(config)
                privacy_report = validator.validate(synthetic, reference)
                
                results['privacy'] = privacy_report.to_dict()
                
                # Display results
                table = Table(title="Privacy Metrics", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                table.add_column("Status", style="green")
                
                risk_color = {
                    'low': 'green',
                    'medium': 'yellow',
                    'high': 'red',
                    'critical': 'bold red'
                }[privacy_report.overall_risk]
                
                table.add_row(
                    "Risk Level",
                    privacy_report.overall_risk.upper(),
                    f"[{risk_color}]{'✓ Pass' if privacy_report.passed else '✗ Fail'}[/{risk_color}]"
                )
                
                if privacy_report.k_anonymity_score:
                    table.add_row(
                        "K-Anonymity",
                        str(privacy_report.k_anonymity_score),
                        "✓" if privacy_report.k_anonymity_score >= args.k_anonymity else "✗"
                    )
                
                if privacy_report.reid_risk_score:
                    table.add_row(
                        "Re-ID Risk",
                        f"{privacy_report.reid_risk_score:.3f}",
                        ""
                    )
                
                console.print(table)
                
                # Recommendations
                if privacy_report.recommendations:
                    console.print("\n[bold]Recommendations:[/bold]")
                    for i, rec in enumerate(privacy_report.recommendations, 1):
                        console.print(f"  {i}. {rec}")
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"\n✓ Results saved to: {args.output}")
            
            console.print("\n[bold green]✓ Validation complete![/bold green]")
        
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
            if args.verbose:
                console.print_exception()
            sys.exit(1)
    
    def cmd_analyze(self, args):
        """Analyze data schema"""
        console.print(Panel.fit(
            "🔍 [bold]Schema Analysis[/bold]",
            border_style="cyan"
        ))
        
        try:
            # Load data
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading data...", total=None)
                data = read_data(args.input)
                progress.update(task, completed=True)
            
            console.print(f"✓ Loaded {len(data):,} rows, {len(data.columns)} columns")
            
            # Analyze schema
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing schema...", total=None)
                analyzer = SchemaAnalyzer(
                    enable_pii_detection=args.detailed,
                    enable_statistical_profiling=args.detailed
                )
                profiles = analyzer.analyze_dataframe(data)
                progress.update(task, completed=True)
            
            # Display results
            table = Table(title="Column Analysis", show_header=True)
            table.add_column("Column", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Completeness", style="green")
            table.add_column("Uniqueness", style="blue")
            
            if args.detailed:
                table.add_column("PII", style="red")
            
            results = []
            
            for col_name, profile in profiles.items():
                row = [
                    col_name,
                    profile.inferred_type,
                    f"{profile.completeness*100:.1f}%",
                    f"{profile.uniqueness*100:.1f}%"
                ]
                
                if args.detailed:
                    pii = "✓" if profile.contains_pii else ""
                    row.append(pii)
                
                table.add_row(*row)
                
                # Collect for JSON output
                result = {
                    'column': col_name,
                    'type': profile.inferred_type,
                    'completeness': profile.completeness,
                    'uniqueness': profile.uniqueness,
                    'contains_pii': profile.contains_pii
                }
                
                if args.detailed and profile.inferred_type == 'numeric':
                    result.update({
                        'min': profile.min_value,
                        'max': profile.max_value,
                        'mean': profile.mean,
                        'std': profile.std
                    })
                
                results.append(result)
            
            console.print(table)
            
            # Save results
            if args.output:
                output_data = {
                    'num_rows': len(data),
                    'num_columns': len(data.columns),
                    'memory_mb': data.memory_usage(deep=True).sum() / 1024**2,
                    'columns': results
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                console.print(f"\n✓ Analysis saved to: {args.output}")
            
            console.print("\n[bold green]✓ Analysis complete![/bold green]")
        
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
            if args.verbose:
                console.print_exception()
            sys.exit(1)
    
    def cmd_config(self, args):
        """Manage configurations"""
        console.print(Panel.fit(
            "⚙️ [bold]Configuration Management[/bold]",
            border_style="magenta"
        ))
        
        try:
            if args.config_command == 'list':
                presets = self.config_loader.list_presets()
                
                table = Table(title="Available Presets", show_header=True)
                table.add_column("Preset", style="cyan")
                table.add_column("Description", style="white")
                
                descriptions = {
                    'default': 'Default configuration',
                    'analytics': 'Analytics-focused with correlation preservation',
                    'survey': 'Survey data with template-based text',
                    'healthcare': 'Healthcare with full anonymization',
                    'finance': 'Finance with precision and constraints'
                }
                
                for preset in presets:
                    desc = descriptions.get(preset, 'Custom preset')
                    table.add_row(preset, desc)
                
                console.print(table)
            
            elif args.config_command == 'show':
                config = self.config_loader.load_preset(args.preset)
                
                console.print(f"\n[bold]Preset: {args.preset}[/bold]\n")
                console.print_json(data=config.to_dict())
            
            elif args.config_command == 'create':
                config = get_default_config()
                self.config_loader.save_config(config, args.output)
                
                console.print(f"✓ Created configuration file: {args.output}")
                console.print("  Edit this file to customize settings")
            
            else:
                console.print("Use 'config list', 'config show <preset>', or 'config create <file>'")
        
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
            if args.verbose:
                console.print_exception()
            sys.exit(1)


def main():
    """CLI entry point"""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
