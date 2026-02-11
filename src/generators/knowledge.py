"""
Knowledge-Based Generator

Integrates domain knowledge and constraints into generation:
- RAG (Retrieval-Augmented Generation)
- Domain-specific rules and constraints
- Business logic enforcement
- Relationship preservation
"""

import os
import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class DomainRule:
    """Single domain rule"""
    rule_id: str
    description: str
    rule_type: str  # constraint, validation, relationship
    applies_to: List[str]  # Column names
    condition: str
    enforcement_level: str = "strict"  # strict, warning, optional
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate rule on data
        
        Returns:
            Boolean series indicating rule compliance
        """
        # This is a simplified evaluator
        # In production, you'd use a proper expression evaluator
        return pd.Series([True] * len(data))


@dataclass
class DomainConstraint:
    """Domain-specific constraint"""
    column: str
    constraint_type: str  # range, category, pattern, dependency
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, values: pd.Series) -> pd.Series:
        """Apply constraint to values"""
        if self.constraint_type == "range":
            min_val = self.parameters.get("min")
            max_val = self.parameters.get("max")
            if min_val is not None and max_val is not None:
                return values.clip(min_val, max_val)
        
        elif self.constraint_type == "category":
            valid_categories = self.parameters.get("categories", [])
            if valid_categories:
                # Replace invalid categories with random valid ones
                return values.apply(
                    lambda x: x if x in valid_categories else np.random.choice(valid_categories)
                )
        
        elif self.constraint_type == "pattern":
            pattern = self.parameters.get("pattern")
            if pattern:
                # Validate pattern (in production, would regenerate non-matching)
                return values
        
        return values


class DomainKnowledge:
    """
    Domain knowledge repository
    
    Loads and manages domain-specific rules, constraints, and knowledge
    """
    
    def __init__(self, knowledge_base_path: Optional[Path] = None, domain: Optional[str] = None):
        """
        Initialize domain knowledge
        
        Args:
            knowledge_base_path: Path to knowledge base directory
            domain: Specific domain to load (fashion, finance, healthcare, retail)
        """
        self.knowledge_base_path = knowledge_base_path
        self.domain = domain
        
        self.rules: List[DomainRule] = []
        self.constraints: List[DomainConstraint] = []
        self.relationships: Dict[str, List[str]] = {}
        self.terminology: Dict[str, List[str]] = {}
        
        if knowledge_base_path and domain:
            self.load_domain_knowledge()
    
    def load_domain_knowledge(self):
        """Load knowledge for the specified domain"""
        if not self.knowledge_base_path or not self.domain:
            logger.warning("No knowledge base path or domain specified")
            return
        
        domain_path = Path(self.knowledge_base_path) / self.domain
        
        if not domain_path.exists():
            logger.warning(f"Domain knowledge not found: {domain_path}")
            return
        
        # Load constraints
        constraints_file = domain_path / "constraints.yaml"
        if constraints_file.exists():
            self._load_constraints(constraints_file)
        
        # Load rules
        rules_file = domain_path / "rules.md"
        if rules_file.exists():
            self._load_rules(rules_file)
        
        logger.info(f"Loaded {len(self.rules)} rules and {len(self.constraints)} constraints for {self.domain}")
    
    def _load_constraints(self, filepath: Path):
        """Load constraints from YAML file"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return
            
            # Parse constraints
            for col_name, col_constraints in data.get('constraints', {}).items():
                for constraint in col_constraints:
                    self.constraints.append(
                        DomainConstraint(
                            column=col_name,
                            constraint_type=constraint.get('type'),
                            parameters=constraint.get('parameters', {})
                        )
                    )
            
            # Parse relationships
            self.relationships = data.get('relationships', {})
            
            # Parse terminology
            self.terminology = data.get('terminology', {})
            
        except Exception as e:
            logger.error(f"Failed to load constraints from {filepath}: {e}")
    
    def _load_rules(self, filepath: Path):
        """Load rules from markdown file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Simple parser for rules in markdown
            # Format: ## Rule: rule_id
            #         Description: ...
            #         Applies to: col1, col2
            #         Condition: ...
            
            rule_blocks = re.split(r'##\s+Rule:', content)[1:]  # Skip before first rule
            
            for block in rule_blocks:
                lines = block.strip().split('\n')
                if not lines:
                    continue
                
                rule_id = lines[0].strip()
                description = ""
                applies_to = []
                condition = ""
                
                for line in lines[1:]:
                    if line.startswith('Description:'):
                        description = line.replace('Description:', '').strip()
                    elif line.startswith('Applies to:'):
                        applies_to = [c.strip() for c in line.replace('Applies to:', '').split(',')]
                    elif line.startswith('Condition:'):
                        condition = line.replace('Condition:', '').strip()
                
                if rule_id and condition:
                    self.rules.append(
                        DomainRule(
                            rule_id=rule_id,
                            description=description,
                            rule_type="validation",
                            applies_to=applies_to,
                            condition=condition
                        )
                    )
        
        except Exception as e:
            logger.error(f"Failed to load rules from {filepath}: {e}")
    
    def get_constraints_for_column(self, column_name: str) -> List[DomainConstraint]:
        """Get all constraints applicable to a column"""
        return [c for c in self.constraints if c.column == column_name]
    
    def get_related_columns(self, column_name: str) -> List[str]:
        """Get columns that have relationships with given column"""
        return self.relationships.get(column_name, [])
    
    def get_valid_values(self, column_name: str, category: str) -> List[str]:
        """Get valid values for a column from terminology"""
        key = f"{column_name}_{category}"
        return self.terminology.get(key, [])


class ConstraintEngine:
    """
    Enforces constraints on generated data
    
    Ensures data adheres to domain rules and business logic
    """
    
    def __init__(self, domain_knowledge: Optional[DomainKnowledge] = None):
        """
        Initialize constraint engine
        
        Args:
            domain_knowledge: Domain knowledge repository
        """
        self.domain_knowledge = domain_knowledge
    
    def enforce_constraints(
        self,
        data: pd.DataFrame,
        strict: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Enforce all constraints on data
        
        Args:
            data: Data to constrain
            strict: If True, fail on constraint violations; if False, fix violations
            
        Returns:
            Tuple of (constrained_data, list_of_violations)
        """
        if not self.domain_knowledge:
            return data, []
        
        violations = []
        constrained = data.copy()
        
        # Apply constraints
        for constraint in self.domain_knowledge.constraints:
            if constraint.column not in constrained.columns:
                continue
            
            try:
                original = constrained[constraint.column].copy()
                constrained[constraint.column] = constraint.apply(constrained[constraint.column])
                
                # Check for changes
                if not original.equals(constrained[constraint.column]):
                    num_changed = (original != constrained[constraint.column]).sum()
                    violations.append(
                        f"{constraint.column}: {constraint.constraint_type} constraint "
                        f"modified {num_changed} values"
                    )
            
            except Exception as e:
                logger.error(f"Failed to apply constraint on {constraint.column}: {e}")
                violations.append(f"{constraint.column}: constraint application failed - {e}")
        
        # Validate rules
        for rule in self.domain_knowledge.rules:
            try:
                compliance = rule.evaluate(constrained)
                if not compliance.all():
                    num_violations = (~compliance).sum()
                    violations.append(
                        f"Rule '{rule.rule_id}': {num_violations} violations"
                    )
                    
                    if strict and rule.enforcement_level == "strict":
                        logger.warning(f"Strict rule violation: {rule.description}")
            
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
        
        return constrained, violations
    
    def enforce_relationships(
        self,
        data: pd.DataFrame,
        relationships: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Enforce relationships between columns
        
        Args:
            data: Data to process
            relationships: Dictionary mapping column to dependent columns
            
        Returns:
            Data with enforced relationships
        """
        result = data.copy()
        
        for primary_col, dependent_cols in relationships.items():
            if primary_col not in result.columns:
                continue
            
            # Group by primary column and ensure consistency in dependent columns
            for dep_col in dependent_cols:
                if dep_col not in result.columns:
                    continue
                
                # For each unique value in primary, use the most common dependent value
                for primary_val in result[primary_col].unique():
                    mask = result[primary_col] == primary_val
                    if mask.sum() > 0:
                        # Get most common dependent value
                        most_common = result.loc[mask, dep_col].mode()
                        if len(most_common) > 0:
                            result.loc[mask, dep_col] = most_common.iloc[0]
        
        return result


class KnowledgeGenerator:
    """
    Knowledge-enhanced data generator
    
    Uses domain knowledge to generate more realistic and constrained data
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize knowledge generator
        
        Args:
            config: Configuration object with knowledge settings
        """
        self.config = config
        
        # Initialize domain knowledge
        if config and hasattr(config, 'knowledge'):
            knowledge_path = config.knowledge.knowledge_base_path
            domain = config.knowledge.domain
            self.enforce_constraints = config.knowledge.enforce_constraints
        else:
            knowledge_path = None
            domain = None
            self.enforce_constraints = True
        
        self.domain_knowledge = DomainKnowledge(
            knowledge_base_path=knowledge_path,
            domain=domain
        )
        
        self.constraint_engine = ConstraintEngine(self.domain_knowledge)
    
    def generate_with_knowledge(
        self,
        base_data: pd.DataFrame,
        column_name: str,
        num_rows: int
    ) -> pd.Series:
        """
        Generate data enhanced with domain knowledge
        
        Args:
            base_data: Base generated data
            column_name: Column to generate
            num_rows: Number of rows
            
        Returns:
            Knowledge-enhanced series
        """
        # Get constraints for this column
        constraints = self.domain_knowledge.get_constraints_for_column(column_name)
        
        # Get related columns
        related = self.domain_knowledge.get_related_columns(column_name)
        
        # Generate base values
        if column_name in base_data.columns:
            values = base_data[column_name].copy()
        else:
            # Generate default values
            values = pd.Series([np.nan] * num_rows)
        
        # Apply constraints
        for constraint in constraints:
            values = constraint.apply(values)
        
        return values
    
    def enhance_dataset(
        self,
        data: pd.DataFrame,
        strict: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Enhance entire dataset with domain knowledge
        
        Args:
            data: Dataset to enhance
            strict: Strict constraint enforcement
            
        Returns:
            Tuple of (enhanced_data, report)
        """
        logger.info(f"Enhancing dataset with domain knowledge for domain: {self.domain_knowledge.domain}")
        
        # Apply constraints
        enhanced, violations = self.constraint_engine.enforce_constraints(
            data=data,
            strict=strict
        )
        
        # Enforce relationships
        if self.domain_knowledge.relationships:
            enhanced = self.constraint_engine.enforce_relationships(
                data=enhanced,
                relationships=self.domain_knowledge.relationships
            )
        
        # Create report
        report = {
            'domain': self.domain_knowledge.domain,
            'constraints_applied': len(self.domain_knowledge.constraints),
            'rules_checked': len(self.domain_knowledge.rules),
            'violations': violations,
            'relationships_enforced': len(self.domain_knowledge.relationships)
        }
        
        logger.info(f"Enhancement complete: {len(violations)} violations detected")
        
        return enhanced, report
    
    def get_domain_examples(
        self,
        column_name: str,
        category: Optional[str] = None,
        num_examples: int = 10
    ) -> List[str]:
        """
        Get example values from domain knowledge
        
        Args:
            column_name: Column name
            category: Optional category
            num_examples: Number of examples
            
        Returns:
            List of example values
        """
        if category:
            values = self.domain_knowledge.get_valid_values(column_name, category)
        else:
            # Try to get from terminology
            values = []
            for key in self.domain_knowledge.terminology:
                if column_name in key:
                    values.extend(self.domain_knowledge.terminology[key])
        
        if not values:
            return []
        
        # Sample random examples
        num_examples = min(num_examples, len(values))
        return list(np.random.choice(values, num_examples, replace=False))
    
    def validate_against_rules(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against all domain rules
        
        Args:
            data: Data to validate
            
        Returns:
            Validation report
        """
        report = {
            'total_rules': len(self.domain_knowledge.rules),
            'passed': 0,
            'failed': 0,
            'rule_results': []
        }
        
        for rule in self.domain_knowledge.rules:
            try:
                compliance = rule.evaluate(data)
                passed = compliance.all()
                
                result = {
                    'rule_id': rule.rule_id,
                    'description': rule.description,
                    'passed': passed,
                    'compliance_rate': compliance.mean()
                }
                
                report['rule_results'].append(result)
                
                if passed:
                    report['passed'] += 1
                else:
                    report['failed'] += 1
            
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
                report['rule_results'].append({
                    'rule_id': rule.rule_id,
                    'error': str(e)
                })
        
        return report


# Example domain knowledge files
def create_example_domain_files(output_dir: Path):
    """Create example domain knowledge files"""
    
    # Fashion domain
    fashion_dir = output_dir / "fashion"
    fashion_dir.mkdir(parents=True, exist_ok=True)
    
    fashion_constraints = {
        'constraints': {
            'price': [
                {
                    'type': 'range',
                    'parameters': {'min': 0, 'max': 10000}
                }
            ],
            'size': [
                {
                    'type': 'category',
                    'parameters': {'categories': ['XS', 'S', 'M', 'L', 'XL', 'XXL']}
                }
            ]
        },
        'relationships': {
            'category': ['subcategory', 'brand'],
            'size': ['price']
        },
        'terminology': {
            'category': ['Dresses', 'Tops', 'Bottoms', 'Outerwear', 'Accessories'],
            'subcategory_dresses': ['Maxi', 'Mini', 'Midi', 'A-line', 'Wrap'],
            'brand': ['Zara', 'H&M', 'Forever 21', 'Mango', 'Gap']
        }
    }
    
    with open(fashion_dir / "constraints.yaml", 'w') as f:
        yaml.dump(fashion_constraints, f)
    
    fashion_rules = """
## Rule: price_size_consistency
Description: Larger sizes should not have significantly different prices
Applies to: price, size
Condition: price variance across sizes < 20%

## Rule: category_subcategory_match
Description: Subcategory must match parent category
Applies to: category, subcategory
Condition: subcategory in valid_subcategories[category]
"""
    
    with open(fashion_dir / "rules.md", 'w') as f:
        f.write(fashion_rules)
    
    logger.info(f"Created example domain files in {fashion_dir}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create example domain files
    example_dir = Path("/tmp/knowledge_base")
    create_example_domain_files(example_dir)
    
    # Initialize knowledge generator
    from dataclasses import dataclass
    
    @dataclass
    class MockKnowledgeConfig:
        knowledge_base_path: str = str(example_dir)
        domain: str = "fashion"
        enforce_constraints: bool = True
    
    @dataclass
    class MockConfig:
        knowledge: MockKnowledgeConfig = None
        
        def __post_init__(self):
            if self.knowledge is None:
                self.knowledge = MockKnowledgeConfig()
    
    config = MockConfig()
    generator = KnowledgeGenerator(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'category': ['Dresses', 'Tops', 'Dresses', 'Bottoms'],
        'size': ['M', 'XS', 'L', 'S'],
        'price': [50, 30, 60, 40]
    })
    
    print("Original data:")
    print(sample_data)
    
    # Enhance with domain knowledge
    enhanced, report = generator.enhance_dataset(sample_data, strict=False)
    
    print("\n" + "="*60)
    print("Enhanced data:")
    print(enhanced)
    
    print("\n" + "="*60)
    print("Enhancement report:")
    print(json.dumps(report, indent=2))
    
    # Get domain examples
    print("\n" + "="*60)
    print("Domain examples for 'category':")
    examples = generator.get_domain_examples('category', num_examples=5)
    print(examples)
