# src/dependencies/registry.py

from src.dependencies.rules import DependencyRule
from src.generators.pii import email_from_name


DEPENDENCY_RULES = [

    # -------------------------
    # Email depends on name
    # -------------------------
    DependencyRule(
        target="email",
        sources=["first_name", "last_name"],
        func=email_from_name
    ),

]