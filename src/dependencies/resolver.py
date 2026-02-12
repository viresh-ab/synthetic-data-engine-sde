import pandas as pd
from .rules import DependencyRule


class DependencyResolver:

    def __init__(self, rules: list[DependencyRule]):
        self.rules = rules

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:

        for rule in self.rules:
            missing = [c for c in rule.sources if c not in df.columns]
            if missing:
                continue

            df[rule.target] = [
                rule.func(*vals)
                for vals in df[rule.sources].itertuples(index=False, name=None)
            ]

        return df