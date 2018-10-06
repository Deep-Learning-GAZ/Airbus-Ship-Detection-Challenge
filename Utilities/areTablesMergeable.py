import pandas as pd


def areTablesMergeable(a: pd.DataFrame, b: pd.DataFrame, merge_column_name: str) -> bool:
    setA = set(a[merge_column_name])
    setB = set(b[merge_column_name])
    return len(setA & setB) == 0
