import pandas as pd


def areTablesMergeable(a: pd.DataFrame, b: pd.DataFrame, merge_column_name: str) -> bool:
    """
    Checks if the two DataFrames have intersecting values in the specified column
    :param a: DataFrame to check
    :param b: DataFrame to check
    :param merge_column_name: the column to compare
    :return:
    """
    setA = set(a[merge_column_name])
    setB = set(b[merge_column_name])
    return len(setA & setB) == 0
