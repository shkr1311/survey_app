import pandas as pd
import numpy as np
from scipy import stats

def log_step(logs, step):
    logs.append(step)

def auto_clean(df):
    logs = []
    removed_rows = pd.DataFrame()
    original_shape = df.shape

    # Detect column types
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = 'datetime'
        else:
            types[col] = 'categorical'
    log_step(logs, f"Detected column types: {types}")

    for col in df.columns:
        coltype = types[col]
        null_pct = df[col].isnull().mean()
        if coltype == 'numeric':
            if null_pct < 0.1:
                df[col].fillna(df[col].mean(), inplace=True)
                log_step(logs, f"{col}: <10% null, filled with mean")
            elif null_pct < 0.3:
                df[col].fillna(df[col].median(), inplace=True)
                log_step(logs, f"{col}: 10–30% null, filled with median")
            else:
                df.drop(columns=[col], inplace=True)
                log_step(logs, f"{col}: >30% null, column dropped")
        elif coltype == 'categorical':
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)
                log_step(logs, f"{col}: categorical, filled nulls with mode")
            else:
                df[col].fillna('Unknown', inplace=True)
        # Add datetime handling if desired

    # Remove outliers (IQR method) for numeric columns
    for col, coltype in types.items():
        if coltype == 'numeric' and col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            mask = (df[col] < low) | (df[col] > high)
            removed = df[mask]
            if not removed.empty:
                removed_rows = pd.concat([removed_rows, removed])
                df = df[~mask]
                log_step(logs, f"{col}: removed {mask.sum()} outliers using IQR")

    return df.reset_index(drop=True), logs, removed_rows, types

def rule_clean(df, rules):
    logs = []
    removed_rows = pd.DataFrame()
    keep_mask = pd.Series([True] * len(df), index=df.index)

    for col, (rule, val) in rules.items():
        if col not in df.columns:
            logs.append(f"Rule on {col}: column not found, skipped")
            continue
        if rule == '>':
            mask = df[col] > val
        elif rule == '<':
            mask = df[col] < val
        elif rule == 'in':
            mask = df[col].isin(val)
        elif rule == 'not in':
            mask = ~df[col].isin(val)
        else:
            logs.append(f"Rule on {col}: unknown opcode '{rule}', skipped")
            mask = True

        bad_mask = ~mask
        if bad_mask.sum() > 0:
            logs.append(f"{col}: removed {bad_mask.sum()} rows by rule {rule} {val}")
            removed_rows = pd.concat([removed_rows, df[bad_mask]])
        keep_mask = keep_mask & mask

    cleaned = df[keep_mask]
    return cleaned.reset_index(drop=True), logs, removed_rows
