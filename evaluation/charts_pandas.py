import os
import json
from collections.abc import Sequence, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALGORITHM_KEY = "algorithm_name"
ORACLE_KEY = "oracle"

ORACLE_HATCHES = {
    "None": "",    # No hatch
    "Random": "...", # Dots
    # "oracle_3": "xx",  # Crosshatch
    # "oracle_4": "o",   # Circles
    # "oracle_5": "*",   # Stars
    # "oracle_6": "/",   # Diagonal
    # "oracle_7": "\\",  # Opposite diagonal
    # "oracle_8": "|",   # Vertical
    # "oracle_9": "-",   # Horizontal
    # "oracle_10": "+",  # Plus
}

def default_group_by_formatter(column_name, column_value):
    if column_name == "algorithm_name":
        return column_value

    if column_name == "oracle":
        if column_value == "None":
            return ""
        return f"<{column_value}>"

    if column_name == "glitch_percent":
        return f"{column_value}%"

    if column_name == "original_automaton":
        return Path(column_value).stem

    return f"{column_name}: {column_value}"


def is_valid_result(result: pd.Series) -> bool:
    return pd.isna(result.get("exception", np.nan))


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON results into a Pandas DataFrame."""
    data = []
    for filename in os.listdir(results_dir):
        if filename.startswith('info') or not filename.endswith(".json"):
            continue
        with open(os.path.join(results_dir, filename), "r") as f:
            result = json.load(f)
            result["results_file"] = filename
            data.append(result)
    return pd.DataFrame(data)


def bar_chart(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
              only_if=None, title: str = None, group_by: Sequence[str] = None,
              group_by_formatter: Callable[[str, Any], str] = default_group_by_formatter,
              custom_axes_and_legend: bool = False):
    """Creates a bar chart of the results given in df, using the aggregated value of the @key column as y-axis."""

    if group_by is None:
        group_by = []
    else:
        group_by = list(*group_by)

    if only_if:
        df = df[df.apply(only_if, axis=1)]

    pivot_df = df.groupby(by=group_by)[key].agg(agg_method).unstack()
    pivot_df = pivot_df.astype(float)  # ensure numeric data

    ax = pivot_df.plot(kind="bar", figsize=(12, 6), width=0.8)

    # show value labels on bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f")

    # for bars, col in zip(ax.containers, pivot_df.columns):
    #     hatch = ""
    #     for c in col_names:
    #         if c in ORACLE_HATCHES:
    #             hatch = ORACLE_HATCHES[c]
    #     bar.set_hatch(hatch)

    # format x ticks
    formatted_xticks = [
        " ".join(group_by_formatter(col, val) for col, val in zip(group_by, index))
        for index, _ in pivot_df.iterrows()
    ]
    ax.set_xticklabels(formatted_xticks)

    # format legend labels
    if custom_axes_and_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [group_by_formatter(group_by[-1], label) for label in labels], title=group_by[-1])
        plt.xlabel("/".join(group_by[:-1]))

    full_groupby_name = ", ".join(group_by[:-1]) + " and " + group_by[-1]
    plt.title(title or f"{key} for each {full_groupby_name} ({agg_method} over results)")
    plt.ylabel(key)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def bar_chart_per_algorithm(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
                            only_if=None, title: str = None, group_by: Sequence[str] = None):
    group_by = list(group_by or []).insert(0, ALGORITHM_KEY)
    bar_chart(df, key, agg_method, only_if, title, group_by)


def bar_chart_per_algorithm_and_oracle(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
                            only_if=None, title: str = None, group_by: Sequence[str] = None):
    group_by = [ALGORITHM_KEY, ORACLE_KEY] + list(group_by or [])
    bar_chart(df, key, agg_method, only_if, title, group_by)