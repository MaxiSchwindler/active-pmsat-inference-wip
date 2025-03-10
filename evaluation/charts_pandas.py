import os
import json
from collections.abc import Sequence, Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALGORITHM_KEY = "algorithm_name"
ORACLE_KEY = "oracle"

ORACLE_HATCHES = {
    "None": "",
    "Random": "...",
    # "oracle_3": "xx",
    # "oracle_4": "o",
    # "oracle_5": "*",
    # "oracle_6": "/",
    # "oracle_7": "\\",
    # "oracle_8": "|",
    # "oracle_9": "-",
    # "oracle_10": "+",
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


def load_gsm_comparison_data(results_dir, results: pd.DataFrame = None):
    """ Load GSM comparison data from the "GSM_comparison" folder inside the given results_dir.
    If a `results` dataframe is given, the comparison data is added there; otherwise, a new
    dataframe containing only the comparison data is constructed and returned."""
    gsm_comparison_folder = Path(results_dir) / "GSM_comparison"
    if not gsm_comparison_folder.exists():
        print(
            f"No GSM comparison folder {gsm_comparison_folder} found.\nCreate it by running evaluation/compare_with_gsm.py {results_dir}")
        return

    data = []
    for file in gsm_comparison_folder.iterdir():
        if file.name.startswith('info') or not file.name.endswith(".json"):
            continue
        with open(file.resolve(), "r") as f:
            result = json.load(f)
            result["comparison_results_file"] = file.name
            data.append(result)

    comparison_results = pd.json_normalize(data)
    comparison_results["model_name"] = comparison_results["original_automaton"].apply(lambda x: Path(x).stem)
    comparison_results["apmsl_variant"] = comparison_results.apply(
        lambda row: f"{row['algorithm_name']} ({row['oracle']})", axis=1
    )

    if results is None:
        # no other df given -> return new df
        return comparison_results

    if len(results) != len(comparison_results):
        print(f"Different number of rows in results ({len(results)}) and in comparison results ({len(comparison_results)})")

    # given a results df in which to write comparison data
    results_file_counts = results['results_file'].value_counts()
    for stat in ("bisimilar", "Precision", "Recall", "F-Score", "total_time"):  #  "learning_rounds"
        for gsm_alg in ("GSM_with_purge_mismatches", "GSM_without_purge_mismatches"):
            col_name = f"{gsm_alg}.{stat}"

            results[col_name] = np.nan
            results[col_name] = results[col_name].astype(object)

            for index, row in comparison_results.iterrows():
                assert results_file_counts.get(row['results_file'], 0) == 1
                assert results[results['results_file'] == row['results_file']][stat].values[0] == row[
                    f"apmsl_algorithm.{stat}"]
                assert results[results['results_file'] == row['results_file']]["original_automaton"].values[0] == row[
                    "original_automaton"]

                results.loc[
                    results['results_file'] == row['results_file'],
                    col_name
                ] = row[col_name]

            assert results[col_name].isna().sum() == 0, f"Not all rows got a value for {col_name}"

    print("Added GSM comparison data")


def bar_chart(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
              only_if=None, title: str = None, group_by: Sequence[str] = None,
              group_by_formatter: Callable[[str, Any], str] = default_group_by_formatter,
              custom_axes_and_legend: bool = False):
    """Creates a bar chart of the results given in df, using the aggregated value of the @key column as y-axis."""

    if group_by is None:
        group_by = []
    else:
        group_by = list(g for g in group_by)

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


def multiple_bar_charts(df: pd.DataFrame, keys: Sequence[str], agg_method: str | Callable = "mean",
              only_if=None, title: str = None, group_by: Sequence[str] = None,
              group_by_formatter: Callable[[str, Any], str] = default_group_by_formatter,
              custom_axes_and_legend: bool = False, positioning_mode: Literal["beside", "below", "twocols"] = "below",
              figsize: tuple[int, int] = (12, 6)):
    """Creates bar charts for the given keys, one chart per key, all grouped by the same columns."""

    if group_by is None:
        group_by = []
    else:
        group_by = list(g for g in group_by)

    if only_if:
        df = df[df.apply(only_if, axis=1)]

    n_keys = len(keys)
    plot_kwargs = {}
    if positioning_mode == "twocols":
        # Calculate the number of subplots needed based on the number of keys
        n_cols = 2  # Number of columns for subplots (you can adjust this as needed)
        n_rows = (n_keys + n_cols - 1) // n_cols  # Ceiling division to determine number of rows
    elif positioning_mode == "below":
        n_cols = 1  # Set to 1 for a single column layout
        n_rows = n_keys  # Set the number of rows to match the number of keys
        plot_kwargs["sharex"] = True
    elif positioning_mode == "beside":
        n_cols = n_keys  # Set the number of columns to match the number of keys
        n_rows = 1  # Set to 1 for a single row layout
        plot_kwargs["sharey"] = True

    # Create subplots with enough rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), **plot_kwargs)
    axes = axes.flatten()  # Flatten the axes array to make it easier to index

    # Format the group_by labels for x-ticks
    pivot_df_sample = df.groupby(by=group_by).size()  # Just to extract the group_by levels

    if False:
        formatted_xticks = [
            " ".join(group_by_formatter(col, val) for col, val in zip(group_by, index))
            for index in pivot_df_sample.index
        ]

    # Iterate over the keys and plot them on separate subplots
    for i, key in enumerate(keys):
        ax = axes[i]

        # Create the pivot table for the current key
        pivot_df = df.groupby(by=group_by)[key].agg(agg_method).unstack()
        pivot_df = pivot_df.astype(float)  # Ensure numeric data

        # Plot the data
        pivot_df.plot(kind="bar", ax=ax, width=0.8)

        # Add value labels on bars
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.2f")

        # Set the x-tick labels
        if False:
            ax.set_xticklabels(formatted_xticks, rotation=45, ha='right')

        # Set the title and labels for the current chart
        full_groupby_name = ", ".join(group_by[:-1]) + " and " + group_by[-1]
        ax.set_title(f"{key} for each {full_groupby_name} ({agg_method} over results)")
        ax.set_ylabel(key)

        # Format the legend for the current chart
        if custom_axes_and_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [group_by_formatter(group_by[-1], label) for label in labels], title=group_by[-1])
            ax.set_xlabel("/".join(group_by[:-1]))

    # Hide any unused subplots (if the number of keys is less than the total subplot space)
    for j in range(n_keys, len(axes)):
        axes[j].axis("off")

    # Adjust layout to ensure it doesn't overlap
    plt.tight_layout()
    plt.show()


def bar_chart_per_algorithm(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
                            only_if=None, title: str = None, group_by: Sequence[str] = None):
    group_by = list(g for g in group_by).insert(0, ALGORITHM_KEY)
    bar_chart(df, key, agg_method, only_if, title, group_by)


def bar_chart_per_algorithm_and_oracle(df: pd.DataFrame, key: str, agg_method: str | Callable = "mean",
                            only_if=None, title: str = None, group_by: Sequence[str] = None):
    group_by = [ALGORITHM_KEY, ORACLE_KEY] + list(g for g in group_by)
    bar_chart(df, key, agg_method, only_if, title, group_by)


def multiple_bar_charts_per_algorithm_and_oracle(df: pd.DataFrame, keys: Sequence[str], group_by: Sequence[str] = None, **kwargs):
    group_by = [ALGORITHM_KEY, ORACLE_KEY] + list(g for g in group_by)
    multiple_bar_charts(df, keys, group_by=group_by, **kwargs)