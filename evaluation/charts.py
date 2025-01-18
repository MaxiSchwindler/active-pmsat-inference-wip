import itertools
import json
import os
from collections import defaultdict
from collections.abc import Sequence, Callable
from functools import cmp_to_key
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from active_pmsatlearn.learnalgo import get_total_num_additional_traces

RAISE = object()

Key: TypeAlias = str | Sequence[str] | Callable[[dict], float]


def get_algorithm_names(results: list[dict]) -> list[str]:
    """ All algorithm names in the list of results"""
    return sorted(list(set([entry["algorithm_name"] for entry in results])))


def get_oracle_names(results: list[dict]) -> list[str]:
    """ All oracle names in the list of results"""
    return sorted(list(set([entry["oracle"] for entry in results])))


def is_valid_result(result: dict) -> bool:
    return "exception" not in result


def load_results(results_dir: str) -> list[dict]:
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith('info') or not filename.endswith(".json"):
            continue
        with open(os.path.join(results_dir, filename), "r") as f:
            results.append(json.load(f))
    return results


def get_val(result: dict, key: Key, *, default=RAISE):
    """
    Get the value of @key in the given result.
    Supports nested keys, e.g. ("last_pmsat_info", "num_vars")
    """
    assert isinstance(key, Sequence) or isinstance(key, str) or callable(key)

    try:
        if callable(key):
            return key(result)
        elif isinstance(key, str):
            return result[key]
        elif len(key) == 1:
            return result[key[0]]
        elif isinstance(key, (list, tuple)):
            return get_val(result[key[0]], key[1:])
    except KeyError:
        if default is not RAISE:
            return default
        raise


def get_pretty_key(key: Key):
    if isinstance(key, str):
        return key
    if isinstance(key, Sequence):
        return ".".join(key)
    if callable(key):
        return key.__name__
    raise TypeError(f"{key} must be str, Sequence or callable!")


def show_bar_chart(data: dict[str, float | np.floating], x_label: str, y_label: str, title: str):
    """ Create and show a bar chart with the given data, labels and title"""
    fig, ax = plt.subplots(figsize=(10, 6))
    p = ax.bar(*zip(*data.items()))
    ax.bar_label(p, label_type='center', fmt='%.2f')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def bar_chart_per_algorithm(results: list[dict], key: Key, stat_method=np.mean, only_if=lambda res: True):
    algs = get_algorithm_names(results)

    key_per_alg = {
        a: stat_method([get_val(entry, key) for entry in results if entry['algorithm_name'] == a and only_if(entry)])
        for a in algs}

    pretty_key = get_pretty_key(key)
    show_bar_chart(
        data=key_per_alg,
        x_label='Algorithm',
        y_label=pretty_key,
        title=f'{pretty_key} for each algorithm ({stat_method.__name__} over results)'
    )


def bar_chart_per_algorithm_and_oracle(results: list[dict], key: Key, stat_method=np.mean,
                                       only_if=lambda res: True, title: str =None):
    algs = get_algorithm_names(results)
    eq_oracles = get_oracle_names(results)

    key_per_alg_and_orc = {str((a, o)): stat_method([get_val(entry, key) for entry in results
                                                     if entry['algorithm_name'] == a
                                                     and entry['oracle'] == o
                                                     and only_if(entry)])
                           for (a, o) in itertools.product(algs, eq_oracles)}

    pretty_key = get_pretty_key(key)
    show_bar_chart(
        data=key_per_alg_and_orc,
        x_label='Algorithm/Oracle',
        y_label=pretty_key,
        title=f'{pretty_key} for each algorithm and oracle ({stat_method.__name__} over results)' if title is None else title
    )


def bar_chart_per_number_of_original_states(results: list[dict], key: Key, stat_method=np.mean, only_if=lambda res: True):
    min_num_states = min(r['original_automaton_size'] for r in results)
    max_num_states = max(r['original_automaton_size'] for r in results)

    key_per_alg_and_orc = {str(num_states): stat_method([get_val(entry, key) for entry in results
                                                         if entry['original_automaton_size'] == num_states
                                                         and only_if(entry)])
                           for num_states in range(min_num_states, max_num_states+1)}

    pretty_key = get_pretty_key(key)
    show_bar_chart(
        data=key_per_alg_and_orc,
        x_label='Number of states',
        y_label=pretty_key,
        title=f'{pretty_key} for each number of states in the original ({stat_method.__name__} over results)'
    )


def line_chart_per_number_of_original_states(results: list[dict], *keys: Sequence[Key],
                                             stat_method=np.mean, only_if=lambda res: True):
    min_num_states = min(r['original_automaton_size'] for r in results)
    max_num_states = max(r['original_automaton_size'] for r in results)

    plt.figure(figsize=(10, 6))

    def add_labels(line):
        x, y = line.get_data()
        for x, y in zip(x, y):
            plt.annotate(f'{y:.2f}', (x, y))

    for key in keys:
        key_per_num_states = {str(num_states): stat_method([get_val(entry, key) for entry in results
                                                            if entry['original_automaton_size'] == num_states
                                                            and only_if(entry)])
                              for num_states in range(min_num_states, max_num_states + 1)}

        pretty_key = get_pretty_key(key)
        line, = plt.plot(*zip(*key_per_num_states.items()), 'o-', label=pretty_key)
        add_labels(line)

    plt.xlabel("Number of states")
    plt.ylabel("Value")
    plt.title(f"Values over number of states rounds")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def line_chart_per_number_of_original_states_per_alg_and_orac(results: list[dict], key: Key,
                                                              algs: Sequence[str], oracles: Sequence[str],
                                                              stat_method=np.mean, only_if=lambda res: True):
    min_num_states = min(r['original_automaton_size'] for r in results)
    max_num_states = max(r['original_automaton_size'] for r in results)

    plt.figure(figsize=(10, 6))

    def add_labels(line):
        x, y = line.get_data()
        for x, y in zip(x, y):
            plt.annotate(f'{y:.2f}', (x, y))

    for (a, o) in itertools.product(algs, oracles):
        key_per_num_states = {str(num_states): stat_method([get_val(entry, key) for entry in results
                                                            if entry['original_automaton_size'] == num_states
                                                            and entry['algorithm_name'] == a
                                                            and entry['oracle'] == o
                                                            and only_if(entry)])
                              for num_states in range(min_num_states, max_num_states + 1)}

        line, = plt.plot(*zip(*key_per_num_states.items()), 'o-', label=str((a,o)))
        add_labels(line)

    plt.xlabel("Number of states")
    pretty_key = get_pretty_key(key)
    plt.ylabel(pretty_key)
    plt.title(f"{pretty_key} over number of states, for each algorithm/oracle")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()



def stacked_bar_chart_add_traces_per_algorithm_and_oracle(results: list[dict], key: Key, stat_method=np.mean, only_if=lambda res: True):
    algs = get_algorithm_names(results)
    eq_oracles = get_oracle_names(results)

    list_per_alg_and_orac: dict[str, list[dict]] = {str((a, o)): [get_total_num_additional_traces(entry['detailed_learning_info'])
                                                                  for entry in results
                                                                  if entry['algorithm_name'] == a
                                                                  and entry['oracle'] == o
                                                                  and only_if(entry)]
                                                    for (a, o) in itertools.product(algs, eq_oracles)}
    all_processing_steps = set()
    for lst in list_per_alg_and_orac.values():
        for dct in lst:
            for k in dct.keys():
                all_processing_steps.add(k)

    order = ['input_completeness', 'glitch', 'window_cex', 'replay', 'random_walks', 'cex']
    all_processing_steps = sorted(list(all_processing_steps), key=lambda x: order.index(x.split('_', 1)[-1]) if x.split('_', 1)[-1] in order else len(order)+1)

    x_labels = []
    proc_name_to_list_of_num_traces_per_alg: dict[str, list[int]] = {p: [] for p in all_processing_steps}

    for alg_and_orac, add_traces_per_result in list_per_alg_and_orac.items():
        x_labels.append(alg_and_orac)

        proc_name_to_num_traces = defaultdict(int)
        for add_traces in add_traces_per_result:
            for processing_step, number_of_additional_traces in add_traces.items():
                proc_name_to_num_traces[processing_step] += number_of_additional_traces

        for processing_step in all_processing_steps:
            proc_name_to_list_of_num_traces_per_alg[processing_step].append(proc_name_to_num_traces.get(processing_step, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(x_labels))

    for proc_step, list_of_num_traces_per_alg in proc_name_to_list_of_num_traces_per_alg.items():
        p = ax.bar(x_labels, list_of_num_traces_per_alg, width=0.5, label=proc_step, bottom=bottom)
        bottom += list_of_num_traces_per_alg

    ax.set_title("Number of additional traces per algorithm/oracle")
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()


def line_chart_over_learning_rounds(results: list[dict], key: Key, only_if=lambda res: True):
    algs = get_algorithm_names(results)

    plt.figure(figsize=(10, 6))
    colors = {a: plt.colormaps['tab10'](i) for i, a in enumerate(algs)}
    legend_labels = set()

    def add_line(result, color, label):
        learning_info = result['detailed_learning_info']
        steps = sorted(int(step) for step in learning_info.keys())
        values = [get_val(learning_info[str(step)], key, default=-1) for step in steps]

        kwargs = {}
        if label not in legend_labels:
            kwargs["label"] = label
            legend_labels.add(label)

        plt.plot(steps, values, color=color, alpha=0.7, **kwargs)

    for result in results:
        if not only_if(result):
            continue
        alg = result["algorithm_name"]
        if alg in algs:
            add_line(result, color=colors[alg], label=alg)

    plt.xlabel("learning round")
    plt.ylabel(key)
    plt.title(f"{key} across learning rounds")
    plt.legend(title="Algorithm", loc="upper left")
    plt.tight_layout()
    plt.show()


def scatterplot_per_alg(results: list[dict], x_key: Key, y_key: Key, only_if=lambda res: True):
    algs = get_algorithm_names(results)

    fix, ax = plt.subplots()
    colors = {a: plt.colormaps['tab10'](i) for i, a in enumerate(algs)}

    for alg in algs:
        x = [get_val(result, x_key) for result in results if result["algorithm_name"] == alg and only_if(result)]
        y = [get_val(result, y_key) for result in results if result["algorithm_name"] == alg and only_if(result)]
        ax.scatter(x, y, color=colors[alg], label=alg, alpha=0.5)

    ax.legend()
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    ax.grid(True)
    plt.show()