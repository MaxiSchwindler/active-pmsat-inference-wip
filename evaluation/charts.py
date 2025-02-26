import itertools
import json
import os
from collections import defaultdict
from collections.abc import Sequence, Callable
from functools import cmp_to_key
from typing import TypeAlias, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def get_all_possible_values_for_key(results: list[dict], key: Key, only_if=lambda res: True) -> list[Any]:
    return sorted(list(set(get_val(entry, key) for entry in results if only_if(entry))))


def load_results(results_dir: str) -> list[dict]:
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith('info') or not filename.endswith(".json"):
            continue
        with open(os.path.join(results_dir, filename), "r") as f:
            results.append(json.load(f))
        results[-1]["results_file"] = filename
    return results


def get_val(result: dict, key: Key, *, default=RAISE, callable_kwargs: dict = None):
    """
    Get the value of @key in the given result.
    Supports nested keys, e.g. ("last_pmsat_info", "num_vars")
    """
    assert isinstance(key, Sequence) or isinstance(key, str) or callable(key)
    if callable_kwargs:
        assert callable(key), f"{key} is not callable but callable_kwargs is specified!"
    else:
        callable_kwargs = {}

    try:
        if callable(key):
            return key(result, **callable_kwargs)
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
                                       only_if=lambda res: True, title: str=None, group_by: Key = None):
    algs = get_algorithm_names(results)
    eq_oracles = get_oracle_names(results)

    pretty_key = get_pretty_key(key)
    if group_by is None:
        key_per_alg_and_orc = {str((a, o)): stat_method([get_val(entry, key) for entry in results
                                                         if entry['algorithm_name'] == a
                                                         and entry['oracle'] == o
                                                         and only_if(entry)])
                               for (a, o) in itertools.product(algs, eq_oracles)}

        show_bar_chart(
            data=key_per_alg_and_orc,
            x_label='Algorithm/Oracle',
            y_label=pretty_key,
            title=f'{pretty_key} for each algorithm and oracle ({stat_method.__name__} over results)' if title is None else title
        )
    else:
        oracle_hatches = {oracle: hatch for oracle, hatch in
                          zip(eq_oracles, ['', '...', 'xx', 'o', '*', '/', '\\', '|', '-', '+',  'O', ])}

        all_possible_vals_for_group_by_key = get_all_possible_values_for_key(results, group_by, only_if)
        group_by_val_to_list_of_values = {g: [] for g in all_possible_vals_for_group_by_key}

        alg_and_oracles = []
        for (a, o) in itertools.product(algs, eq_oracles):
            for group_by_val in all_possible_vals_for_group_by_key:
                val = stat_method([get_val(entry, key) for entry in results
                                   if only_if(entry)
                                   and get_val(entry, group_by) == group_by_val
                                   and entry['algorithm_name'] == a
                                   and entry['oracle'] == o])
                group_by_val_to_list_of_values[group_by_val].append(val)
            alg_and_oracles.append((a, o))

        label_locations = np.arange(len(alg_and_oracles))
        width = 1 / (len(all_possible_vals_for_group_by_key)+1)
        multiplier = 0

        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

        for group_by_val, list_of_values in group_by_val_to_list_of_values.items():
            offset = width * multiplier
            rects = ax.bar(label_locations + offset, list_of_values, width,
                           label=group_by_val,
                           hatch=[oracle_hatches[o] for a, o in alg_and_oracles])
            ax.bar_label(rects, label_type='center', fmt='%.2f')
            multiplier += 1

        ax.set_ylabel(pretty_key)
        ax.set_xlabel("Algorithm/Oracle")
        ax.set_xticks(label_locations + width, [str(ao) for ao in alg_and_oracles])
        ax.set_title(f'{pretty_key} for each algorithm and oracle ({stat_method.__name__} over results)' if title is None else title)
        ax.legend(title="Glitch Percentage")
        plt.xticks(rotation=45, ha='right')


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
    plt.title(f"Values over number of states of original automaton")
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

    marker_per_alg = {alg: marker for alg, marker in
                      zip(algs, ['o', '*', 'p', 'h', 'd', 'v', '8', 's', 'P', 'H', 'X', '1', '2', '3' ])}
    color_per_alg =  {alg: color for alg, color in
                      zip(algs, mcolors.TABLEAU_COLORS)}
    line_style_per_oracle = {orac: style for orac, style in
                             zip(oracles, ['-', ':', '-.', '--'])}

    for (a, o) in itertools.product(algs, oracles):
        key_per_num_states = {str(num_states): stat_method([get_val(entry, key) for entry in results
                                                            if entry['original_automaton_size'] == num_states
                                                            and entry['algorithm_name'] == a
                                                            and entry['oracle'] == o
                                                            and only_if(entry)])
                              for num_states in range(min_num_states, max_num_states + 1)}
        fmt = f"{line_style_per_oracle[o]}{marker_per_alg[a]}"
        color = color_per_alg[a]
        line, = plt.plot(*zip(*key_per_num_states.items()), fmt, label=str((a,o)), color=color)
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
        ax.bar_label(p, label_type='center', fmt='%.2f')
        bottom += list_of_num_traces_per_alg

    ax.set_title("Number of additional traces per algorithm/oracle")
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()


def line_chart_over_learning_rounds(results: list[dict], key: Key, only_if=lambda res: True, stay_at_last_val=lambda val: False,
                                    get_from_full_result: bool = False, colorcode_by_alg: bool = True):
    algs = get_algorithm_names(results)

    plt.figure(figsize=(10, 6))
    colors = {a: plt.colormaps['tab10'](i) for i, a in enumerate(algs)}
    legend_labels = set()

    def add_line(result, color, label):
        learning_info = result['detailed_learning_info']
        steps = sorted(int(step) for step in learning_info.keys())
        if not get_from_full_result:
            values = [get_val(learning_info[str(step)], key, default=-1) for step in steps]
        else:
            values = [get_val(result, key, default=-1, callable_kwargs=dict(step=step)) for step in steps]

        for i in range(len(values)):
            if stay_at_last_val(values[i]) and i > 0:
                values[i] = values[i-1]

        kwargs = {}
        if label not in legend_labels:
            kwargs["label"] = label
            legend_labels.add(label)
        if colorcode_by_alg:
            kwargs["color"] = color

        lines = plt.plot(steps, values, alpha=0.7, **kwargs)
        plt.scatter(steps[-1], values[-1], alpha=0.7, marker='*', color=lines[0].get_color())

    for result in results:
        if not only_if(result):
            continue
        alg = result["algorithm_name"]
        if alg in algs:
            add_line(result, color=colors[alg], label=alg)

    plt.xlabel("learning round")
    pretty_key = get_pretty_key(key)
    plt.ylabel(pretty_key)
    plt.title(f"{pretty_key} across learning rounds")
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
    plt.xlabel(get_pretty_key(x_key))
    plt.ylabel(get_pretty_key(y_key))
    ax.grid(True)
    plt.show()