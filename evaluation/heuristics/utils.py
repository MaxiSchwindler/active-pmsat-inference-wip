import io
import itertools
import json
import re
import os
import io
import sys
import functools
from contextlib import redirect_stdout
from pathlib import Path
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
from aalpy import MooreMachine, MooreState
from aalpy.utils import load_automaton_from_file

from typing import Any, TypeAlias
from collections.abc import Callable
HeuristicFunction: TypeAlias = Callable[[MooreMachine, dict[str, Any], list[list[str | list[str]]]], float]

DEFAULT_DATA_DIR = "../../generated_data_4"
EXAMPLES = "ping_pong_example", "simple_example_server", "simple_example_server_with_glitch"


def load_data(model_name: str, models_dir: str) -> tuple[MooreMachine, tuple[MooreMachine, dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """ Load the data from one learned model.
    NOTE: This only returns actually learned models!
          If a model could not be learned with the given number of states, it is not included in learned_models!
          Therefore, no info about runs with too few states (not possible to learn) or too many states (timeout) is included!

    Returns a tuple of:
        real model
        list of tuples of (learned model, pmsat info, learning result)
        info dict of whole run (containing used traces)
    """

    def load_automaton(*args, **kwargs) -> MooreMachine:
        """ Call aalpy.utils.load_automaton_from_file, but suppress it's print output
        (aalpy prints a warning that our automata are not input complete) """
        with io.StringIO() as f, redirect_stdout(f):
            return load_automaton_from_file(*args, **kwargs)

    model_dir = Path(models_dir) / model_name
    assert model_dir.exists(), f"{model_dir} does not exist!"
    real_model = load_automaton(model_dir / "RealModel.dot", "moore")

    pmsat_learned_models = {}
    pmsat_infos = {}
    learning_results = {}
    model_pattern = r"LearnedModel_(\d+)States\.dot"
    info_pattern = r"LearningInfo_(\d+)States\.json"
    result_pattern = r"LearningResults_(\d+)States\.json"
    for root, _, files in os.walk(model_dir):
        for file in files:
            if match := re.search(model_pattern, file):
                pmsat_learned_models[match.group(1)] = load_automaton(model_dir / file, "moore")
            elif match := re.search(info_pattern, file):
                with open(model_dir / file, "r") as f:
                    pmsat_infos[match.group(1)] = json.load(f)
            elif match := re.search(result_pattern, file):
                with open(model_dir / file, "r") as f:
                    learning_results[match.group(1)] = json.load(f)

    models_and_info = []
    for num_states_learned, model in pmsat_learned_models.items():
        pmsat_info = pmsat_infos[num_states_learned]
        learning_result = learning_results[num_states_learned]
        models_and_info.append((model, pmsat_info, learning_result))

    with open(model_dir / "info.json", "r") as f:
        info = json.load(f)

    assert len(models_and_info) > 0, f"No learned models found for {model_name} in {model_dir}"

    return real_model, models_and_info, info


def load_example(example: str) -> tuple[MooreMachine, tuple[MooreMachine, dict[str, Any]], dict[str, Any]]:
    """ Load the example from the pmsat-inference repo.

    Returns a tuple of:
        real model
        tuple or (learned model, pmsat info)
        info dict (containing used traces)
    """

    def load_automaton(*args, **kwargs) -> MooreMachine:
        """ Call aalpy.utils.load_automaton_from_file, but suppress it's print output
        (aalpy prints a warning that our automata are not input complete) """
        with io.StringIO() as f, redirect_stdout(f):
            return load_automaton_from_file(*args, **kwargs)

    assert example in EXAMPLES
    example_dir = Path("../pmsat-inference/examples-results/" + example)
    real_model = load_automaton(example_dir / "RealModel.dot", "moore")

    pmsat_learned_models = {}
    pmsat_infos = {}
    base_pattern = "pmsatLearned-rc2-N(\d)-RN(2|3|4)"
    model_pattern = fr"{base_pattern}\.dot"
    info_pattern = fr"{base_pattern}\.json"
    for root, _, files in os.walk(example_dir):
        for file in files:
            if match := re.search(model_pattern, file):
                pmsat_learned_models[match.group(1)] = load_automaton(example_dir / file, "moore")
            elif match := re.search(info_pattern, file):
                with open(example_dir / file, "r") as f:
                    pmsat_infos[match.group(1)] = json.load(f)

    models_and_info = []
    for num_states_learned, model in pmsat_learned_models.items():
        pmsat_info = pmsat_infos[num_states_learned]
        models_and_info.append((model, pmsat_info))

    with open(example_dir / "info.json", "r") as f:
        info = json.load(f)

    return real_model, models_and_info, info


def calculate_heurisic_scores(real_model: MooreMachine, learned_models: list[tuple[MooreMachine, dict, dict]],
                              infos: dict, heuristic: HeuristicFunction) -> dict[int, float]:
    """
    Calculate the heuristic scores for multiple learned models of one single automaton.
    :param real_model: the original automaton
    :param learned_models: list of tuples of (learned model, pmsat info)
    :param infos: learning info dict for the whole run (e.g. containing used traces)
    :param heuristic: the heuristic function to calculate the scores with

    :return a dict of (distance_to_real_model -> heuristic score)
    """
    traces = infos["traces"]

    def distance(real_model: MooreMachine, learned_model: MooreMachine):
        return len(learned_model.states) - len(real_model.states)

    distance_to_heuristic = {
        distance(real_model, learned_model): heuristic(learned_model, learning_info, traces) for
        learned_model, learning_info, learning_result in learned_models
    }

    return distance_to_heuristic


def plot_heuristic_scores(real_model: MooreMachine, learned_models: list[tuple[MooreMachine, dict, dict]], infos: dict,
                          heuristic: HeuristicFunction, new_plot: bool = True, show_plot: bool = True,
                          as_line: bool = False, mark_best_model: bool = False):
    """
    Plot the heuristic scores of multiple learned models of one original automaton over distance to the original automaton.
    This calls calculate_heuristic_scores() to calculate the values.

    :param real_model: the original automaton
    :param learned_models: list of tuples of (learned model, pmsat info)
    :param infos: learning info dict for the whole run (e.g. containing used traces)
    :param heuristic: the heuristic function to calculate the scores with
    :param new_plot: whether to create a new plot, or to write to the current plot instance in plt
    :param show_plot: whether to show the plot directly
    :param as_line: Whether to connect all points in this plot by a line
    :param mark_best_model: whether to mark the best model in the plot by
    """
    scores = calculate_heurisic_scores(real_model, learned_models, infos, heuristic)
    if new_plot:
        plt.figure()

    plt.axvline(0, color='red', linewidth=0.5, linestyle='--')  # TODO this line is added many times

    if as_line:
        plt.plot(scores.keys(), scores.values())
    else:
        plt.scatter(scores.keys(), scores.values())

    if mark_best_model:
        best_key = max(scores, key=scores.get)
        best_value = scores[best_key]
        plt.scatter([best_key], [best_value], marker='*', facecolor='none', edgecolors="red", s=75, zorder=5)

    curr_ticks, curr_tick_labels = plt.xticks()
    if not set(scores.keys()).issubset(set(curr_ticks)) or any(int(ct) != ct for ct in curr_ticks):
        # add new ticks if (a) current keys are not in current ticks or (b) current ticks contains floats
        plt.xticks(list(scores.keys()))

    plt.xlabel("Distance to ground truth")
    plt.ylabel(f"{heuristic.__name__} score")
    plt.title(f"Scores of heuristic '{heuristic.__name__}'")

    if show_plot:
        plt.show()


def calculate_heuristic_scores_for_example(example: str, heuristic: HeuristicFunction) -> dict[int, float]:
    """ Calculate the heuristic scores of a given example from the pmsat-inference repo. """
    real_model, learned_models, infos = load_example(example)
    return calculate_heurisic_scores(real_model, learned_models, infos, heuristic)


def plot_heuristic_scores_for_example(example: str, heuristic: HeuristicFunction, **kwargs):
    """ Plot the heuristic scores of a given example from the pmsat-inference repo."""
    real_model, learned_models, infos = load_example(example)
    plot_heuristic_scores(real_model, learned_models, infos, heuristic, **kwargs)


def calculate_heuristic_scores_for_model(model: str, heuristic: HeuristicFunction,
                                         models_dir: str = DEFAULT_DATA_DIR) -> dict[int, float]:
    """ Calculate the heuristic scores of the learnt models of a given model from the given models directory"""
    real_model, learned_models, infos = load_data(model, models_dir)
    return calculate_heurisic_scores(real_model, learned_models, infos, heuristic)


def plot_heuristic_scores_for_model(model: str, heuristic: HeuristicFunction, models_dir: str = DEFAULT_DATA_DIR,
                                    **kwargs):
    """ Plot the heuristic scores of the learnt models of a given model from the given models directory"""
    real_model, learned_models, infos = load_data(model, models_dir)
    plot_heuristic_scores(real_model, learned_models, infos, heuristic, **kwargs)


def plot_heuristic_scores_for_all_models_in_dir(heuristic: HeuristicFunction, models_dir: str = DEFAULT_DATA_DIR,
                                                **kwargs):
    """ Plot the heuristic scores of learnt models of all models in the given models directory"""
    plt.figure()
    for model_name in os.listdir(models_dir):
        if not (Path(models_dir) / model_name).is_dir():
            continue
        plot_heuristic_scores_for_model(model_name, heuristic, models_dir=models_dir, new_plot=False, show_plot=False,
                                        **kwargs)
    plt.show()


def _calc_dist_to_num_best_models(heuristic: HeuristicFunction, models_dir=DEFAULT_DATA_DIR):
    dist_to_num_best_models_at_dist = defaultdict(int)
    for model_name in os.listdir(models_dir):
        if not (Path(models_dir) / model_name).is_dir():
            continue
        scores = calculate_heuristic_scores_for_model(model_name, heuristic, models_dir=models_dir)
        best_key = max(scores, key=scores.get)
        dist_to_num_best_models_at_dist[best_key] += 1

    return dict(dist_to_num_best_models_at_dist)


def plot_number_of_best_models_identified(heuristic: HeuristicFunction, models_dir: str = DEFAULT_DATA_DIR,
                                          new_plot=False, show_plot=True, as_line=False):
    """
    Plot the number of best models identified with the given heuristic, calculated on all models in the given models directory

    :param heuristic: The heuristic to use
    :param models_dir: The models directory
    :param new_plot: Whether to create a new plot instance or use the existing one
    :param show_plot: Whether to show the plot
    :param as_line: Whether to show the data points as a line or as disconnected points
    """
    dist_to_num_best_models_at_dist = _calc_dist_to_num_best_models(heuristic, models_dir=models_dir)
    if new_plot:
        plt.figure()

    label = heuristic.__name__
    if as_line:
        keys = sorted(dist_to_num_best_models_at_dist.keys())
        vals = [dist_to_num_best_models_at_dist[k] for k in keys]
        plt.plot(keys, vals, label=label)
    else:
        plt.scatter(dist_to_num_best_models_at_dist.keys(), dist_to_num_best_models_at_dist.values(), label=label)

    plt.title(f"Number of best models identified at distance")
    plt.xlabel("Distance to ground truth")
    plt.ylabel("Number of best models identified")
    plt.legend(loc=(1.04, 1))

    if show_plot:
        plt.show()


def print_heuristic_stats_for_all_models_in_dir(*heuristics: HeuristicFunction, models_dir=DEFAULT_DATA_DIR, **kwargs):
    from prettytable import PrettyTable

    num_models = sum(1 for m in os.listdir(models_dir) if (Path(models_dir) / m).is_dir())

    stats = {}
    for heuristic in heuristics:
        stats[heuristic.__name__] = _calc_dist_to_num_best_models(heuristic, models_dir=models_dir)

    all_distances = set()
    for dist_to_num_best in stats.values():
        all_distances.update(dist_to_num_best.keys())
    all_distances = sorted(all_distances)

    table = PrettyTable()
    table.field_names = ["Distance"] + [heuristic.__name__ for heuristic in heuristics]

    for distance in all_distances:
        row = [distance]
        for heuristic in heuristics:
            dist_to_num_best = stats[heuristic.__name__]
            num_best_models = dist_to_num_best.get(distance, 0)
            percent = (num_best_models / num_models * 100) if num_models > 0 else 0
            row.append(f"{num_best_models} ({percent:.1f}%)")
        table.add_row(row)

    print(table)


def compare_heuristics(heuristics: list[HeuristicFunction], models_dir: str = DEFAULT_DATA_DIR):
    for heuristic in heuristics:
        plot_heuristic_scores_for_all_models_in_dir(heuristic, models_dir=models_dir, as_line=True,
                                                    mark_best_model=True)

    plt.figure()
    for heuristic in heuristics:
        plot_number_of_best_models_identified(heuristic, models_dir=models_dir, new_plot=False, show_plot=False,
                                              as_line=True)
    plt.show()

    print_heuristic_stats_for_all_models_in_dir(*heuristics, models_dir=models_dir)