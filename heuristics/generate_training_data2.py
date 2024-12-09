import argparse
import ast
import csv
import itertools
import os
import sys
from asyncio import timeout
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
from aalpy import load_automaton_from_file, MooreMachine
from aalpy.SULs import MooreSUL

from active_pmsatlearn.log import get_logger, set_current_process_name
from active_pmsatlearn.utils import trace_query, get_num_outputs, Trace, timeit
from evaluation.generate_automata import add_automata_generation_arguments
from evaluation.learn_automata import get_all_automata_files_from_dir, get_generated_automata_files_to_learn
from evaluation.utils import parse_range, new_file
from pmsatlearn import run_pmSATLearn

logger = get_logger(__name__)


def _calculate_metrics(learned_model: MooreMachine, info: dict[str, Any], traces: list[Trace]) -> dict:
    metrics = dict()
    complete_num_steps = sum(len(trace) for trace in traces)

    # metrics["num_states"] = len(learned_model.states)
    learned_model.compute_prefixes()
    metrics["num_states_unreachable"] = sum(1 for state in learned_model.states if state.prefix is None)
    metrics["percent_states_unreachable"] = metrics["num_states_unreachable"] / len(learned_model.states) * 100

    #metrics["num_glitches"] = len(info["glitch_steps"])
    metrics["percent_glitches"] = len(info["glitch_steps"]) / complete_num_steps * 100

    metrics["mean_glitch_trans_freq"] = np.mean(info["glitched_delta_freq"] or [np.nan])
    metrics["median_glitch_trans_freq"] = np.median(info["glitched_delta_freq"] or [np.nan])
    metrics["min_glitch_trans_freq"] = np.min(info["glitched_delta_freq"] or [np.nan])
    metrics["max_glitch_trans_freq"] = np.max(info["glitched_delta_freq"] or [np.nan])

    metrics["mean_dominant_trans_freq"] = np.mean(info["dominant_delta_freq"])
    metrics["median_dominant_trans_freq"] = np.median(info["dominant_delta_freq"])
    metrics["max_dominant_trans_freq"] = np.max(info["dominant_delta_freq"])
    metrics["min_dominant_trans_freq"] = np.min(info["dominant_delta_freq"])

    # Convert numpy types to built-ins
    metrics = {
        key: (value.item() if isinstance(value, (np.int64, np.float64)) else value)
        for key, value in metrics.items()
    }

    return metrics


def learn_automaton_and_calculate_metrics(automaton: MooreMachine, diff_min_states=3, diff_max_states=3, extension_length=3, ):
    inputs = list(itertools.product(automaton.get_input_alphabet(), repeat=extension_length))
    sul = MooreSUL(automaton)
    traces = [trace_query(sul, input_combination) for input_combination in inputs]

    logger.info(f"Calculating metrics for learning a MooreMachine with {len(automaton.states)} states, "
                f"{len(automaton.get_input_alphabet())} inputs, {get_num_outputs(traces)} outputs. "
                f"Learning with {diff_min_states} states to {diff_max_states} states, from {len(traces)} traces.")

    results = {}
    learned_models = []
    orig_num_states = len(automaton.states)
    min_states = orig_num_states - diff_min_states
    max_states = orig_num_states + diff_max_states
    for num_states in range(min_states, max_states + 1):
        logger.info(f"Learning {num_states}-state automaton...")
        learned_model, _, info = run_pmSATLearn(data=traces,
                                                n_states=num_states,
                                                automata_type="moore",
                                                glitchless_data=[],
                                                pm_strategy="rc2",
                                                timeout=None,
                                                cost_scheme="per_step",
                                                print_info=False)
        results[num_states - orig_num_states] = _calculate_metrics(learned_model, info, traces) if learned_model is not None else None
        learned_models.append(learned_model)

    assert len(learned_models) == len(results)
    # for num_states, result in results.items():
    #     if result is not None:
    #         assert num_states == result["num_states"], f"{num_states} != {results['num_states']}"

    return results


def write_metrics_to_csv(metrics: list[dict[int, dict[str, float] | None]], csv_file):
    if os.path.splitext(csv_file)[-1] != ".csv":
        csv_file += ".csv"
    logger.info(f"Writing metrics to {csv_file}...")

    metric_names = None

    headers = set(k for stat in metrics for k in stat.keys())
    headers = sorted(headers, key=lambda x: x if isinstance(x, int) else float('inf'))
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for results_for_one_automaton_multiple_num_states in metrics:
            # results_for_one_automaton_multiple_num_states = {1: {metric_name: ...}, 2: ...}
            row = []
            for column in headers:
                entry = results_for_one_automaton_multiple_num_states[column]
                if isinstance(column, str):
                    assert column == "distances"
                    row.append(entry)
                elif isinstance(column, int):
                    num_states = column
                    single_result = entry
                    if single_result is None:
                        row.append(None)
                    else:
                        metric_values_as_list = []
                        metric_names_ = []
                        for metric_name, metric_val in single_result.items():
                            metric_names_.append(f"'{metric_name}'")
                            metric_values_as_list.append(metric_val)

                        if metric_names is None:
                            metric_names = metric_names_
                            f.write("# metrics: [" + ",".join(metric_names) + "]\n")
                        else:
                            assert metric_names == metric_names_, f"Different metric names in first row and current row! First row: {metric_names} | Current row: {metric_names_}"
                        row.append(metric_values_as_list)
                else:
                    assert False
            writer.writerow(row)

    logger.info(f"Finished writing metrics to {csv_file}.")


def parse_metrics_from_csv(csv_file):
    logger.info(f'Loading data from {csv_file}')
    df = pd.read_csv(csv_file, comment='#')

    # extract comment which tells us metric names
    with open(csv_file, 'r') as f:
        while not (line := f.readline()).startswith("#"):
            pass
        metric_names = ast.literal_eval(line.split("# metrics: ")[1])

    results = []

    for r_idx, row in df.iterrows():
        try:
            def str_to_py(string):
                if string in (None, np.nan):
                    return []
                else:
                    string = string.replace('nan', 'None')
                    return ast.literal_eval(string)

            metrics = {num_states_str: dict(zip(metric_names, str_to_py(row[num_states_str]))) for num_states_str in row.keys()}
            results.append(metrics)

        except Exception as e:
            raise type(e)(f"{str(e)} (occurred in row {r_idx}:\n{row}\n)")

    return results

def learn_automata_and_calculate_metrics(automata: list[MooreMachine], *args, **kwargs):
    metrics = [learn_automaton_and_calculate_metrics(a, *args, **kwargs) for a in automata]
    pprint(metrics)

    csv_file = new_file("training_data.csv")
    write_metrics_to_csv(metrics, csv_file)
    return metrics


@timeit("learning automata and calculating metrics")
def learn_automata_from_files_and_calculate_metrics(automata_files: list[str], *args, **kwargs):
    return learn_automata_and_calculate_metrics([load_automaton_from_file(automaton_file, 'moore') for automaton_file in automata_files],
                                                *args, **kwargs)


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Generate training data. Learn the same automaton with PMSATLearn, '
                                                     'with different numbers of states, and collect the results.')
        parser.add_argument('-dmin', '--diff_min_states', type=int, default=3,
                            help='Difference of minimum number states to learn to real number of states')
        parser.add_argument('-dmax', '--diff_max_states', type=int, default=3,
                            help='Difference of maximum number states to learn to real number of states')
        parser.add_argument('-el', '--extension_length', type=int, default=3,
                            help='Extension length to produce traces')

        parser.add_argument('--learn_all_automata_from_dir', type=str,
                            help='Learn all automata from a directory. If this directory is specified, '
                                 'all arguments concerning automata generation except -t are ignored. '
                                 'You have to ensure yourself that all of these automata are instances of the '
                                 'automaton type given with -t.')
        add_automata_generation_arguments(parser)
        parser.add_argument('--reuse', type=bool, default=True,
                            help="Whether to reuse existing automata or to force generation of new ones")
        parser.add_argument('-rd', '--results_dir', type=str, default="learning_results",
                            help='Directory to store results in.')
        parser.add_argument('-pl', '--print_level', type=int, default=0,
                            help="Print level for all algorithms. Usually ranges from 0 (nothing) to 3 (everything).")
        args = parser.parse_args()

        diff_min_states = args.diff_min_states
        diff_max_states = args.diff_max_states
        extension_length = args.extension_length

        learn_from_dir = args.learn_all_automata_from_dir
        num_automata_per_combination = args.num_automata_per_combination
        automata_type = args.type
        num_states = args.num_states
        num_inputs = args.num_inputs
        num_outputs = args.num_outputs
        generated_automata_dir = args.generated_automata_dir
        reuse_existing_automata = args.reuse

    else:
        diff_min_states = int(input("Difference of minimum number states to learn to real number of states: "))
        diff_max_states = int(input("Difference of maximum number states to learn to real number of states: "))
        extension_length = int(input("Extension length: "))

        learn_from_dir = bool(input("Learn all automata from a directory? (y/n) (default: no): ") in ("y", "Y", "yes"))
        if learn_from_dir:
            learn_from_dir = input(f"Enter directory (all contained automata must be moore machines): ")
        else:
            num_automata_per_combination = int(input("Enter number of automata to generate/learn (default: 5): ") or 5)
            num_states = parse_range(input("Enter number of states per automaton: "))
            num_inputs = parse_range(input("Enter number of inputs per automaton: "))
            num_outputs = parse_range(input("Enter number of outputs per automaton: "))
            generated_automata_dir = input("Enter directory to store generated automata in: ") or "generated_automata"
            reuse_existing_automata = input("Reuse existing automata? (y/n) (default: yes): ") in ("y", "Y", "yes", "")


    if learn_from_dir:
        logger.warning("Learning from directory has not been tested; confirm manually!")
        files_to_learn = get_all_automata_files_from_dir(learn_from_dir)
    else:
        files_to_learn = get_generated_automata_files_to_learn(num_automata_per_combination, automata_type,
                                                               num_states, num_inputs, num_outputs,
                                                               generated_automata_dir, reuse_existing_automata)

    set_current_process_name(os.path.basename(__file__))
    learn_automata_from_files_and_calculate_metrics(files_to_learn,
                                                    diff_min_states=diff_min_states,
                                                    diff_max_states=diff_max_states,
                                                    extension_length=extension_length, )




if __name__ == '__main__':
    main()