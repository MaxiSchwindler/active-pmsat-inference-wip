import argparse
import ast
import csv
import itertools
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pebble
from aalpy import load_automaton_from_file, MooreMachine
from aalpy.SULs import MooreSUL
from pebble import concurrent

from active_pmsatlearn.log import get_logger, set_current_process_name
from active_pmsatlearn.utils import trace_query, get_num_outputs, timeit
from active_pmsatlearn.defs import Trace
from evaluation.generate_automata import add_automata_generation_arguments
from evaluation.learn_automata import get_all_automata_files_from_dir, get_generated_automata_files_to_learn
from evaluation.utils import parse_range, new_file, GlitchingSUL
from pmsatlearn import run_pmSATLearn

logger = get_logger(__name__)


def _calculate_metrics(learned_model: MooreMachine, info: dict[str, Any], traces: list[Trace]) -> dict[str, float]:
    metrics = dict()
    complete_num_steps = sum(len(trace[1:]) for trace in traces)  # !!!

    metrics["num_states"] = len(learned_model.states)
    learned_model.compute_prefixes()
    metrics["num_states_unreachable"] = sum(1 for state in learned_model.states if state.prefix is None)
    metrics["percent_states_unreachable"] = metrics["num_states_unreachable"] / len(learned_model.states) * 100

    metrics["num_glitches"] = len(info["glitch_steps"])
    metrics["percent_glitches"] = len(info["glitch_steps"]) / complete_num_steps * 100

    metrics["mean_glitch_trans_freq"] = np.mean(info["glitched_delta_freq"] or [np.nan])
    metrics["median_glitch_trans_freq"] = np.median(info["glitched_delta_freq"] or [np.nan])
    metrics["min_glitch_trans_freq"] = np.min(info["glitched_delta_freq"] or [np.nan])
    metrics["max_glitch_trans_freq"] = np.max(info["glitched_delta_freq"] or [np.nan])

    metrics["mean_dominant_trans_freq"] = np.mean(info["dominant_delta_freq"])
    metrics["median_dominant_trans_freq"] = np.median(info["dominant_delta_freq"])
    metrics["max_dominant_trans_freq"] = np.max(info["dominant_delta_freq"])
    metrics["min_dominant_trans_freq"] = np.min(info["dominant_delta_freq"])

    metrics["glitched_delta_freq"] = info["glitched_delta_freq"]
    metrics["dominant_delta_freq"] = info["dominant_delta_freq"]

    # Convert numpy types to built-ins
    metrics = {
        key: (value.item() if isinstance(value, (np.int64, np.float64)) else value)
        for key, value in metrics.items()
    }

    return metrics


def write_single_automaton_results(results_dir: str | Path, original_automaton_file: str | Path,
                                   learned_model: MooreMachine | None, learning_info: dict[str, Any], result: dict[str, Any]):
    results_dir = Path(results_dir)
    original_automaton_file = Path(original_automaton_file)
    assert original_automaton_file.is_file()
    assert original_automaton_file.parent == results_dir

    assert learning_info["num_states"] == result["num_states"]

    # write learned model to dir
    if learned_model is not None:
        num_states = len(learned_model.states)
        assert num_states == learning_info["num_states"]
        learned_model.save(results_dir/f"LearnedModel_{num_states}States.dot")
    else:
        num_states = learning_info["num_states"]

    # write learning info to dir
    with open(results_dir/f"LearningInfo_{num_states}States.json", 'w') as f:
        json.dump(learning_info, f, indent=4)

    # write metric results to dir
    with open(results_dir/f"LearningResults_{num_states}States.json", 'w') as f:
        json.dump(result, f, indent=4)


def learn_automaton_and_calculate_metrics(automaton_file: str, *,
                                          diff_min_states=3, diff_max_states=3, extension_length=3,
                                          glitch_percentage=0,
                                          results_dir="generated_data") -> list[dict[str, float]]:
    # create directory for this automaton's result and copy automaton file there
    results_dir = Path(results_dir) / Path(automaton_file).stem
    os.makedirs(results_dir, exist_ok=False)
    automaton_file = shutil.copyfile(automaton_file, results_dir/"RealModel.dot")

    # load automaton, set up sul, create traces
    automaton = load_automaton_from_file(automaton_file, "moore")
    input_alphabet = automaton.get_input_alphabet()
    output_alphabet = sorted(list(set(s.output for s in automaton.states)))
    orig_num_states = len(automaton.states)

    inputs = list(itertools.product(automaton.get_input_alphabet(), repeat=extension_length))
    sul = MooreSUL(automaton)
    if glitch_percentage:
        sul = GlitchingSUL(sul, glitch_percentage=glitch_percentage)
    traces = [trace_query(sul, input_combination) for input_combination in inputs]

    # write info (incl. traces) to file
    info = dict()
    info["original_automaton"] = str(automaton_file)
    info["num_states"] = orig_num_states
    info["glitch_percent"] = glitch_percentage
    info["input_alphabet"] = input_alphabet
    info["output_alphabet"] = output_alphabet
    info["traces"] = traces
    with open(results_dir/"info.json", 'w') as f:
        json.dump(info, f, indent=4)

    # set up loop
    min_states = orig_num_states - diff_min_states
    max_states = orig_num_states + diff_max_states
    logger.info(f"Calculating metrics for learning a MooreMachine with {len(automaton.states)} states, "
                f"{len(input_alphabet)} inputs, {len(output_alphabet)} outputs. "
                f"Learning with {min_states} states to {max_states} states, from {len(traces)} traces.")

    # learn for each num_state, calculate results and write single results to file within automaton dir
    results = []
    for num_states in range(min_states, max_states + 1):
        logger.info(f"Learning {num_states}-state automaton...")
        learned_model, _, learning_info = run_pmSATLearn(data=traces,
                                                n_states=num_states,
                                                automata_type="moore",
                                                glitchless_data=[],
                                                pm_strategy="rc2",
                                                timeout=5*60,
                                                cost_scheme="per_step",
                                                print_info=False)
        result = _calculate_metrics(learned_model, learning_info, traces) if learned_model is not None else {}
        additional_knowledge = {
            "distance": num_states - len(automaton.states),
            "automaton": str(automaton_file),
            "num_traces": len(traces),
            "num_states": num_states
        }
        result = additional_knowledge | result
        results.append(result)
        write_single_automaton_results(results_dir, automaton_file, learned_model, learning_info, result)

    # write the results of all runs to file in folder
    write_metrics_to_csv(results, results_dir/"results.csv")

    return results


def learn_automaton_and_calculate_metrics_mp(kwargs):
    set_current_process_name(f"LEARN_{uuid.uuid4().hex[:4]}")
    return learn_automaton_and_calculate_metrics(**kwargs)


def write_metrics_to_csv(metrics: list[dict[str, float]], csv_file):
    if os.path.splitext(csv_file)[-1] != ".csv":
        csv_file += ".csv"
    logger.info(f"Writing metrics to {csv_file}...")

    max_fields = -1
    for result in metrics:
        if len(result.keys()) > max_fields:
            fieldnames = list(result.keys())
            max_fields = len(fieldnames)

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    logger.info(f"Finished writing metrics to {csv_file}.")


@timeit("learning automata and calculating metrics")
def learn_automata_and_calculate_metrics(automata_files: list[str], multiprocessing=True, results_dir="generated_data", **kwargs) -> list[dict[str, float]]:
    results_dir = new_file(results_dir)
    if multiprocessing:
        pool = pebble.ProcessPool(max_workers=4)
        def generator():
            for a in automata_files:
                yield {"automaton_file": a, "results_dir": results_dir} | kwargs

        futures = pool.map(learn_automaton_and_calculate_metrics_mp, generator())
        metrics = []
        for future in futures.result():
            metrics.extend(future)
    else:
        metrics = []
        for a in automata_files:
            metrics.extend(learn_automaton_and_calculate_metrics(a, results_dir=results_dir, **kwargs))

    csv_file = os.path.join(results_dir, "complete_data.csv")
    write_metrics_to_csv(metrics, csv_file)
    return metrics


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Generate data. Learn the same automaton with PMSATLearn, '
                                                     'with different numbers of states, and collect the results.')
        parser.add_argument('-dmin', '--diff_min_states', type=int, default=3,
                            help='Difference of minimum number states to learn to real number of states')
        parser.add_argument('-dmax', '--diff_max_states', type=int, default=3,
                            help='Difference of maximum number states to learn to real number of states')
        parser.add_argument('-el', '--extension_length', type=int, default=3,
                            help='Extension length to produce traces')
        parser.add_argument('-gp', '--glitch_percent', type=float, default=0,
                            help='Percentage of glitches')

        # TODO: refactor to --files like in learn_automata.py
        parser.add_argument('--learn_all_automata_from_dir', type=str,
                            help='Learn all automata from a directory. If this directory is specified, '
                                 'all arguments concerning automata generation except -t are ignored. '
                                 'You have to ensure yourself that all of these automata are instances of the '
                                 'automaton type given with -t.')
        add_automata_generation_arguments(parser)
        parser.add_argument('--reuse', type=bool, default=True,
                            help="Whether to reuse existing automata or to force generation of new ones")
        parser.add_argument('-mp', '--multiprocessing', type=bool, default=True,
                            help="Whether to use multiprocessing or not")
        parser.add_argument('-rd', '--results_dir', type=str, default="generated_data",
                            help="Directory to store results in")
        args = parser.parse_args()

        diff_min_states = args.diff_min_states
        diff_max_states = args.diff_max_states
        extension_length = args.extension_length
        glitch_percent = args.glitch_percent

        learn_from_dir = args.learn_all_automata_from_dir
        num_automata_per_combination = args.num_automata_per_combination
        automata_type = args.type
        num_states = args.num_states
        num_inputs = args.num_inputs
        num_outputs = args.num_outputs
        generated_automata_dir = args.generated_automata_dir
        reuse_existing_automata = args.reuse
        use_multiprocessing = args.multiprocessing
        results_dir = args.results_dir

    else:
        # TODO: remove/refactor like learn_automata.py
        diff_min_states = int(input("Difference of minimum number states to learn to real number of states: "))
        diff_max_states = int(input("Difference of maximum number states to learn to real number of states: "))
        extension_length = int(input("Extension length: "))
        glitch_percent = float(input("Glitch percentage: "))

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
        use_multiprocessing = input("Use multiprocessing? (y/n) (default: yes): ") in ("y", "Y", "yes", "")
        results_dir = input("Enter directory to store results in: ") or "generated_data"

    if learn_from_dir:
        logger.warning("Learning from directory has not been tested; confirm manually!")
        files_to_learn = get_all_automata_files_from_dir(learn_from_dir)
    else:
        files_to_learn = get_generated_automata_files_to_learn(num_automata_per_combination, automata_type,
                                                               num_states, num_inputs, num_outputs,
                                                               generated_automata_dir, reuse_existing_automata)
    set_current_process_name(os.path.basename(__file__))
    learn_automata_and_calculate_metrics(files_to_learn,
                                         multiprocessing=use_multiprocessing,
                                         diff_min_states=diff_min_states,
                                         diff_max_states=diff_max_states,
                                         extension_length=extension_length,
                                         glitch_percentage=glitch_percent,
                                         results_dir=results_dir,)




if __name__ == '__main__':
    main()