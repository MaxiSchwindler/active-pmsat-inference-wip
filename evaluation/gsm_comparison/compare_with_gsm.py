import argparse
import json
import pathlib
import time
from pathlib import Path

import aalpy
from aalpy import MooreMachine, load_automaton_from_file, bisimilar

from active_pmsatlearn import get_logger
from evaluation.charts import load_results
from evaluation.gsm_comparison.gsm import run as run_gsm_baseline
from evaluation.learn_automata import compute_accuracy
from evaluation.utils import existing_dir

logger = get_logger("COMPARE_WITH_GSM")


def compare_result_with_gsm(result: dict) -> dict:
    logger.info(f"Comparing {result['results_file']}")

    automaton_file = result['original_automaton']
    # TODO quickfix after repo transition - remove in the future
    old_sub_path = 'MastersThesis\\active-pmsat-inference-wip'
    new_sub_path = 'MastersThesis\\OLD_active-pmsat-inference-wip'
    if old_sub_path in automaton_file:
        automaton_file = automaton_file.replace(old_sub_path, new_sub_path)

    original_automaton: MooreMachine = load_automaton_from_file(automaton_file, "moore")

    num_rounds = result['learning_rounds']
    traces_of_last_round = result['detailed_learning_info'][str(num_rounds)]["traces_used_to_learn"]
    percent_glitches = result['glitch_percent']

    gsm_kwargs = dict(
        data=traces_of_last_round,
        output_behavior="moore",
        failure_rate=percent_glitches / 100,
        certainty=0.05,
    )

    logger.info(f"Running GSM without purge_mismatches...")
    start_time = time.time()
    gsm_mm_nopurge: MooreMachine = run_gsm_baseline(**gsm_kwargs, purge_mismatches=False)
    end_time = time.time()
    gsm_stats_nopurge = {
        "accuracy": compute_accuracy(original_automaton, gsm_mm_nopurge),
        "learned_automaton_size": gsm_mm_nopurge.size,
        "bisimilar": bisimilar(original_automaton, gsm_mm_nopurge),
        "total_time": end_time - start_time,
    }

    logger.info(f"Running GSM with purge_mismatches...")
    start_time = time.time()
    gsm_mm_purge: MooreMachine = run_gsm_baseline(**gsm_kwargs, purge_mismatches=True)
    end_time = time.time()
    gsm_stats_purge = {
        "accuracy": compute_accuracy(original_automaton, gsm_mm_purge),
        "learned_automaton_size": gsm_mm_purge.size,
        "bisimilar": bisimilar(original_automaton, gsm_mm_purge),
        "total_time": end_time - start_time,
    }

    def _relevant(stats: dict) -> dict:
        # remove other stats/things we don't care about
        return {
            "bisimilar": stats["bisimilar"],
            "Precision": stats["Precision"],
            "Recall": stats["Recall"],
            "F-Score": stats["F-Score"],
            "num_states": stats["learned_automaton_size"],
            "total_time": stats["total_time"],
        }

    return {
        "results_file": result['results_file'],
        "original_automaton": result['original_automaton'],
        "original_num_states": original_automaton.size,
        "glitch_percent": result["glitch_percent"],
        "algorithm_name": result['algorithm_name'],
        "oracle": result['oracle'],

        "apmsl_algorithm": _relevant(result),
        "GSM_with_purge_mismatches": _relevant(gsm_stats_purge),
        "GSM_without_purge_mismatches": _relevant(gsm_stats_nopurge),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare learning results (from learn_automata.py) with a GSM baseline.')
    parser.add_argument('folder', type=existing_dir, nargs="?",
                        help='Directory of learning results.')
    args = parser.parse_args()

    learning_results = load_results(args.folder)
    if not learning_results:
        raise ValueError(f"No learning result files found in {args.folder}.")

    comparison_files_folder: Path = args.folder / "GSM_comparison"
    if comparison_files_folder.exists():
        raise ValueError(f"Folder {comparison_files_folder.absolute()} already exists.")
    comparison_files_folder.mkdir(exist_ok=False)

    logger.info(f"Collected {len(learning_results)} learning result files")
    for result in learning_results:
        if result.get("exception", None) is not None:
            logger.info(f"Result {result['results_file']} was invalid, skipping")
            continue

        comparison_result = compare_result_with_gsm(result)
        with open(comparison_files_folder / f"COMPARE_{comparison_result['results_file']}", "w") as f:
            json.dump(comparison_result, f, indent=4)


if __name__ == '__main__':
    main()