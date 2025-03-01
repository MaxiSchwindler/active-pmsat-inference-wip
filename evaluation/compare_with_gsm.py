import argparse
import json
import pathlib
from pathlib import Path

import aalpy
from aalpy import MooreMachine, load_automaton_from_file, bisimilar

from active_pmsatlearn import get_logger
from evaluation.charts import load_results
from evaluation.gsm_baseline import run as run_gsm_baseline
from evaluation.learn_automata import calculate_statistics
from evaluation.utils import existing_dir

logger = get_logger("COMPARE_WITH_GSM")


def compare_result_with_gsm(result: dict) -> dict:
    logger.info(f"Comparing {result['results_file']}")

    original_automaton: MooreMachine = load_automaton_from_file(result['original_automaton'], "moore")

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
    gsm_mm_nopurge: MooreMachine = run_gsm_baseline(**gsm_kwargs, purge_mismatches=False)
    gsm_stats_nopurge = calculate_statistics(original_automaton, gsm_mm_nopurge)
    gsm_stats_nopurge["learned_automaton_size"] = gsm_mm_nopurge.size
    gsm_stats_nopurge["bisimilar"] = bisimilar(original_automaton, gsm_mm_nopurge)

    logger.info(f"Running GSM with purge_mismatches...")
    gsm_mm_purge: MooreMachine = run_gsm_baseline(**gsm_kwargs, purge_mismatches=True)
    gsm_stats_purge = calculate_statistics(original_automaton, gsm_mm_purge)
    gsm_stats_purge["learned_automaton_size"] = gsm_mm_purge.size
    gsm_stats_purge["bisimilar"] = bisimilar(original_automaton, gsm_mm_purge)

    def _relevant(stats: dict) -> dict:
        # remove other stats/things we don't care about
        return {
            "bisimilar": stats["bisimilar"],
            "Precision": stats["Precision"],
            "Recall": stats["Recall"],
            "F-Score": stats["F-Score"],
            "num_states": stats["learned_automaton_size"],
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