import argparse
import json
import os
import statistics
from pathlib import Path

from tqdm import tqdm

from aalpy.automata import MooreMachine
from aalpy.utils import load_automaton_from_file
import tempfile

from evaluation import charts
from evaluation.utils import new_file
from f_similarity import stochastic_conformance

# v2:
# CORRECT_PRECISION_NAME = "Precision_v2"
# CORRECT_RECALL_NAME = "Recall_v2"
# CORRECT_FSCORE_NAME = "F-Score_v2"
#
# NUM_SAMPLES = 100_000
# RESET_PROB = 0.09

# v3:
CORRECT_PRECISION_NAME = "Precision_v3"
CORRECT_RECALL_NAME = "Recall_v3"
CORRECT_FSCORE_NAME = "F-Score_v3"

NUM_SAMPLES = 100_000
RESET_PROB = 0.15


def load_automaton_from_string(automaton_string: str) -> MooreMachine | None:
    if automaton_string == "":
        return None

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dot', delete=False) as tmp:
        tmp.write(automaton_string)
        tmp.flush()
        tmp_path = tmp.name

    try:
        return load_automaton_from_file(tmp_path, "moore")
    finally:
        os.remove(tmp_path)


def calculate_correct_stats(learned_model: MooreMachine, ground_truth: MooreMachine):
    if learned_model is None:
        return 0.0

    precision = stochastic_conformance(automaton_generator=ground_truth, automaton_tester=learned_model, num_samples=NUM_SAMPLES, reset_prob=RESET_PROB)
    recall = stochastic_conformance(automaton_generator=learned_model, automaton_tester=ground_truth, num_samples=NUM_SAMPLES, reset_prob=RESET_PROB)
    f_score = statistics.harmonic_mean((precision, recall))
    return precision, recall, f_score


def add_correct_stats_to_results(results_dir: str, new_results_dir: str, is_server_results: bool):
    print("loading results...")
    results = charts.load_results(results_dir, remove_traces_used_to_learn=True, is_server_results=is_server_results)
    print(f"loaded {len(results)} results.")

    for result in tqdm(results):
        if CORRECT_FSCORE_NAME in result:
            print("Correct stats already calculated! Running on wrong directory?")
            return

        ground_truth = load_automaton_from_file(result["original_automaton"], "moore")
        learned_model = load_automaton_from_string(result["learned_model"])

        precision, recall, f_score = calculate_correct_stats(learned_model=learned_model, ground_truth=ground_truth)
        result[CORRECT_PRECISION_NAME] = precision
        result[CORRECT_RECALL_NAME] = recall
        result[CORRECT_FSCORE_NAME] = f_score

        orig_filename = Path(result["results_file"])
        del result["results_file"]

        with open(Path(new_results_dir) / orig_filename.name, "w") as f:
            json.dump(result, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Calculate & add correct F-Scores to results. This may take long.')
    parser.add_argument("--existing_results_dir", type=str, required=True)
    parser.add_argument("--new_results_dir", type=str, required=True)
    parser.add_argument("--is_server_results", type=bool, default=False)

    args = parser.parse_args()

    results_dir = args.existing_results_dir
    new_results_dir = new_file(args.new_results_dir)
    os.makedirs(new_results_dir, exist_ok=False)

    add_correct_stats_to_results(results_dir, new_results_dir, args.is_server_results)


if __name__ == "__main__":
    main()