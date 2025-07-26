import argparse
import json
import os
from pathlib import Path

from aalpy.automata import MooreMachine
from aalpy.utils import load_automaton_from_file
import tempfile
import evaluation
import evaluation.learn_automata
from evaluation import charts
from evaluation.utils import new_file

HYPS_ACCS_KEY = "Hypotheses_Accuracy"


def load_automaton_from_string(automaton_string: str) -> MooreMachine:
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dot', delete=False) as tmp:
        tmp.write(automaton_string)
        tmp.flush()
        tmp_path = tmp.name

    try:
        return load_automaton_from_file(tmp_path, "moore")
    finally:
        os.remove(tmp_path)


def add_hyp_accuracies_to_results(results_dir: str, new_results_dir: str, is_server_results):
    print("loading results...")
    results = charts.load_results(results_dir, remove_traces_used_to_learn=True, is_server_results=is_server_results)
    print(f"loaded {len(results)} results.")

    for i, result in enumerate(results):
        print(f"Processing result {i+1}/{len(results)}")

        ground_truth = load_automaton_from_file(result["original_automaton"], "moore")
        for learning_round in range(1, result["learning_rounds"]+1):
            print(f"learning round {learning_round}", end="\r")
            learning_round_info = result["detailed_learning_info"][str(learning_round)]
            if HYPS_ACCS_KEY in learning_round_info:
                print("Score already calculated! Running on wrong directory?")
                return

            if result["algorithm_name"].startswith("APMSL"):
                acc_scores = {}
                for num_states, hyp in learning_round_info["hyp"].items():
                    if hyp is not None:
                        acc_scores[num_states] = evaluation.learn_automata.compute_accuracy(hyp, ground_truth)
                learning_round_info[HYPS_ACCS_KEY] = acc_scores

            elif result["algorithm_name"].startswith("GSM"):
                hyp = learning_round_info["hyp"]
                learning_round_info[HYPS_ACCS_KEY] = evaluation.learn_automata.compute_accuracy(hyp, ground_truth)

            else:
                raise ValueError(f"Unknown algorithm {result['algorithm_name']}")

        orig_filename = Path(result["results_file"])
        del result["results_file"]

        with open(Path(new_results_dir) / orig_filename.name, "w") as f:
            json.dump(result, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Calculate & add intermediary Accuracy-Scores to results. This may take long.')
    parser.add_argument("--existing_results_dir", type=str, required=True)
    parser.add_argument("--new_results_dir", type=str, required=True)
    parser.add_argument("--is_server_results", type=bool, default=False)

    args = parser.parse_args()

    results_dir = args.existing_results_dir
    new_results_dir = new_file(args.new_results_dir)
    os.makedirs(new_results_dir, exist_ok=False)

    add_hyp_accuracies_to_results(results_dir, new_results_dir, args.is_server_results)


if __name__ == "__main__":
    main()