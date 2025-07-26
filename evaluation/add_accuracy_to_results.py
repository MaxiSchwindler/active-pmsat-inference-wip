import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from aalpy import bisimilar
from aalpy.automata import MooreMachine
from aalpy.utils import load_automaton_from_file
import tempfile

from evaluation import charts
from evaluation.utils import new_file


ACCURACY_KEY = "Accuracy"
INPUT_COMPLETENESS_KEY = "learned_model_input_complete"
INPUT_COMPLETENESS_KEY_GT = "ground_truth_input_complete"

NUM_SAMPLES = 100_000
RESET_PROB = 0.09


def compute_accuracy(
    automaton_a: MooreMachine | None,
    automaton_b: MooreMachine | None,
    num_samples: int = NUM_SAMPLES,
    reset_prob: float | None = RESET_PROB,
) -> float:
    if reset_prob <= 0:
        raise ValueError("Reset Probability should be > 0")

    if automaton_b is None or automaton_a is None:
        return 0.

    inputs_a = sorted(set(automaton_a.get_input_alphabet()))
    inputs_b = sorted(set(automaton_b.get_input_alphabet()))

    if inputs_a != inputs_b:
        raise ValueError("Only automaton with the same input alphabet can be compared")

    if bisimilar(automaton_a, automaton_b):
        return 1.

    random_words = []
    for _ in range(num_samples):
        random_words.append([])
        while True:
            inp = random.choice(inputs_a)
            random_words[-1].append(inp)
            if random.random() < reset_prob:
                break

    assert len(random_words) == num_samples

    correct = 0
    for random_word in random_words:
        out_a = automaton_a.execute_sequence(automaton_a.initial_state, random_word)
        out_b = automaton_b.execute_sequence(automaton_b.initial_state, random_word)
        if out_a == out_b:
            correct += 1

    return correct / num_samples


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


def load_results(results_dir: str, is_server_results: bool) -> list[dict]:
    print("loading results...")
    results = charts.load_results(results_dir, remove_traces_used_to_learn=True, is_server_results=is_server_results)
    print(f"loaded {len(results)} results.")
    return results


def write_result(result: dict, new_results_dir: str):
    orig_filename = Path(result["results_file"])
    del result["results_file"]

    with open(Path(new_results_dir) / orig_filename.name, "w") as f:
        json.dump(result, f, indent=4)


def add_to_results(results_dir: str, new_results_dir: str, is_server_results: bool, to_add: tuple[str] = (ACCURACY_KEY, INPUT_COMPLETENESS_KEY)):
    results = load_results(results_dir, is_server_results=is_server_results)

    for result in tqdm(results):
        if ACCURACY_KEY in result:
            raise ValueError("Accuracy already calculated! Running on wrong directory?")

        ground_truth = load_automaton_from_file(result["original_automaton"], "moore")
        learned_model = load_automaton_from_string(result["learned_model"])

        if ACCURACY_KEY in to_add:
            result[ACCURACY_KEY] = compute_accuracy(automaton_a=ground_truth, automaton_b=learned_model, num_samples=NUM_SAMPLES, reset_prob=RESET_PROB)

        if INPUT_COMPLETENESS_KEY in to_add:
            result[INPUT_COMPLETENESS_KEY] = learned_model.is_input_complete()
            result[INPUT_COMPLETENESS_KEY_GT] = ground_truth.is_input_complete()

        write_result(result, new_results_dir)


def main():
    parser = argparse.ArgumentParser(description='Calculate & add Accuracy to results. This may take long.')
    parser.add_argument("--existing_results_dir", type=str, required=True)
    parser.add_argument("--new_results_dir", type=str, required=True)
    parser.add_argument("--is_server_results", type=bool, default=False)
    parser.add_argument("--calculate_input_completeness", type=bool, default=True)

    args = parser.parse_args()

    results_dir = args.existing_results_dir
    new_results_dir = new_file(args.new_results_dir)
    os.makedirs(new_results_dir, exist_ok=False)

    to_add = (ACCURACY_KEY, INPUT_COMPLETENESS_KEY) if args.calculate_input_completeness else (ACCURACY_KEY, )

    add_to_results(results_dir, new_results_dir, is_server_results=args.is_server_results, to_add=to_add)


if __name__ == "__main__":
    main()