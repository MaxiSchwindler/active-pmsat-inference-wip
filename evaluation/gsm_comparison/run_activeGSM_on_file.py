import json
import logging
import os
import argparse
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime

from aalpy.SULs import MooreSUL
from aalpy.utils import load_automaton_from_file
from aalpy.utils import bisimilar

import active_pmsatlearn.log
import evaluation.visualize_run
from active_pmsatlearn.defs import *
from active_pmsatlearn.learnalgo import run_activePmSATLearn
from evaluation.defs import oracles
from evaluation.gsm_comparison.active_gsm import run_activeGSM
from evaluation.learn_automata import setup_sul, calculate_statistics
from evaluation.utils import new_file


def get_sul_from_file(file: str, glitch_percent: float = 0):
    """Load the automaton from the file and set up the SUL with glitches if needed."""
    mm = load_automaton_from_file(file, "moore")
    return setup_sul(mm, glitch_percent=glitch_percent, glitch_mode="enter_random_state")


def get_oracle(oracle_name: str, sul):
    """Return the oracle based on the given name."""
    return oracles[oracle_name](sul)


def run_active_gsm(sul, oracle, failure_rate):
    """Run active GSM on the given file and return the learned model and info."""
    return run_activeGSM(
        alphabet=sul.automaton.get_input_alphabet(),
        sul=sul,
        automaton_type="moore",
        failure_rate=failure_rate,
        certainty=0.05,
        eq_oracle=oracle,
        extension_length=3,
        timeout=60,
        print_level=2,
        return_data=True,
    )


def save_learning_results(file: str, info):
    """Save the learning results as a JSON file."""
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{now}_learning_results_{os.path.basename(file)}.json"
    with open(filename, "w") as f:
        json.dump(info, f, indent=4)
    return filename


def compare_learned_model_with_original(original_model, learned_model):
    """Compare the original and learned models, print bisimilarity and statistics."""
    print(f"Original model: {len(original_model.states)} states")
    print(f"Learned  model: {len(learned_model.states)} states")

    b = bisimilar(original_model, learned_model, return_cex=True)
    if isinstance(b, Sequence):
        orig = original_model.execute_sequence(original_model.initial_state, b)
        learned = learned_model.execute_sequence(learned_model.initial_state, b)
        print(f"Bisimilar: False. Counterexample: {b}. Output original: {orig}. Output learned: {learned}")
    else:
        print(f"Bisimilar: {b is None}")

    print("Calculate statistics...")
    stats = calculate_statistics(original_model, learned_model)
    max_len = len(max(stats.keys()))
    for k, v in stats.items():
        print(f"{k: <{max_len}}: {v}")


def save_models(original_model, learned_model):
    """Save the learned and original models."""
    print("Saving models...")
    learned_model.save("LearnedModel")
    original_model.save("OriginalModel")


@contextmanager
def log_to_file(file_path):
    logger = active_pmsatlearn.log.get_logger("APMSL")
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler
    logger.addHandler(file_handler)
    try:
        yield
    finally:
        # Remove the file handler to clean up
        logger.removeHandler(file_handler)
        file_handler.close()


def learn_file(file: str, glitch_percent: float, oracle_name: str = None, log_file: str = None):
    sul = get_sul_from_file(file, glitch_percent)
    oracle = get_oracle(oracle_name, sul)

    print(f"Learning {file} with SUL {sul} and oracle {oracle}")
    cm = log_to_file(log_file) if log_file is not None else nullcontext()
    with cm:
        learned_model, info = run_active_gsm(sul, oracle, failure_rate=glitch_percent / 100)

    results_file = save_learning_results(file, info)

    if learned_model:
        compare_learned_model_with_original(sul.automaton, learned_model)
        save_models(sul.automaton, learned_model)
    else:
        print("Did not learn model.")

    return learned_model, info, results_file


def main():
    parser = argparse.ArgumentParser(description="Learn Moore machines from .dot files.")
    parser.add_argument("file", help="Path to a .dot file")
    parser.add_argument("--glitch_percent", type=float, default=0.0,
                        help="Percentage of glitches to simulate in the learning process (default is 0.0).")
    parser.add_argument("--oracle", type=str, default='None', choices=oracles,
                        help="Name of equality oracle")
    parser.add_argument("--log-file", type=str, default=None, help="Write log messages to this file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"{args.file} does not exist")
        exit(2)
    if not args.file.endswith(".dot"):
        print(f"{args.file} is not a .dot file")
        exit(3)

    log_file = args.log_file
    if log_file is not None:
        log_file = new_file(log_file)

    learned_model, info, results_file = learn_file(args.file, args.glitch_percent, args.oracle, log_file)



if __name__ == '__main__':
    main()
