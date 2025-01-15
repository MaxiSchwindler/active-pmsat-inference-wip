import json
import os.path
import sys
from collections.abc import Sequence
from datetime import datetime

from aalpy.SULs import MooreSUL
from aalpy.utils import load_automaton_from_file
from aalpy.utils import bisimilar

from active_pmsatlearn.defs import EqOracleTermination
from active_pmsatlearn.learnalgo import run_activePmSATLearn
from evaluation.defs import oracles
from evaluation.learn_automata import setup_sul, calculate_statistics


def get_sul_from_file(file: str, glitch_percent: float = 0):
    mm = load_automaton_from_file(file, "moore")
    return setup_sul(mm, glitch_percent=glitch_percent)


def get_oracle(oracle_name: str, sul):
    return oracles[oracle_name](sul)


def main():
    args = sys.argv[1:]
    if len(args) not in (1, 2):
        print("USAGE: python run_apmsl_on_file_or_dir.py FILE_OR_DIR [GLITCH_PERCENT] [LEARNING_ALG_KWARGS]")
        exit(1)

    file_or_dir = args[0]
    if not os.path.exists(file_or_dir):
        print(f"{file_or_dir} does not exist")
        exit(2)

    files = [os.path.join(file_or_dir, f) for f in os.listdir(file_or_dir)] if os.path.isdir(file_or_dir) else [file_or_dir]
    dot_files = [f for f in files if f.endswith('.dot')]

    if len(dot_files) == 0:
        if os.path.isdir(file_or_dir):
            print(f"No .dot files found in {file_or_dir}")
        else:
            print(f"{file_or_dir} is not a .dot file")
        exit(3)

    glitch_percent = float(args[1]) if len(args) > 1 else 0.0

    for file in dot_files:
        sul = get_sul_from_file(file, glitch_percent)
        oracle = get_oracle("Random", sul)

        print(f"Learning {file}...")
        learned_model, info = run_activePmSATLearn(
            alphabet=oracle.alphabet,
            sul=sul,
            automaton_type="moore",
            extension_length=3,
            heuristic_function='intermediary',
            pm_strategy="rc2",
            timeout=200,
            print_level=2,
            return_data=True,
            window_cex_processing=False,
            termination_mode=EqOracleTermination(get_oracle('Random', sul))
            # input_completeness_preprocessing=False,
            # glitch_processing=False,
            # random_state_exploration=True,
            # discard_glitched_traces=False
        )

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{now}_learning_results_{os.path.basename(file)}.json", "w") as f:
            json.dump(info, f, indent=4)

        if learned_model:
            print(f"Original model: {len(sul.automaton.states)} states")
            print(f"Learned  model: {len(learned_model.states)} states")

            b = bisimilar(sul.automaton, learned_model, return_cex=True)
            if isinstance(b, Sequence):
                orig = sul.automaton.execute_sequence(sul.automaton.initial_state, b)
                learned = learned_model.execute_sequence(learned_model.initial_state, b)
                print(f"Bisimilar: False. Counterexample: {b}. Output original: {orig}. Output learned: {learned}")
            else:
                print(f"Bisimilar: {b is None}")

            print("Calculate statistics...")
            stats = calculate_statistics(sul.automaton, learned_model)
            max_len = len(max(stats.keys()))
            for k, v in stats.items():
                print(f"{k: <{max_len}}: {v}")

            print("Saving models...")
            learned_model.save("LearnedModel")
            sul.automaton.save("OriginalModel")
        else:
            print("Did not learn model.")


if __name__ == '__main__':
    main()