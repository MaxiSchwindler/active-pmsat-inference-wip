import json
import os.path
import sys
from collections.abc import Sequence
from datetime import datetime

import aalpy
from aalpy import run_KV
from aalpy.SULs import MooreSUL
from aalpy.utils import bisimilar

from active_pmsatlearn.learnalgo import run_activePmSATLearn
from active_pmsatlearn.oracles import RobustRandomWalkEqOracle, RobustPerfectMooreEqOracle

from evaluation.learn_automata import compute_accuracy
from evaluation.robust_sul_wrapper import RobustSUL
from evaluation.utils import GlitchingSUL


def get_sul_from_file(file: str, glitch_percent: float = 0):
    mm = aalpy.utils.load_automaton_from_file(file, "moore")
    sul = MooreSUL(mm)
    if glitch_percent:
        sul = GlitchingSUL(sul, glitch_percent)

    sul = RobustSUL(sul)
    return sul


def get_oracle(oracle_name: str, sul):
    if oracle_name == "RandomWithoutAbort":
        return RobustRandomWalkEqOracle(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            num_steps=sul.automaton.size * 5_000,
            reset_after_cex=True,
            reset_prob=0.09,
            max_num_tries=None
        )
    raise NotImplementedError(f"No robust oracle {oracle_name} implemented")


def main():
    args = sys.argv[1:]
    if len(args) not in (1, 2):
        print("USAGE: python simple_run_apmsl_on_file_or_dir.py FILE_OR_DIR [GLITCH_PERCENT]")
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

        print(f"Learning {file}...")
        learned_model, info = run_KV(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            automaton_type="moore",
            print_level=2,
            return_data=True,
            eq_oracle=get_oracle("RandomWithoutAbort", sul),
            cache_and_non_det_check=False
        )

        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # with open(f"{now}_learning_results_{os.path.basename(file)}.json", "w") as f:
        #     json.dump(info, f, indent=4)

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

            print("Calculate Accuracy...")
            acc = compute_accuracy(sul.automaton, learned_model)
            print(f"Accuracy: {acc}")

            print(f"Learning Steps: {info['steps_learning']}")
            print(f"EQ Oracle Steps: {info['steps_eq_oracle']}")
        else:
            print("Did not learn model.")


if __name__ == '__main__':
    main()