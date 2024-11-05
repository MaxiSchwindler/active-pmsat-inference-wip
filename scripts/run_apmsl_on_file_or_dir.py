import json
import os.path
import sys
from datetime import datetime

from aalpy.SULs import MooreSUL
from aalpy.utils import load_automaton_from_file

from active_pmsatlearn import run_activePmSATLearn
from evaluation.defs import oracles


def get_sul_from_file(file: str):
    mm = load_automaton_from_file(file, "moore")
    return MooreSUL(mm)


def get_oracle(oracle_name: str, sul):
    return oracles[oracle_name](sul)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("USAGE: python run_apmsl_on_file_or_dir.py FILE_OR_DIR")
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

    for file in dot_files:
        sul = get_sul_from_file(file)
        oracle = get_oracle("Perfect", sul)

        print(f"Learning {file}...")
        learned_model, info = run_activePmSATLearn(
            alphabet=oracle.alphabet,
            sul=sul,
            eq_oracle=oracle,
            automaton_type="moore",
            extension_length=2,
            pm_strategy="rc2",
            timeout=None,
            print_level=3,
            return_data=True,
            input_completeness_processing=True,
            cex_processing=True,
            glitch_processing=True,
        )

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{now}_learning_results_{os.path.basename(file)}.json", "w") as f:
            json.dump(info, f, indent=4)


if __name__ == '__main__':
    main()