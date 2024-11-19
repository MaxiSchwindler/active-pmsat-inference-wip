import sys
import os
import argparse
import uuid
from os import PathLike
from typing import Literal, Any
from itertools import takewhile

from aalpy.automata import MealyMachine, MooreMachine
from aalpy.base import Automaton
from aalpy.utils import save_automaton_to_file, generate_random_deterministic_automata

from evaluation.utils import *
SUPPORTED_AUTOMATA_TYPES = ("mealy", "moore")


def generate_automaton_filename(automaton: Automaton, num_states: int, num_inputs: int, num_outputs: int):
    """ Generate a descriptive filename with the given params"""
    automaton_type = type(automaton).__name__
    assert len(automaton.states) == num_states
    assert len(automaton.get_input_alphabet()) == num_inputs

    return f"{automaton_type}_{num_states}States_{num_inputs}Inputs_{num_outputs}Outputs_{uuid.uuid4().hex}"


def parse_automaton_filename(filename: str, print_error: bool = False) -> dict[str, str | int] | None:
    try:
        def get_leading_int(string):
            leading_digits = ''.join(takewhile(str.isdigit, string))
            return int(leading_digits) if leading_digits else None

        filename = os.path.basename(filename)
        a_type, num_states_str, num_inputs_str, num_outputs_str, a_hash = filename.split("_")

        info = dict()
        if a_type in ("MooreMachine", "MealyMachine"):
            info["automaton_type"] = "moore" if a_type.startswith("Moore") else "mealy"
        else:
            raise ValueError(f"Unknown automaton type {a_type}")

        def _raise(msg):
            raise ValueError(msg)

        info["num_states"] = get_leading_int(num_states_str) or _raise("Could not parse number of states")
        info["num_inputs"] = get_leading_int(num_inputs_str) or _raise("Could not parse number of inputs")
        info["num_outputs"] = get_leading_int(num_outputs_str) or _raise("Could not parse number of outputs")
        info["hash"] = a_hash
        return info
    except ValueError as e:
        if print_error:
            print(repr(e))
        return None


def is_valid_automata_configuration(automaton_type: Literal["mealy", "moore"],
                                    num_states: int, num_inputs: int, num_outputs: int,
                                    _print_reason: bool = False) -> bool:
    if automaton_type == "moore" and num_outputs > num_states:
        if _print_reason:
            print(f"Cannot generate a Moore Machine with more outputs ({num_outputs}) than states ({num_states})")
        return False
    if automaton_type == "mealy" and num_outputs > num_states * num_inputs:
        if _print_reason:
            print(f"Cannot generate a Mealy Machine with more outputs ({num_outputs}) than inputs times states "
                  f"({num_inputs}*{num_states}={num_inputs*num_states})")
        return False
    return True


def generate_automaton(automaton_type: Literal["mealy", "moore"], num_states: int, num_inputs: int, num_outputs: int,
                       destination_folder: str) -> tuple[MealyMachine | MooreMachine, str] | tuple[None, None]:
    """
    Generates an automaton of the given type with the given parameters, saves it to destination folder, and returns
    a tuple of automaton and file path.
    If the parameters are invalid (e.g. more outputs than states in a Moore Machine), None is returned.
    """

    if not is_valid_automata_configuration(automaton_type, num_states, num_inputs, num_outputs, _print_reason=True):
        return None, None

    automaton: MealyMachine | MooreMachine = generate_random_deterministic_automata(automaton_type,
                                                                                    num_states,
                                                                                    num_inputs,
                                                                                    num_outputs)

    os.makedirs(destination_folder, exist_ok=True)
    filename = generate_automaton_filename(automaton, num_states, num_inputs, num_outputs)
    path = os.path.join(os.path.abspath(destination_folder), filename)

    save_automaton_to_file(automaton, path, file_type='dot')

    return automaton, f"{path}.dot"


def generate_automata(automaton_type: Literal["mealy", "moore"],
                      num_states: int | range, num_inputs: int | range, num_outputs: int | range,
                      destination_folder: str) -> tuple[list[MealyMachine | MooreMachine], list[str]]:
    automata = []
    files = []
    for current_num_states in parse_range(num_states):
        for current_num_inputs in parse_range(num_inputs):
            for current_num_outputs in parse_range(num_outputs):
                automaton, file = generate_automaton(automaton_type,
                                                     current_num_states, current_num_inputs, current_num_outputs,
                                                     destination_folder)
                if automaton:
                    automata.append(automaton)
                    files.append(file)
    return automata, files


def matching_automata_files_in_directory(directory: str, automata_type: Literal["mealy", "moore"],
                                         num_states: int | None, num_inputs: int | None, num_outputs: int | None) -> list[str]:
    def match(given_val, filename_val):
        return given_val is None or given_val == filename_val

    if not os.path.isdir(directory):
        return []

    return [
        os.path.join(os.path.abspath(directory), filename) for filename in os.listdir(directory)
        if (info := parse_automaton_filename(filename))
        and info["automaton_type"] == automata_type
        and match(info["num_states"], num_states)
        and match(info["num_inputs"], num_inputs)
        and match(info["num_outputs"], num_outputs)
    ]


def add_automata_generation_arguments(parser: argparse.ArgumentParser, learn=False):
    parser.add_argument('-n', '--num_automata_per_combination', type=int, default=1,
                        help=f'Number of automata *of each combination* to {"generate" if not learn else "learn"}.'
                             f'This means that a unique automaton is {"generated" if not learn else "learned"} '
                             f'num times for every combination of num_states/num_inputs/num_outputs, '
                             f'for a maximum of (num * num_states * num_inputs * num_outputs) '
                             f'automata. Invalid combinations (more outputs than possible) are ignored.')
    parser.add_argument('-t', '--type', type=str, choices=SUPPORTED_AUTOMATA_TYPES, default='moore',
                        help=f'Type of automata to {"generate" if not learn else "learn"}')
    parser.add_argument('-ns', '--num_states', type=parse_range, required=True,
                        help='Number of states per automaton. Can be either a single number or a range (e.g. 1-10)')
    parser.add_argument('-ni', '--num_inputs', type=parse_range, required=True,
                        help='Number of inputs per automaton (size of the input alphabet). '
                             'Can be either a single number or a range (e.g. 1-10)')
    parser.add_argument('-no', '--num_outputs', type=parse_range, required=True,
                        help='Number of outputs per automaton (size of the output alphabet). '
                             'Can be either a single number or a range (e.g. 1-10)')
    parser.add_argument('--generated_automata_dir', type=str, default="generated_automata",
                        help='Directory to store newly generated automata in.')


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Generate automata. You can also run interactively.')
        add_automata_generation_arguments(parser)
        args = parser.parse_args()

        num_automata_per_combination = args.num_automata_per_combination
        automata_type = args.type
        num_states = args.num_states
        num_inputs = args.num_inputs
        num_outputs = args.num_outputs
        destination_folder = args.generated_automata_dir
    else:
        num_automata_per_combination = int(input("Enter number of automata to generate: "))
        automata_type = input("Enter type of automata to generate: ")
        if automata_type not in SUPPORTED_AUTOMATA_TYPES:
            raise TypeError(f"Unsupported type of automata: '{automata_type}'")
        num_states = parse_range(input("Enter number of states per automaton: "))
        num_inputs = parse_range(input("Enter number of inputs per automaton: "))
        num_outputs = parse_range(input("Enter number of outputs per automaton: "))
        destination_folder = input("Enter directory to store generated automata in: ") or "generated_automata"

    for i in range(num_automata_per_combination):
        generate_automata(automata_type, num_states, num_inputs, num_outputs, destination_folder)


if __name__ == '__main__':
    main()