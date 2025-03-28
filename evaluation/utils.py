import argparse
import builtins
import os
import itertools
import pathlib
import random
from collections import defaultdict
from pprint import pprint
from typing import Any, Iterable, Literal
from collections.abc import Sequence

import numpy as np
from aalpy.SULs import MealySUL, MooreSUL
from aalpy.automata import MooreMachine
from aalpy.base import SUL
from contextlib import contextmanager


def parse_range(value: str | int | range) -> range:
    """
    Parse an integer-range from a string or a single int.
    Format is "<start>-<stop>" or "<start>-<stop>-<step>".
    Note that other than a standard-python-range, <stop> is inclusive.
    If a single int is given, a range containing only this int is returned.
    """
    if isinstance(value, str):
        parts = value.split('-')
        if len(parts) == 3:
            start, end, step = map(int, parts)
            return range(start, end + 1, step)
        elif len(parts) == 2:
            start, end = map(int, parts)
            return range(start, end + 1)
        else:
            start, end = map(int, [value, value])
            return range(start, end + 1)
    elif isinstance(value, int):
        return range(value, value+1)
    elif isinstance(value, range):
        return value
    else:
        raise TypeError(f"Could not parse range: Unexpected type {type(value)}")


def format_range(value: range | Any) -> str:
    if isinstance(value, range):
        return f"{value.start}-{value.stop-1}" + (f"-{value.step}" if value.step != 1 else "")
    return str(value)


def new_file(path: str) -> str | None:
    if not path:
        return

    abspath = os.path.abspath(path)
    path, ext = os.path.splitext(path)

    try:
        path_, existing_index = path.rsplit("_", 1)
        existing_index = int(existing_index)
        path = path_
    except:
        existing_index = 0

    i = existing_index + 1

    while os.path.exists(abspath):
        if i == 1:
            print(f"'{path}' already exists")

        abspath = os.path.abspath(f"{path}_{i}{ext}")
        i += 1
    return abspath


def existing_dir(path) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")
    return path



def get_user_choices(message: str, choices: Sequence[str]) -> Sequence[str]:
    while True:
        print(message + ". Choices: " + ", ".join(choices))

        user_input = input("Enter your choice(s): ").strip()
        selected_choices = [choice.strip() for choice in user_input.split(',')]
        if all(choice in choices for choice in selected_choices):
            return selected_choices

        print("Invalid choice(s) entered, please try again.\n")


class MaxStepsReached(Exception):
    pass


class TracedMooreSUL(MooreSUL):
    def __init__(self, mm: MooreMachine) -> None:
        super().__init__(mm)
        self.current_trace = [self.automaton.step(None)]
        self.traces = [self.current_trace]

        self.taken_transitions = {}
        for s1 in mm.states:
            self.taken_transitions[s1.state_id] = {}
            for letter in mm.get_input_alphabet():
                self.taken_transitions[s1.state_id][letter] = {}
                for s2 in mm.states:
                    self.taken_transitions[s1.state_id][letter][s2.state_id] = 0

    def pre(self):
        super().pre()
        assert self.automaton.current_state == self.automaton.initial_state
        if len(self.current_trace) > 1:
            new_trace = [self.automaton.step(None)]
            self.traces.append(new_trace)
            self.current_trace = new_trace

    def step(self, letter=None):
        old_state = self.automaton.current_state
        output = super().step(letter)
        current_state = self.automaton.current_state
        if letter is None:
            return output
        entry = (letter, output)
        self.traces[-1].append(entry)
        self.taken_transitions[old_state.state_id][letter][current_state.state_id] += 1
        return output

    @staticmethod
    def flatten_transitions_dict(transitions: dict[str, dict[str, dict[str, int]]]) -> dict[tuple[str, str, str], int]:
        transitions_as_tuple_dict = {}
        for s1 in transitions:
            for l in transitions[s1]:
                for s2 in transitions[s1][l]:
                    transitions_as_tuple_dict[(s1, l, s2)] = transitions[s1][l][s2]
        return transitions_as_tuple_dict


class SULWrapper(SUL):
    def __init__(self, sul):
        super().__init__()
        self.sul = sul

    def step(self, letter):
        return self.sul.step(letter)

    def pre(self):
        return self.sul.pre()

    def post(self):
        return self.sul.post()

    def __getattr__(self, item):
        return getattr(self.sul, item)

    # def __setattr__(self, key, value):
    #     setattr(self.sul, key, value)


class MaxStepsSUL(SULWrapper):
    """ Restricts the wrapped SUL to max_num_steps. After reaching this threshold, a MaxStepsReached exception is raised. """
    def __init__(self, sul: MealySUL | MooreSUL, max_num_steps: int):
        super().__init__(sul)
        self.max_num_steps = max_num_steps

    def step(self, letter):
        # TODO: i think num_steps is only increased in query for some reason, so this won't work during a query
        if self.sul.num_steps >= self.max_num_steps:
            raise MaxStepsReached
        return self.sul.step(letter)


class GlitchingSUL(SULWrapper):
    """ Makes the wrapped SUL glitch a certain percentage of the time"""

    FAULT_TYPES = ("enter_random_state", )

    def __init__(self,
                 sul: TracedMooreSUL | MooreSUL,
                 glitch_percentage: float,
                 fault_type: str = 'enter_random_state'):
        super().__init__(sul)
        self.glitch_percentage = glitch_percentage
        assert fault_type in self.FAULT_TYPES
        self.fault_type = fault_type
        self.num_glitched_steps = 0

    def step(self, input):
        if input is None:
            return self.sul.step(input)

        if random.random() > (self.glitch_percentage / 100):
            # no glitch
            return self.sul.step(input)

        self.num_glitched_steps += 1
        if self.fault_type == "enter_random_state":
            assert len(states := self.sul.automaton.states) > 1
            current_state = self.sul.automaton.current_state
            while (wrong_next_state := random.choice(states)) == current_state.transitions[input]:
                pass
            fault_output = wrong_next_state.output
            if isinstance(self.sul, TracedMooreSUL):
                self.traces[-1].append((input, fault_output))
                self.taken_transitions[current_state.state_id][input][wrong_next_state.state_id] += 1
            self.sul.automaton.current_state = wrong_next_state
            # self.sul.num_steps += 1  # already done in sul.query with len(word)
            return fault_output
        else:
            raise TypeError(f"Unsupported fault type '{self.fault_type}'")

    @contextmanager
    def dont_glitch(self):
        prev = self.glitch_percentage
        try:
            self.glitch_percentage = 0
            yield
        finally:
            self.glitch_percentage = prev

    def __repr__(self):
        return f"{type(self).__name__}(glitch_percentage={self.glitch_percentage}, fault_type='{self.fault_type}')"


def dict_product(options) -> Iterable[dict]:
    """
    >>> list(dict_product({'number': [1, 2], 'character': 'ab'}))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(options.keys(), x)) for x in itertools.product(*options.values()))


def is_builtin_type(obj: Any) -> bool:
    return type(obj).__name__ in dir(builtins)


def get_args_from_input(parser):
    args_dict = {}
    for action in parser._actions:
        if action.dest in ('help', 'explain'):
            continue

        arg_name = action.dest
        default_value = action.default
        arg_type = action.type if action.type else str
        required = action.required
        nargs = action.nargs == '+'

        prompt = f"Enter value{'s' if nargs else ''} for {arg_name}{' (space-seperated)' if nargs else ''}"
        if default_value is not None:
            prompt += f" (default: {default_value})"
        if required:
            prompt += " [required]"

        prompt += ": "

        while True:
            user_input = input(prompt)

            # Use default if input is empty
            if not user_input.strip() and not required:
                args_dict[arg_name] = default_value
                break

            try:
                if nargs:
                    args_dict[arg_name] = [arg_type(value) for value in user_input.split()]
                else:
                    args_dict[arg_name] = arg_type(user_input)
                break
            except Exception as e:
                print(f"Invalid value for {arg_name}: {e}")

    return argparse.Namespace(**args_dict)


def print_results_info(results: list[dict]):
    def sep():
        print("-" * 40)

    algorithms = list(set([entry["algorithm_name"] for entry in results]))
    valid_results = [entry for entry in results if "exception" not in entry]
    results_with_model = [entry for entry in valid_results if entry["learned_automaton_size"] != 0 ]

    sep()
    print("RESULTS:")
    sep()
    print(f"Algorithms: {algorithms}")
    print(f"Oracles: {list(set(r['oracle'] for r in results))}")
    sep()

    num_automata = len(set(r['original_automaton'] for r in results))
    min_num_states = min(r['original_automaton_size'] for r in results)
    max_num_states = max(r['original_automaton_size'] for r in results)
    min_num_inputs = min(r['original_automaton_num_inputs'] for r in results)
    max_num_inputs = max(r['original_automaton_num_inputs'] for r in results)
    min_num_outputs = min(r['original_automaton_num_outputs'] for r in results)
    max_num_outputs = max(r['original_automaton_num_outputs'] for r in results)

    print(f"Unique automata: {num_automata}")
    print(f"Number of states per automaton: {min_num_states} to {max_num_states}")
    print(f"Number of inputs per automaton: {min_num_inputs} to {max_num_inputs}")
    print(f"Number of outputs per automaton: {min_num_outputs} to {max_num_outputs}")
    sep()

    num_results = len(results)
    num_valid = len(valid_results)
    num_learned = sum(r['learned_correctly'] for r in results)
    num_timed_out = sum(r['timed_out'] for r in valid_results)
    num_bisimilar = sum(r['bisimilar'] for r in results)
    num_returned_model = sum(r['learned_automaton_size'] is not None for r in valid_results)

    print(f"Valid results (no exceptions): {num_valid} of {num_results} ({num_valid / num_results * 100:.2f}%)")
    if num_valid > 0:
        print(f"Returned models (did not return None): {num_returned_model} of {num_valid} ({num_returned_model / num_valid * 100:.2f}%)")
        print(f"Learned correctly: {num_learned} of {num_valid} ({num_learned / num_valid * 100:.2f}%)")
        print(f"Bisimilar: {num_bisimilar} of {num_valid} ({num_bisimilar / num_valid * 100:.2f}%)")
    else:
        print(f"No returned models!")

    if num_timed_out:
        num_timed_out_but_correct = sum(r['timed_out'] and r['learned_correctly'] for r in valid_results)
        num_timed_out_but_bisimilar = sum(r['timed_out'] and r['bisimilar'] for r in valid_results)
        print(f"Timed out: {num_timed_out} of {num_results} ({num_timed_out / num_results * 100:.2f}%)")
        print(f"  - Nevertheless correct: {num_timed_out_but_correct}")
        print(f"  - Nevertheless bisimilar: {num_timed_out_but_bisimilar}")
    sep()

    print("Statistics:")
    for stat in ["Precision", "Recall", "F-Score", "Precision (all steps)", "Precision (all traces)",
                 "Precision per trace (mean)", "Precision per trace (median)",
                 "Strong accuracy (mean)","Strong accuracy (median)",
                 "Medium accuracy (mean)","Medium accuracy (median)",
                 "Weak accuracy (mean)","Weak accuracy (median)"]:
        print(f"  {stat}: ")
        for stat_method in (np.mean, np.median, np.min, np.max):
            val_all = stat_method([r[stat] for r in results])
            val_corr = stat_method([r[stat] for r in (res for res in valid_results if res["learned_correctly"])] or [0])
            val_timed = stat_method([r[stat] for r in (res for res in valid_results if res["timed_out"])] or [0])
            val_not_timed = stat_method([r[stat] for r in (res for res in valid_results if not res["timed_out"])] or [0])
            print(f"    {stat_method.__name__:6}: {val_all:.2f} (with TO: {val_timed:.2f} | no TO: {val_not_timed:.2f})")
    sep()

    glitch_percent = set(r.get('glitch_percent', None) for r in results)
    # assert len(glitch_percent) == 1
    # glitch_percent = list(glitch_percent)[0]

    if glitch_percent:
        print(f"Glitch percentage: {glitch_percent}%")
        print("Fault type: 'enter_random_state'")
        sep()

    exceptions = [r['exception'] for r in results if "exception" in r]
    unique_exceptions = list(set(exceptions))
    num_exceptions = len(exceptions)

    if num_exceptions:
        print(f"Exceptions: {num_exceptions}")
        print(f"Unique exceptions: {len(unique_exceptions)}")

    for e in unique_exceptions:
        print("\t", e)
        num_times = exceptions.count(e)
        perc_of_exc = num_times / num_exceptions * 100
        perc_of_runs = num_times / num_results * 100
        unique_automata = list(set(r['original_automaton'] for r in results if r.get('exception', None) == e))
        unique_algorithms = list(set(r['algorithm_name'] for r in results if r.get('exception', None) == e))

        print("\t\t",
              f"occurred {num_times} times ({perc_of_exc:.2f}% of all exceptions / {perc_of_runs:.2f}% of all runs)")
        print("\t\t", f"occurred in {len(unique_automata)} unique automata (", end="")

        every_time = []
        for a in unique_automata:
            runs_with_a = sum(1 for r in results if r['original_automaton'] == a)
            runs_with_a_and_e = sum(
                1 for r in results if (r['original_automaton'] == a) and (r.get('exception', None) == e))
            print(f"{runs_with_a_and_e}/{runs_with_a}", end="")
            if a != unique_automata[-1]:
                print(", ", end="")
            else:
                print(')')
            if runs_with_a == runs_with_a_and_e:
                every_time.append(a)

        for a in every_time:
            print("\t\t\t", f"occurred every time with {a}")

        print("\t\t", f"occurred in {len(unique_algorithms)} unique algorithms: {unique_algorithms}")


def print_results_info_per_alg(results: list[dict], exclude_results_with_eq_oracle=False):
    def sep():
        print("-" * 40)

    algorithms = sorted(list(set([entry["algorithm_name"] for entry in results])))
    valid_results = [result for result in results if "exception" not in result]
    if exclude_results_with_eq_oracle:
        valid_results = [result for result in valid_results if result["oracle"] in ("None", None)]

    sep()
    print("ALGORITHM RESULTS")
    sep()

    for algorithm in algorithms:
        print(f"  {algorithm}")
        alg_results = [result for result in valid_results if result["algorithm_name"] == algorithm]

        print("    Overall Statistics:")
        for stat in ["learned_correctly", "Precision", "Recall", "F-Score", "Precision (all traces)",]:
            val = np.mean([r[stat] for r in alg_results])
            print(f"      {stat}: {val:.2f}")

        print("    Per Glitch Percentage:")
        for glitch_percent in sorted(list(set(r["glitch_percent"] for r in alg_results))):
            glitch_results = [result for result in alg_results if result["glitch_percent"] == glitch_percent]
            print(f"      {glitch_percent}%: ")
            for stat in ["learned_correctly", "Precision", "Recall", "F-Score", "Precision (all traces)",]:
                val = np.mean([r[stat] for r in glitch_results])
                print(f"        {stat}: {val:.2f}")
        sep()