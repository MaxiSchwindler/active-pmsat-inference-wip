import argparse
import builtins
import os
import itertools
import random
from collections import defaultdict
from pprint import pprint
from typing import Any, Iterable, Literal
from collections.abc import Sequence

from aalpy.SULs import MealySUL, MooreSUL
from aalpy.automata import MooreMachine
from aalpy.base import SUL
from contextlib import contextmanager


def parse_range(value: str | int | range) -> range:
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

    def pre(self):
        super().pre()
        assert self.automaton.current_state == self.automaton.initial_state
        if len(self.current_trace) > 1:
            new_trace = [self.automaton.step(None)]
            self.traces.append(new_trace)
            self.current_trace = new_trace

    def step(self, input):
        output = super().step(input)
        if input is None:
            return output
        entry = (input, output)
        self.traces[-1].append(entry)
        return output


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


class MaxStepsSUL(SULWrapper):
    """ Restricts the wrapped SUL to max_num_steps. After reaching this threshold, a MaxStepsReached exception is raised. """
    def __init__(self, sul: MealySUL | MooreSUL, max_num_steps: int):
        super().__init__(sul)
        self.max_num_steps = max_num_steps

    def step(self, letter):
        if self.sul.num_steps >= self.max_num_steps:
            raise MaxStepsReached
        return self.sul.step(letter)


class GlitchingSUL(SULWrapper):
    """ Makes the wrapped SUL glitch a certain percentage of the time"""
    def __init__(self,
                 sul: TracedMooreSUL | MooreSUL,
                 glitch_percentage: float,
                 fault_type: Literal["send_random_input", "enter_random_state", "discard_io_tuple"] = 'send_random_input'):
        super().__init__(sul)
        self.glitch_percentage = glitch_percentage
        self.fault_type = fault_type
        self.num_glitched_steps = 0

    def step(self, input):
        if input is None:
            return self.sul.step(input)

        if random.random() > (self.glitch_percentage / 100):
            # no glitch
            return self.sul.step(input)

        self.num_glitched_steps += 1
        match self.fault_type:
            case "send_random_input":
                assert len(alphabet := self.sul.automaton.get_input_alphabet()) > 1
                while (fault_input := random.choice(alphabet)) == input:
                    pass  # don't choose the same input as fault input, otherwise the actual glitch percentage drops
                return self.sul.step(fault_input)

            case "enter_random_state":
                assert len(states := self.sul.automaton.states) > 1
                while (fault_state := random.choice(states)) == self.sul.automaton.current_state:
                    pass
                fault_output = fault_state.output
                self.sul.current_state = fault_state
                if isinstance(self.sul, TracedMooreSUL):
                    self.traces[-1].append((input, fault_output))
                return fault_output

            case "discard_io_tuple":
                # this fault mode probably does not make a lot of sense in an active context...
                # since the algorithm knows what inputs were sent, losing an entire (i,o) doesn't make much sense.
                assert isinstance(self.sul, TracedMooreSUL), ("SUL must be traced, and all outputs must be "
                                                              "received via .traces or .current_trace for "
                                                              "glitching with fault_mode 'discard_io_tuple' to work.")
                output = self.sul.step(input)
                self.traces[-1].pop()
                return output

            case _other:
                raise TypeError(f"Unsupported fault type '{_other}'")

    @contextmanager
    def dont_glitch(self):
        prev = self.glitch_percentage
        try:
            self.glitch_percentage = 0
            yield
        finally:
            self.glitch_percentage = prev



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
