import argparse
import re
import sys
from functools import partial
from dataclasses import dataclass
from inspect import signature, Signature, Parameter
from typing import Callable, Any

import aalpy.learning_algs
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.oracles import RandomWMethodEqOracle

import active_pmsatlearn.learnalgo
from active_pmsatlearn.learnalgo import run_activePmSATLearn
from active_pmsatlearn.oracles import RobustRandomWalkEqOracle, RobustPerfectMooreEqOracle
from evaluation.utils import dict_product

DEBUG_WRAPPER = True

SECOND = SECONDS = 1
MINUTE = MINUTES = 60
HOUR = HOURS = MINUTE * 60


class RobustPerfectMooreOracle(RobustPerfectMooreEqOracle):
    def __init__(self, sul: MooreSUL):
        super().__init__(sul)


class RobustRandomWalkOracle(RobustRandomWalkEqOracle):
    def __init__(self, sul: MooreSUL | MealySUL):
        super().__init__(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            num_steps=sul.automaton.size * 5_000,
            reset_after_cex=True,
            reset_prob=0.09
        )


oracles = {
    "None": lambda sul: None,
    "Perfect": RobustPerfectMooreOracle,
    "Random": RobustRandomWalkOracle,
}


class AlgorithmWrapper:
    function: Callable = lambda *args, **kwargs: None
    default_parameters: dict[str, Any] = {}
    positional_arguments: tuple[str] = ()
    aliases: dict[str, str] = {}
    shortcuts: dict[dict[str, Any]] = {}

    @classmethod
    def get_default_kwargs(cls):
        common_params = {}
        for c in reversed(cls.mro()):
            common_params.update(getattr(c, "default_parameters", {}))
        return common_params

    def __init__(self, *args, **kwargs):
        self.kwargs = self.get_default_kwargs().copy()
        self.kwargs |= kwargs

        for key, value in list(self.kwargs.items()):
            full_name = self.aliases.get(key, None)
            if full_name is not None:
                assert full_name not in self.kwargs, f"Specifying both kwarg '{full_name}' and its alias '{key}' at the same time is forbidden"
                self.kwargs.pop(key)
                self.kwargs[full_name] = value

        positional_args = []
        for a in args:
            if a in self.shortcuts:
                for k, v in self.shortcuts[a].items():
                    if k in self.kwargs and self.kwargs[k] != v:
                        raise ValueError(f"Key {k} is already in kwargs, but would be overwritten by shortcut {a}")
                    self.kwargs[k] = v
            else:
                positional_args.append(a)

        for key, value in zip(self.positional_arguments, positional_args):
            assert key not in self.kwargs, f"Key {key} is in kwargs, but {key} is also given as positional argument!"
            self.kwargs[key] = value

    def run(self, **kwargs):
        my_kwargs = self.kwargs.copy()
        my_kwargs |= kwargs
        if DEBUG_WRAPPER:
            print(f"Calling {self.function.__qualname__} with {my_kwargs}")
        try:
            return type(self).function(**my_kwargs)
        except Exception as e:
            print(f"Exception while calling {self.function.__qualname__} with {my_kwargs}: {e}")
            raise e


    def old__init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert cls.function is not None, f"AlgorithmWrapper must define class variable 'function'"

        # update __doc__ (for help())
        cls.__doc__ = cls.function.__doc__

        # update signature of __new__ (should theoretically help for type inference)
        func_sig = signature(cls.function)
        parameters = list(func_sig.parameters.values())

        cls.__new__.__signature__ = Signature(
            parameters=[
                Parameter("cls", Parameter.POSITIONAL_OR_KEYWORD),
                *parameters,
            ],
            return_annotation=func_sig.return_annotation,
        )

    @staticmethod
    def preprocess_algorithm_call(call: str) -> str:
        """Change the call APMSL(WCP) or APMSL(WCP, GP) to APMSL('WCP') or APMSL('WCP', 'GP')"""
        match = re.match(r"(\w+)\((.*)\)", call)
        if not match:
            return call + "()"

        cls_name, args = match.groups()
        cls = globals().get(cls_name)
        if not cls or not issubclass(cls, AlgorithmWrapper):
            return call

        def replacer(arg: str) -> str:
            if arg in cls.shortcuts:
                return f"'{arg}'"
            return arg

        args_list = [replacer(arg.strip()) for arg in args.split(',')]

        processed_args = ', '.join(args_list)
        return f"{cls_name}({processed_args})"

    @classmethod
    def all_implemented_algorithms(cls):
        def all_subclasses(cls):
            return set(cls.__subclasses__()).union(
                [s for c in cls.__subclasses__() for s in all_subclasses(c)])

        return [c.__name__ for c in all_subclasses(cls) if not c.__subclasses__()]

    @classmethod
    def validate_type(cls, string, add_help=True, should_raise=True):
        if string.split('(')[0] in cls.all_implemented_algorithms():
            return cls.preprocess_algorithm_call(string)
            # return cls.ArgparseWrapper(cls.preprocess_algorithm_call(string))
        if should_raise:
            raise argparse.ArgumentTypeError(f"'{string}' is not a valid algorithm. "
                                             f"Choose from {cls.all_implemented_algorithms()}. "
                                             + (f"You can add algorithm parameters in parentheses." if add_help else ""))
        return False

    def print_help(self, name=None):
        print(self.function.__doc__)
        print()

        print("Shortcuts:")
        for key, val in self.shortcuts.items():
            assert isinstance(val, dict)
            if len(val) == 1:
                print(f"  {key}: sets {list(val.keys())[0]} to {list(val.values())[0]}")
            else:
                print(f"  {key}: ")
                for k, v in val.items():
                    print(f"    sets {k} to {v}")
        print()

        print("Aliases:")
        for key, val in self.aliases.items():
            print(f"  {key}: {val}")

        if name:
            print()
            print(f"{name} will run the algorithm with: ")
            params = signature(self.function).parameters
            for param_name, param in params.items():
                if param_name in self.kwargs:
                    val = self.kwargs[param_name]
                else:
                    val = param.default

                val_str = val if not isinstance(val, str) else f"'{val}'"

                if val is not param.empty:
                    if val == param.default:
                        print(f" [default] {param_name}: {val_str}")
                    else:
                        print(f"           {param_name}: {val_str}")

    @property
    def unique_keywords(self):
        unique_keywords = {}
        params = signature(self.function).parameters
        for param_name, param in params.items():
            if param_name in self.kwargs:
                val = self.kwargs[param_name]
            else:
                val = param.default

            if val is not param.empty:
                if val != param.default and param_name not in self.get_default_kwargs():
                    unique_keywords[param_name] = val

        return unique_keywords

    class ExplainAlgorithmAction(argparse.Action):
        def __call__(self, parser, namespace, algorithm_name, option_string=None):
            if not AlgorithmWrapper.validate_type(algorithm_name, should_raise=False):
                parser.error(f"{algorithm_name} is not a valid algorithm.")

            print(algorithm_name)
            algorithm_name = AlgorithmWrapper.preprocess_algorithm_call(algorithm_name)
            print(algorithm_name)

            if not algorithm_name.endswith(")"):
                algorithm_name += "()"
            print(algorithm_name)
            obj: AlgorithmWrapper = eval(algorithm_name)
            obj.print_help(algorithm_name)

            parser.exit()


class MooreLearningAlgorithm(AlgorithmWrapper):
    default_parameters = dict(
        automaton_type='moore',
        return_data=True,
    )


class APMSL(MooreLearningAlgorithm):
    function = run_activePmSATLearn

    DEFAULT_RANDOM_WALK = (100, 10, 50)  # (num_walks, min_walk_len, max_walk_len)

    default_parameters = dict(
        pm_strategy='rc2',
        timeout=60*MINUTES,
    )

    positional_arguments = (
        'sliding_window_size',
        'extension_length',
    )

    aliases = {
        'sws': 'sliding_window_size',
        'el': 'extension_length',

        'icp': 'input_completeness_preprocessing',

        'tm': 'termination_mode',

        'gp': 'glitch_processing',
        'rp': 'replay_glitches',
        'wcp': 'window_cex_processing',
        'rw': 'random_walks',

        'cp': 'cex_processing',
        'dgt': 'discard_glitched_traces',
        'acahc': 'add_cex_as_hard_clauses',
    }

    shortcuts = {
        'ICP':      {'input_completeness_preprocessing': True},
        'NO_ICP':   {'input_completeness_preprocessing': False},

        'GP':       {'glitch_processing': True},
        'NO_GP':    {'glitch_processing': False},
        'REP':      {'replay_glitches': True},
        'NO_REP':   {'replay_glitches': False},
        'WCP':      {'window_cex_processing': True},
        'NO_WCP':   {'window_cex_processing': False},
        'RW':       {'random_walks': DEFAULT_RANDOM_WALK},
        'NO_RW':    {'random_walks': False},
        'ONLY_RW':  {'random_walks': DEFAULT_RANDOM_WALK,
                     'glitch_processing': False,
                     'replay_glitches': False,
                     'window_cex_processing': False,
                     'cex_processing': False,
                     'discard_glitched_traces': False,
                     'add_cex_as_hard_clauses': False,},

        'CP':       {'cex_processing': True},
        'NO_CP':    {'cex_processing': False},
        'ACAHC':    {'add_cex_as_hard_clauses': True},
        'NO_ACAHC': {'add_cex_as_hard_clauses': False},
    }

