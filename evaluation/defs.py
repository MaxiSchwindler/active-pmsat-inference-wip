import argparse
import re
import sys
from functools import partial
from dataclasses import dataclass
from inspect import signature, Signature, Parameter
from typing import Callable, Any

import aalpy.learning_algs
from aalpy import run_KV, run_Lstar
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.oracles import RandomWMethodEqOracle

import active_pmsatlearn.learnalgo
from active_pmsatlearn import GlitchThresholdTermination
from active_pmsatlearn.learnalgo import run_activePmSATLearn
from active_pmsatlearn.oracles import RobustRandomWalkEqOracle, RobustPerfectMooreEqOracle
from evaluation.gsm_comparison.active_gsm import run_activeGSM
from evaluation.utils import dict_product

DEBUG_WRAPPER = True

SECOND = SECONDS = 1
MINUTE = MINUTES = 60
HOUR = HOURS = MINUTE * 60


class RobustPerfectMooreOracle(RobustPerfectMooreEqOracle):
    def __init__(self, sul: MooreSUL):
        super().__init__(sul)


class RobustRandomWalkOracle(RobustRandomWalkEqOracle):
    def __init__(self, sul: MooreSUL | MealySUL, max_num_tries: int | None = 5):
        super().__init__(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            num_steps=sul.automaton.size * 5_000,
            reset_after_cex=True,
            reset_prob=0.09,
            max_num_tries=max_num_tries
        )

class RobustRandomWalkOracleWithoutAbort(RobustRandomWalkOracle):
    def __init__(self, sul: MooreSUL | MealySUL):
        super().__init__(
            sul=sul,
            max_num_tries=None,
        )


oracles = {
    "None": lambda sul: None,
    "Perfect": RobustPerfectMooreOracle,
    "RandomWithAbort": RobustRandomWalkOracle,
    "RandomWithoutAbort": RobustRandomWalkOracleWithoutAbort,
}


class AlgorithmWrapper:
    function: Callable = lambda *args, **kwargs: None
    default_parameters: dict[str, Any] = {}
    positional_arguments: tuple[str] = ()
    aliases: dict[str, str] = {}
    shortcuts: dict[dict[str, Any]] = {}

    default_kwargs_overridable: bool = False

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
                        if self.default_kwargs_overridable:
                            if self.get_default_kwargs()[k] == self.kwargs[k]:
                                self.kwargs[k] = v  # TODO: bit hacky that this is in here twice
                                continue
                        raise ValueError(f"Key {k} is already in kwargs (value: {self.kwargs[k]}), but would be overwritten by shortcut {a} (value: {v})")
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

    def kwargs_from_meta_knowledge(self, **meta_knowledge):
        return {}

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

    DEFAULT_NUM_RANDOM_STEPS = 200

    default_parameters = dict(
        pm_strategy='rc2',
        timeout=5*MINUTES,
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
        'rs': 'random_steps_per_round',
        'rsrp': 'random_steps_per_round_with_reset_prob',
        'tc': 'transition_coverage_steps',

        'cp': 'cex_processing',
        'dgt': 'discard_glitched_traces',
        'acahc': 'add_cex_as_hard_clauses',

        'ddt': 'deduplicate_traces',
        'orpg': 'only_replay_peak_glitches',
        'oait': 'only_add_incongruent_traces',
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

        'RW':       {'random_steps_per_round': DEFAULT_NUM_RANDOM_STEPS},
        'NO_RW':    {'random_steps_per_round': 0},
        'ONLY_RW':  {'random_steps_per_round': DEFAULT_NUM_RANDOM_STEPS,
                     'glitch_processing': False,
                     'replay_glitches': False,
                     'window_cex_processing': False,
                     'transition_coverage_steps': 0,
                     'cex_processing': False,
                     'discard_glitched_traces': False,
                     'add_cex_as_hard_clauses': False,},

        'TC':       {'transition_coverage_steps': DEFAULT_NUM_RANDOM_STEPS},
        'NO_TC':    {'transition_coverage_steps': 0},
        'ONLY_TC':  {'random_steps_per_round': 0,
                     'glitch_processing': False,
                     'replay_glitches': False,
                     'window_cex_processing': False,
                     'transition_coverage_steps': DEFAULT_NUM_RANDOM_STEPS,
                     'cex_processing': False,
                     'discard_glitched_traces': False,
                     'add_cex_as_hard_clauses': False,},

        'CP':       {'cex_processing': True},
        'NO_CP':    {'cex_processing': False},
        'ACAHC':    {'add_cex_as_hard_clauses': True},
        'NO_ACAHC': {'add_cex_as_hard_clauses': False},

        'GTT1':      {'termination_mode': lambda glitch_percent: GlitchThresholdTermination(threshold=glitch_percent+0.5,
                                                                                            first_time=True)},
        'GTT2':      {'termination_mode': lambda glitch_percent: GlitchThresholdTermination(threshold=glitch_percent+0.5,
                                                                                            first_time=False)},
    }


class GSM(MooreLearningAlgorithm):
    function = run_activeGSM

    default_kwargs_overridable = True

    default_parameters = dict(
        timeout=5*MINUTES,
        certainty=0.05,
    )

    positional_arguments = ()

    aliases = {
        'rsrp': 'random_steps_per_round_with_reset_prob',
        'spcs': 'state_prefix_coverage_steps_per_round',
    }

    shortcuts = {
        'PURGE':    {'purge_mismatches': True},
        'NO_PURGE': {'purge_mismatches': False},

        'ICP':      {'input_completeness_preprocessing': True},
        'NO_ICP':   {'input_completeness_preprocessing': False},

        'CP':       {'cex_processing': True},
        'NO_CP':    {'cex_processing': False},

        'UDAC':     {'cex_processing': True,
                     'use_dis_as_cex': True},

        'CT01':     {'certainty': 0.1},
        'CT001':    {'certainty': 0.01},
    }

    def kwargs_from_meta_knowledge(self, glitch_percent, **meta_knowledge):
        return {
            'failure_rate': glitch_percent / 100,
        }

class KVWithGlitchesInOracle(MooreLearningAlgorithm):
    @staticmethod
    def function(**kwargs):
        oracle = kwargs['eq_oracle']
        sul = kwargs['sul']

        def get_glitch_percentage(sul):
            return getattr(sul, 'glitch_percentage', None)

        def set_glitch_percentage(sul, glitch_percent):
            if hasattr(sul, 'glitch_percentage'):
                sul.glitch_percentage = glitch_percent

        stored_glitch_percentage = get_glitch_percentage(sul)

        def get_cex(eq_oracle, hyp):
            assert oracle is eq_oracle
            set_glitch_percentage(sul, stored_glitch_percentage)
            cex = eq_oracle.find_cex(hyp, return_outputs=False)
            sul.glitch_percentage = 0
            return cex

        hyp, info = run_KV(**kwargs, cache_and_non_det_check=False, get_cex=get_cex)  #needs a patch in aalpy to use get_cex as wrapper for oracle.find_cex
        return hyp, info


class _RobustLearningAlg(MooreLearningAlgorithm):
    _non_robust_function = None

    @classmethod
    def function(cls, **kwargs):
        from evaluation.robust_sul_wrapper import RobustSUL
        sul = kwargs['sul']
        kwargs['sul'] = RobustSUL(sul)

        return cls._non_robust_function(**kwargs)


class RobustKV(_RobustLearningAlg):
    _non_robust_function = run_KV

    default_parameters = dict(
        cache_and_non_det_check=False
    )


class RobustLstar(_RobustLearningAlg):
    _non_robust_function = run_Lstar

    default_parameters = dict(
        cache_and_non_det_check=False
    )


class RobustKVCached(_RobustLearningAlg):
    _non_robust_function = run_KV

    default_parameters = dict(
        cache_and_non_det_check=True
    )


class RobustLstarCached(_RobustLearningAlg):
    _non_robust_function = run_Lstar

    default_parameters = dict(
        cache_and_non_det_check=True
    )