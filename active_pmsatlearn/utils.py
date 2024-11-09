import time
from functools import wraps

from aalpy.base import SUL
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.automata import MooreMachine, MealyMachine

from pmsatlearn.learnalgo import Input, Output
from typing import Literal, Any, TypeAlias, Sequence

Trace: TypeAlias = tuple[Output | tuple[Input, Output]]
SupportedAutomaton: TypeAlias = MooreMachine | MealyMachine
SupportedSUL: TypeAlias = MooreSUL | MealySUL


def get_outputs(traces: Sequence[Trace]) -> set[Output]:
    """ Get all unique outputs from a sequence of traces"""
    return set([trace[0] for trace in traces] + [output for trace in traces for input, output in trace[1:]])


def get_num_outputs(traces: Sequence[Trace]) -> int:
    """ Get the number of unique outputs from a list of traces"""
    return len(get_outputs(traces))


def get_prefixes(lst: Sequence):
    """ Get all prefixes from a given sequence"""
    return [lst[:i] for i in range(1, len(lst) + 1)]


def trace_query(sul: SUL, inputs: Sequence[Input]) -> Trace:
    """ Perform a query on the SUL and return the produced trace.
    Returns traces as tuples of tuples, to enable hashability and immutability."""
    if hasattr(sul, "traces"):  # TracedMooreSUL
        sul.query(inputs)
        return tuple(sul.traces[-1])
    else:
        initial_output: Output = sul.query(tuple())[0]
        outputs: list[Output] = sul.query(inputs)
        return initial_output, *zip(inputs, outputs)


def get_input_from_stoc_trans(inp: str):
    if inp.startswith("!"):
        return inp.split("!")[-1].split("[")[0].strip()
    return inp.split(" ")[0]


def timeit(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            func_name = name if name else func.__name__
            print(f"{func_name} took {duration:.4f} seconds")
            return result
        return wrapper
    return decorator


def remove_duplicate_traces(traces, additional_traces):
    traces_set = set(traces)
    additional_traces[:] = [trace for trace in additional_traces if trace not in traces_set]


def log_all(message, level=0):
    print(message)