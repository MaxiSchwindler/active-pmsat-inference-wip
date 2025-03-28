import time
import builtins
from functools import wraps
from contextlib import contextmanager

from aalpy import MooreMachine, MealyMachine
from aalpy.base import SUL

from active_pmsatlearn.defs import Input, Output, Trace
from typing import Sequence


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
    assert None not in inputs, f"input sequence should not contain 'None'; trace_query already adds the initial output"

    if hasattr(sul, "traces"):  # TracedMooreSUL
        sul.query(inputs)
        retval = tuple(sul.traces[-1])
    else:
        initial_output: Output = sul.query(tuple())[0]
        outputs: list[Output] = sul.query(inputs)
        retval = initial_output, *zip(inputs, outputs)

    for given_input, returned_input in zip(inputs, [i for i, o in retval[1:]]):
        assert given_input == returned_input, f"Mismatch between given and received inputs: {given_input} != {returned_input}.\n {inputs=} | {retval=} | {sul=}"

    return retval

def get_input_from_stoc_trans(inp: str):
    if inp.startswith("!"):
        r = inp.split("!")[-1].split("[")[0].strip()
    else:
        r = inp.split(" ")[0]

    try:
        r_i = int(r)
    except ValueError:
        pass
    else:
        raise ValueError(f"Input '{r}' (hyp_stoc: {inp}) could have been a string or an integer originally!")

    return r


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


def remove_congruent_traces(hyp: MooreMachine | MealyMachine, additional_traces):
    incongruent_traces = []
    for trace in additional_traces:
        assert trace[0] == hyp.initial_state.output
        inputs = [i for i, o in trace[1:]]
        outputs_trace = [o for i, o in trace[1:]]
        outputs_hyp = hyp.execute_sequence(hyp.initial_state, inputs)
        if outputs_hyp != outputs_trace:
            incongruent_traces.append(trace)

    additional_traces[:] = incongruent_traces

@contextmanager
def override_print(override_with=None):
    if override_with is None:
        def overridden_print(*args, **kwargs):
            pass
    else:
        def overridden_print(*args, sep=' ', end='\n', file=None):
            override_with(sep.join(map(str, args)))

    original_print = builtins.print
    try:
        builtins.print = overridden_print
        yield
    finally:
        builtins.print = original_print
