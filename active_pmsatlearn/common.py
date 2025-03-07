import random

from typing import Any

from active_pmsatlearn.defs import *
from active_pmsatlearn.log import get_logger
from active_pmsatlearn.utils import trace_query, get_prefixes, get_input_from_stoc_trans

logger = get_logger("APMSL")

#TODO: rename this file to processing.py

def do_input_completeness_preprocessing(hyps: HypothesesWindow, suffix_mode: str, sul: SupportedSUL, alphabet: list[Input],
                                        all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Visit every state in the hypothesis and check if we have a transition for every input.
    If we don't have a transition there, query the SUL for additional traces of
    (state.prefix + input + (alphabet^extension_length)
    :param hyps: hypotheses window
    :param suffix_mode: which suffixes to append to (state.prefix + missing_input). One of:
                 'all_suffixes': all input combinations, i.e. alphabet^extension_length
                 'random_suffix': one random sequence of input combinations from all_input_combinations, i.e. random.choice(alphabet^extension_length)
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :return: a list of additional traces
    """
    new_traces = []
    already_performed_prefixes = set()

    for hyp, _, _ in hyps.values():
        hyp_str = f"{len(hyp.states)}-state hypothesis {id(hyp)}"
        if hyp.is_input_complete():
            logger.debug(f"{hyp_str} is already input complete.")
            continue

        logger.debug(f"Try to force input completeness in {hyp_str} to produce new traces...")

        hyp.compute_prefixes()
        for state in hyp.states:
            if state.prefix is None:
                continue  # ignore unreachable states (reachable only through glitches)

            for inp in alphabet:
                if inp not in state.transitions:
                    prefix = list(state.prefix) + [inp]
                    msg = (f"{hyp_str} didn't have a transition from state {state.state_id} "
                           f"(output='{state.output}', prefix={state.prefix}) with input '{inp}'")

                    if tuple(prefix) in already_performed_prefixes:
                        logger.debug(f"{msg}, but the prefix {prefix} was already performed this round - continue. ")
                        continue

                    logger.debug(f"{msg}. Perform queries with {prefix=} and {suffix_mode=}")

                    if suffix_mode == 'all_suffixes':
                        suffixes = all_input_combinations
                    elif suffix_mode == 'random_suffix':
                        suffixes = [random.choice(all_input_combinations)]
                    else:
                        assert False

                    for suffix in suffixes:
                        trace = trace_query(sul, prefix + list(suffix))
                        new_traces.append(trace)

                    already_performed_prefixes.add(tuple(prefix))

    return new_traces


def do_cex_processing(cex: Trace, sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Counterexample processing, like in Active RPNI
    :param sul: system under learning
    :param cex: counterexample
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :param log: a function to log info to
    :return: list of new traces
    """
    logger.debug("Processing counterexample to produce new traces...")
    new_traces = []
    for prefix in get_prefixes(cex):
        for suffix in all_input_combinations:
            trace = trace_query(sul, list(prefix) + list(suffix))
            new_traces.append(trace)

    return new_traces


def do_glitch_processing(hyps: HypothesesWindow, traces_used_to_learn_hyp: list[Trace], suffix_mode: str,
                         sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Glitch processing. For every glitched transition from state s with input i, query traces for
    s.prefix + i + <specified suffixes>. Note that only one prefix is queried, the one of the state with
    the glitched transition.
    :param hyps: hypotheses window
    :param traces_used_to_learn_hyp: list of traces used to learn the given hypothesis. Currently only used for verification.
    :param suffix_mode: which suffixes to append to (state.prefix + glitched_input). One of:
                     'all_suffixes': all input combinations, i.e. alphabet^extension_length
                     'random_suffix': one random sequence of input combinations from all_input_combinations, i.e. random.choice(alphabet^extension_length)
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :return: list of new traces
    """
    assert suffix_mode in ('all_suffixes', 'random_suffix')
    new_traces = []
    already_performed_prefixes = set()

    for hyp, hyp_stoc, pmsat_info in hyps.values():
        hyp_str = f"{len(hyp.states)}-state hypothesis {id(hyp)}"
        logger.debug(f"Use glitched transitions of {hyp_str} produce new traces...")
        all_glitched_steps = [traces_used_to_learn_hyp[traces_index][trace_index]
                              for (traces_index, trace_index) in pmsat_info["glitch_steps"]]

        all_glitched_transitions = pmsat_info["glitch_trans"]  # [(s1, i, s2), ...]

        hyp.compute_prefixes()
        for s1_id, glitched_input, s2_id in all_glitched_transitions:
            s1 = hyp.get_state_by_id(s1_id)
            s2 = hyp.get_state_by_id(s2_id)
            assert (glitched_input, s2.output) in all_glitched_steps, (
                f"The tuple {(glitched_input, s2.output)} "
                f"was not found in {all_glitched_steps=}")

            msg = (f"{hyp_str} contained a glitched transition from state {s1.state_id} (output='{s1.output}', "
                   f"prefix={s1.prefix}) with input '{glitched_input}' to state {s2.state_id}")

            if s1.prefix is None:
                logger.debug(f"{msg}, but since state {s1.state_id} is unreachable (prefix is None), continue.")
                continue

            prefix = list(s1.prefix) + [glitched_input]

            if tuple(prefix) in already_performed_prefixes:
                logger.debug(f"{msg}, but the prefix {prefix} was already performed this round - continue. ")
                continue

            logger.debug(f"{msg}. Perform queries with {prefix=} and {suffix_mode=}")

            if suffix_mode == 'all_suffixes':
                suffixes = all_input_combinations
            elif suffix_mode == 'random_suffix':
                suffixes = [random.choice(all_input_combinations)]
            else:
                assert False

            for suffix in suffixes:
                trace = trace_query(sul, prefix + list(suffix))
                new_traces.append(trace)

            already_performed_prefixes.add(tuple(prefix))

    return new_traces


def get_incongruent_traces(glitchless_trace, traces) -> set[int]:
    glitchlass_trace_inputs = [step[0] for step in glitchless_trace[1:]]
    glitchless_trace_outputs = [step[1] for step in glitchless_trace[1:]]
    traces_to_remove = set()
    for t_i, trace in enumerate(traces):
        assert glitchless_trace[0] == trace[0], f"Different initial outputs!"
        trace_inputs = [step[0] for step in trace[1:]]
        trace_outputs = [step[1] for step in trace[1:]]
        for s_i in range(min(len(glitchlass_trace_inputs), len(trace_inputs))):
            if trace_inputs[s_i] != glitchlass_trace_inputs[s_i]:
                break
            if trace_outputs[s_i] != glitchless_trace_outputs[s_i]:
                logger.debug(
                    f"Different outputs at step {s_i}! Glitchless trace contains output '{glitchless_trace_outputs[s_i]}' "
                    f"for input '{glitchlass_trace_inputs[s_i]}', but trace {t_i} ({trace}) contains '{trace_outputs[s_i]}'. Removing trace.")
                traces_to_remove.add(t_i)
                break
            assert trace_inputs[s_i] == glitchlass_trace_inputs[s_i]
            assert trace_outputs[s_i] == glitchless_trace_outputs[s_i]
    return traces_to_remove


