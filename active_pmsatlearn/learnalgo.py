import itertools
import logging
import time
import math
import statistics
from collections import defaultdict
from inspect import trace
from pprint import pprint

import numpy as np

from aalpy.base import Oracle

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.oracles import RobustEqOracleMixin

from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT
logger = get_logger("APMSL")

DECREASE = -1
INCREASE = 1


def run_activePmSATLearn(
    alphabet: list,
    sul: SupportedSUL,
    eq_oracle: Oracle,
    automaton_type: Literal["mealy", "moore"],
    extension_length: int = 2,
    pm_strategy: Literal["lsu", "fm", "rc2"] = "rc2",
    timeout: int | float | None = None,
    cost_scheme: Literal["per_step", "per_transition"] = "per_step",
    input_completeness_processing: bool = False,
    cex_processing: bool = True,
    glitch_processing: bool = True,
    discard_glitched_traces: bool = True,
    add_cex_as_hard_clauses: bool = True,
    return_data: bool = True,
    print_level: int | None = 2,
) -> (
        SupportedAutomaton | None | tuple[SupportedAutomaton | None, dict[str, Any]]
):
    """
    Active version of the PMSAT-LEARN algorithm.

    :param alphabet: input alphabet
    :param sul: system under learning
    :param eq_oracle: equivalence oracle
    :param automaton_type: type of automaton to be learned. One of ["mealy", "moore"]
    :param extension_length: length of input combinations for initialisation and processing
    :param pm_strategy: strategy to be used by the solver
    :param timeout: timeout for the entire call (in seconds). This currently only times the solver calls,
                    i.e. equivalence or membership queries could be still performed after the timeout has been reached.
    :param cost_scheme: cost scheme for pmsat-learn
    :param input_completeness_processing: whether input completeness processing should be performed
                                          before querying the oracle for a counterexample
    :param cex_processing: whether counterexample processing should be performed
    :param glitch_processing: whether glitch processing should be performed
    :param return_data: whether to return a dictionary containing learning information
    :param print_level: 0 - None,
                        1 - just results,
                        2 - current round information,
                        3 - pmsat-learn output and logging of traces
    """
    assert isinstance(eq_oracle, RobustEqOracleMixin), f"To ensure reliable counterexamples, a robust oracle must be passed."
    # additional input verification happens in run_pmSATLearn

    logger.setLevel({0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG, 3: DEBUG_EXT}[print_level])

    start_time = time.time()
    must_end_by = start_time + timeout if timeout is not None else 100 * 365 * 24 * 60 * 60  # no timeout -> must end in 100 years :)
    eq_query_time = 0
    learning_rounds = 0
    detailed_learning_info = defaultdict(dict)
    all_input_combinations = list(itertools.product(alphabet, repeat=extension_length))
    glitchless_traces = []

    common_pmsatlearn_kwargs = dict(automata_type=automaton_type,
                                    pm_strategy=pm_strategy,
                                    cost_scheme=cost_scheme,
                                    print_info=print_level > 2)

    def do_input_completeness_processing(current_hyp):
        return _do_input_completeness_processing(hyp=current_hyp, sul=sul, alphabet=alphabet,
                                                 all_input_combinations=all_input_combinations)

    def do_cex_processing(current_cex):
        return _do_cex_processing(sul=sul, cex=current_cex, all_input_combinations=all_input_combinations)

    def do_glitch_processing(current_hyp: SupportedAutomaton, current_pmsat_info: dict[str, Any],
                             current_hyp_stoc: SupportedAutomaton, current_traces: list[Trace]):
        return _do_glitch_processing(sul=sul, hyp=current_hyp, pmsat_info=current_pmsat_info, hyp_stoc=current_hyp_stoc,
                                     traces_used_to_learn_hyp=current_traces, all_input_combinations=all_input_combinations)

    logger.debug(f"Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    # start with minimum possible number of states
    num_states = get_num_outputs(traces)

    while not (timed_out := (time.time() >= must_end_by)):
        learning_rounds += 1
        detailed_learning_info[learning_rounds]["num_traces"] = len(traces)

        #####################################
        #              LEARNING             #
        #####################################

        logger.debug(f"Learning round {learning_rounds}: Running pmSATLearn to learn a {num_states}-state"
                     f"and a {num_states+1}-state automaton from {len(traces)} traces")
        logger.debug(f"Learning small ({num_states} state) hypothesis...")
        hyp_small, hyp_stoc_small, pmsat_info_small = run_pmSATLearn(data=traces,
                                                                     n_states=num_states,
                                                                     timeout=max(must_end_by - time.time(), 0),
                                                                     glitchless_data=glitchless_traces,
                                                                     **common_pmsatlearn_kwargs)
        logger.debug(f"Learning large ({num_states+1} state) hypothesis...")
        hyp_large, hyp_stoc_large, pmsat_info_large = run_pmSATLearn(data=traces,
                                                                     n_states=num_states + 1,
                                                                     timeout=max(must_end_by - time.time(), 0),
                                                                     glitchless_data=glitchless_traces,
                                                                     **common_pmsatlearn_kwargs)

        direction, reason = choose_direction(hyp_small, pmsat_info_small, hyp_large, pmsat_info_large, traces)
        if direction == INCREASE:
            logger.debug(f"Hypothesis with more states ({num_states+1}) was found to be better. Reason: {reason}")
            hyp, hyp_stoc, pmsat_info = hyp_large, hyp_stoc_large, pmsat_info_large
        else:
            logger.debug(f"Hypothesis with fewer states ({num_states}) was found to be better. Reason: {reason}")
            hyp, hyp_stoc, pmsat_info = hyp_small, hyp_stoc_small, pmsat_info_small

        detailed_learning_info[learning_rounds]["direction"] = "increase" if direction == INCREASE else "decrease"
        detailed_learning_info[learning_rounds]["reason"] = reason

        #####################################
        #              EQ-CHECK             #
        #####################################

        detailed_learning_info[learning_rounds]["num_states"] = len(hyp.states) if hyp is not None else None
        detailed_learning_info[learning_rounds]["num_glitches"] = len(pmsat_info["glitch_steps"]) if hyp is not None else 0
        detailed_learning_info[learning_rounds]["is_sat"] = pmsat_info["is_sat"]
        detailed_learning_info[learning_rounds]["timed_out"] = pmsat_info["timed_out"]

        if hyp is not None and pmsat_info["is_sat"]:
            logger.info(f"pmSATLearn learned hypothesis with {len(hyp.states)} states")

            if not hyp.is_input_complete():
                hyp.make_input_complete()  # oracles assume input completeness

            logger.debug("Querying for counterexample...")
            eq_query_start = time.time()
            cex, cex_outputs = eq_oracle.find_cex(hyp)
            eq_query_time += time.time() - eq_query_start

        else:
            # UNSAT - set cex to None to enter next if  # TODO: we can probably be unsat if we don't have enough states, right? then we should repeat with more states
            if pmsat_info["timed_out"]:
                timed_out = True
            cex = None
            hyp = None
            hyp_stoc = None

        if cex is None:
            if pmsat_info["is_sat"]:
                logger.info("No counterexample found. Returning hypothesis")
            else:
                logger.info("UNSAT! Returning None")

            break

        #####################################
        #          POST-PROCESSING          #
        #####################################

        logger.info(f"Counterexample {cex} found - process and continue learning")

        cex_trace = sul.query(tuple())[0], *zip(cex, cex_outputs)
        if add_cex_as_hard_clauses:
            logger.debug(f"Treating counterexample trace {cex_trace} as hard clause")
            glitchless_traces.append(cex_trace)
        else:
            logger.debug(f"Treating counterexample trace {cex_trace} as soft clause")
            traces.append(cex_trace)

        if glitch_processing:
            # we have to do glitch processing first, before appending anything to traces!
            postprocessing_additional_traces_glitch = do_glitch_processing(hyp, pmsat_info, hyp_stoc, traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)  # TODO: does de-duplication affect anything? check!

            logger.debug(f"Produced {len(postprocessing_additional_traces_glitch)} additional traces from glitch processing")
            logger.debug_ext(f"Additional traces from glitch processing: {postprocessing_additional_traces_glitch}")
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_glitch"] = len(postprocessing_additional_traces_glitch)

            traces = traces + postprocessing_additional_traces_glitch

        if cex_processing:
            postprocessing_additional_traces_cex = do_cex_processing(cex)

            remove_duplicate_traces(traces, postprocessing_additional_traces_cex) # TODO: does de-duplication affect anything? check!

            logger.debug(f"Produced {len(postprocessing_additional_traces_cex)} additional traces from counterexample")
            logger.debug_ext(f"Additional traces from counterexample: {postprocessing_additional_traces_cex}")
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_cex"] = len(postprocessing_additional_traces_cex)

            traces = traces + postprocessing_additional_traces_cex

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            logger.debug(f"No additional traces were produced during this round")

        if discard_glitched_traces:
            traces_to_remove = get_incongruent_traces(glitchless_trace=cex_trace, traces=traces)
            for glitchless_trace in glitchless_traces:
                if glitchless_trace == cex_trace:
                    continue  # already handled above
                traces_to_remove.update(get_incongruent_traces(glitchless_trace=glitchless_trace, traces=traces))

            logger.debug(f"Removing {len(traces_to_remove)} traces because they were incongruent with counterexample")
            detailed_learning_info[learning_rounds]["removed_traces"] = len(traces_to_remove)
            traces = [t for t_i, t in enumerate(traces) if t_i not in traces_to_remove]

        num_states = max(num_states + direction, get_num_outputs(traces))

    if timed_out:
        logger.warning(f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")

    if return_data:
        total_time = round(time.time() - start_time, 2)
        eq_query_time = round(eq_query_time, 2)
        learning_time = round(total_time - eq_query_time, 2)

        active_info = build_info_dict(hyp, sul, eq_oracle, learning_rounds, total_time, learning_time,
                                      eq_query_time, pmsat_info, detailed_learning_info, hyp_stoc, timed_out)
        if print_level > 1:
            print_learning_info(active_info)

        return hyp, active_info
    else:
        return hyp


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


def _do_input_completeness_processing(hyp: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input],
                                      all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Visit every state in the hypothesis and check if we have a transition for every input.
    If we don't have a transition there, query the SUL for additional traces of
    (state.prefix + input + (alphabet^extension_length)
    :param hyp: current hypothesis
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :param log: a function to log info to
    :return: a list of additional traces
    """
    logger.debug("Try to force input completeness in hypothesis to produce new traces...")
    if hyp.is_input_complete():
        logger.debug("Hypothesis is already input complete. In the current implementation, this step won't be useful.")

    hyp.compute_prefixes()
    new_traces = []
    for state in hyp.states:
        if state.prefix is None:
            continue  # ignore unreachable states (reachable only through glitches)

        for inp in alphabet:
            if inp not in state.transitions:
                logger.debug(f"Hypothesis didn't have a transition from state {state.state_id} "
                             f"(output='{state.output}', prefix={state.prefix}) with input '{inp}' - create traces!")

                for suffix in all_input_combinations:
                    trace = trace_query(sul, list(state.prefix) + [inp] + list(suffix))
                    new_traces.append(trace)

    return new_traces


def _do_cex_processing(sul: SupportedSUL, cex: Trace, all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
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


def _do_glitch_processing(sul: SupportedSUL, hyp: SupportedAutomaton, pmsat_info: dict[str, Any], hyp_stoc: SupportedAutomaton,
                         traces_used_to_learn_hyp: list[Trace], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Glitch processing. For every glitched transition from state s with input i, query traces for
    s.prefix + i + (alphabet^extension_length). Note that only one prefix is queried, the one of the state with
    the glitched transition.
    This is currently a somewhat hacky implementation, relying on hyp_stoc. TODO: replay on hyp instead
    :param sul: system under learning
    :param pmsat_info: info dict returned from last pmsat_learn call
    :param hyp_stoc: stochastic hypothesis returned from last pmsat_learn call
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :param log: a function to log info to
    :return: list of new traces
    """
    logger.debug("Use glitched transitions to produce new traces...")
    hyp_stoc.compute_prefixes()
    new_traces = []
    all_glitched_steps = [traces_used_to_learn_hyp[traces_index][trace_index] for (traces_index, trace_index) in
                          pmsat_info["glitch_steps"]]
    # all_glitched_trans = [() for (s0_id, inp, s1_id) in pmsat_info["glitch_trans"]]

    for state in hyp_stoc.states:
        for inp, next_state in state.transitions.items():
            if inp.startswith("!"):  # glitched transition
                state_prefix = [get_input_from_stoc_trans(i) for i in state.prefix]
                glitched_input = get_input_from_stoc_trans(inp)
                assert (glitched_input, next_state.output) in all_glitched_steps, (f"The tuple {(glitched_input, next_state.output)}" 
                                                                                   f"was not found in {all_glitched_steps=}")
                logger.debug(f"Hypothesis contained a glitched transition from state {state.state_id} (output='{state.output}', "
                             f"prefix={state_prefix}) with input '{glitched_input}' to state {next_state.state_id} - create traces!")

                assert (state.state_id, glitched_input, next_state.state_id) in pmsat_info["glitch_trans"], f"{pmsat_info['glitch_trans']=}"

                for suffix in all_input_combinations:
                    trace = trace_query(sul, state_prefix + [glitched_input] + list(suffix))
                    new_traces.append(trace)

    return new_traces


def choose_direction(hyp_small: SupportedAutomaton, pmsat_info_small: dict[str, Any],
                     hyp_large: SupportedAutomaton, pmsat_info_large: dict[str, Any],
                     traces: list[Trace]) -> tuple[int, str]:

    if hyp_small is None and hyp_large is not None:
        return INCREASE, "Failed to learn smaller hypothesis"
    elif hyp_small is not None and hyp_large is None:
        return DECREASE, "Failed to learn large hypothesis (how? Probably timeout?)"
    elif hyp_small is None and hyp_large is None:
        # doesn't matter what we give back, both are None
        return INCREASE, "Failed to learn both hypotheses"

    assert id(hyp_large) != id(hyp_small)

    glitches_small = len(pmsat_info_small["glitch_steps"])
    glitches_large = len(pmsat_info_large["glitch_steps"])

    # inferred automata with increasing number of states n will always have the same or decreasing number of glitches
    if glitches_large > glitches_small:
        assert False, "Increasing the number of states should never lead to an increase in glitches!"
    elif glitches_large == glitches_small:
        # we increased the number of states, but the number of glitches stayed the same - didn't help
        return DECREASE, "Equal number of glitches"
    else:
        # we increased the number of states, and the number of glitches went down.
        # however, this does not automatically mean the larger hypothesis is "better" - we could have encoded glitches as dominant transitions
        possible_encoded_glitches_small = find_similar_frequencies(dominant_frequencies=pmsat_info_small["dominant_delta_freq"],
                                                                   glitched_frequencies=pmsat_info_small["glitched_delta_freq"],)
        possible_encoded_glitches_large = find_similar_frequencies(dominant_frequencies=pmsat_info_large["dominant_delta_freq"],
                                                                   glitched_frequencies=pmsat_info_large["glitched_delta_freq"],)
        if possible_encoded_glitches_large <= possible_encoded_glitches_small:
            # we have fewer glitches, and less or equal possible glitches in our dominant transitions -> larger was better
            return INCREASE, (f"Fewer glitches and {'equal' if possible_encoded_glitches_large == possible_encoded_glitches_small else 'fewer'} "
                              f"dominant transitions which could be glitches")
        else:
            # we have fewer glitches, but the number of possible glitches in our dominant transitions increased -> assume that smaller was better
            return DECREASE, "Fewer glitches, but more dominant transitions which could be glitches"  # this can run endlessly

    assert False, "unreachable"

    # Possible optimization: if large has unreachable states which small didn't have, we want to decrease
    hyp_large.compute_prefixes()
    for state in hyp_large.states:
        if state.prefix is None:
            # unreachable state in larger hypothesis - decrease
            return DECREASE


def compute_z_scores(dominant_frequencies, glitched_frequencies):
    """
    Compute the z scores of the dominant frequencies (in relation to dominant frequencies) and
    of glitched frequencies (also in relation to dominant frequencies)
    """
    mean_dominant = statistics.mean(dominant_frequencies)
    std_dev_dominant = statistics.stdev(dominant_frequencies)

    if std_dev_dominant == 0:
        z_scores_dominant = [0] * len(dominant_frequencies)
        z_scores_glitched = [0] * len(glitched_frequencies)
    else:
        z_scores_dominant = [(f - mean_dominant) / std_dev_dominant for f in dominant_frequencies]
        z_scores_glitched = [(g - mean_dominant) / std_dev_dominant for g in glitched_frequencies]

    return z_scores_dominant, z_scores_glitched


def find_similar_frequencies(dominant_frequencies, glitched_frequencies, z_threshold=1.0):
    """
    Calculate the z-score of the dominant frequency relative to the glitched list to find dominant frequencies
    which are similar to glitched frequencies.
    """
    if len(glitched_frequencies) == 0:
        logger.warning("No glitched transitions given - cannot determine similar dominant frequencies")
        return []  # TODO: what to do if glitches == 0? there could be possible glitches in dominant freqs, but no way to compare...

    mean_glitched = np.mean(glitched_frequencies)
    std_glitched = np.std(glitched_frequencies)

    similar_frequencies = []
    for d in dominant_frequencies:
        z_score = (d - mean_glitched) / std_glitched if std_glitched != 0 else 0
        if abs(z_score) <= z_threshold:
            similar_frequencies.append(d)

    return similar_frequencies


def build_info_dict(hyp, sul, eq_oracle, learning_rounds, total_time, learning_time, eq_query_time,
                    last_pmsat_info, detailed_learning_info, hyp_stoc, timed_out):
    active_info = {
        'learning_rounds': learning_rounds,
        'learned_automaton_size': hyp.size if hyp is not None else None,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time,
        'cache_saved': sul.num_cached_queries,
        'last_pmsat_info': last_pmsat_info,
        'detailed_learning_info': detailed_learning_info,
        'hyp_stoc': str(hyp_stoc) if hyp is not None else None,
        'timed_out': timed_out,
    }
    return active_info


def print_learning_info(info: dict[str, Any]):
    """
    Print learning statistics.
    """
    print('-----------------------------------')
    print('Learning Finished.')
    print('Learning Rounds:  {}'.format(info['learning_rounds']))
    print('Number of states: {}'.format(info['learned_automaton_size']))
    print('Time (in seconds)')
    print('  Total                : {}'.format(info['total_time']))
    print('  Learning algorithm   : {}'.format(info['learning_time']))
    print('  Conformance checking : {}'.format(info['eq_oracle_time']))
    print('Learning Algorithm')
    print(' # Membership Queries  : {}'.format(info['queries_learning']))
    if 'cache_saved' in info.keys():
        print(' # MQ Saved by Caching : {}'.format(info['cache_saved']))
    print(' # Steps               : {}'.format(info['steps_learning']))
    print('Equivalence Query')
    print(' # Membership Queries  : {}'.format(info['queries_eq_oracle']))
    print(' # Steps               : {}'.format(info['steps_eq_oracle']))
    print('-----------------------------------')