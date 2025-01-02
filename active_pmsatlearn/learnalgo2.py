import itertools
import logging
import time
import math
import statistics
from collections import defaultdict
from inspect import trace
from pprint import pprint
import random

import numpy as np

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT
logger = get_logger("APMSL_noMAT")

DECREASE = -1
INCREASE = 1


def simple_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace]):
    logger.debug_ext(f"Calculating simple heuristic...")
    score = 0

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    if len(glitched_delta_freq) > 0:
        score += 1 / np.mean(glitched_delta_freq)
        logger.debug_ext(f"Adding inverse of mean of glitched delta frequencies ({np.mean(glitched_delta_freq)}) to {score=}")
    else:
        score += 1
        logger.debug_ext(f"No glitched delta frequencies; {score=}")

    dominant_delta_freq = learning_info["dominant_delta_freq"]
    dominant_delta_freq = [freq for freq in dominant_delta_freq if freq > 0]  # note: 0-freqs are dominant transitions without evidence

    score -= 1 / np.min(dominant_delta_freq)
    logger.debug_ext(f"Subtracting inverse of minimum of dominant delta frequencies ({np.min(dominant_delta_freq)}) from {score=}")

    return score


def advanced_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace],
                       *, reward_lower_glitch_mean=True, reward_lower_num_glitches=True, punish_unreachable_states=True,
                       punish_low_dominant_freq=True, punish_input_incompleteness=True):
    score = 0

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    dominant_delta_freq = learning_info["dominant_delta_freq"]

    # reward lower mean of glitched frequencies
    if reward_lower_glitch_mean:
        if len(glitched_delta_freq) > 0:
            score += 1 / np.mean(glitched_delta_freq)
        else:
            score += 1  # this rewards "no glitches" a lot

    # reward lower numbers of glitches
    if reward_lower_num_glitches:
        num_glitches = len(learning_info["glitch_steps"])
        if num_glitches > 0:
            score += 1 / num_glitches

    # penalize unreachable states (in absolute numbers!) a LOT
    if punish_unreachable_states:
        learned_model.compute_prefixes()
        num_unreachable = sum(1 for state in learned_model.states if state.prefix is None)
        score -= num_unreachable

    # penalize low dominant frequencies
    if punish_low_dominant_freq:
        dominant_delta_freq_without_zero = [freq for freq in dominant_delta_freq if freq > 0]
        score -= 1 / np.min(dominant_delta_freq_without_zero)

    # punish dominant frequencies that are 0 (-> not input complete)
    if punish_input_incompleteness:
        num_dominant_zero_freq = sum(1 for freq in dominant_delta_freq if freq == 0)
        score -= num_dominant_zero_freq

    return score


def run_activePmSATLearn(
    alphabet: list,
    sul: SupportedSUL,
    automaton_type: Literal["mealy", "moore"],
    extension_length: int = 2,
    sliding_window_size: int = 5,
    heuristic_function: str | HeuristicFunction = "simple",
    pm_strategy: Literal["lsu", "fm", "rc2"] = "rc2",
    timeout: int | float | None = None,
    cost_scheme: Literal["per_step", "per_transition"] = "per_step",
    input_completeness_processing: bool = False,
    glitch_processing: bool = True,
    discard_glitched_traces: bool = True,
    random_state_exploration: bool = True,
    return_data: bool = True,
    print_level: int | None = 2,
) -> (
        SupportedAutomaton | None | tuple[SupportedAutomaton | None, dict[str, Any]]
):
    """
    Active version of the PMSAT-LEARN algorithm.

    :param alphabet: input alphabet
    :param sul: system under learning
    :param automaton_type: type of automaton to be learned. One of ["mealy", "moore"]
    :param extension_length: length of input combinations for initialisation and processing
    :param sliding_window_size: size of sliding window (=number of hypotheses learned in one round). Must be an odd number.
    :param heuristic_function: the heuristic function to be used to compare hypotheses. Either 'simple'/'advanced' or a callable,
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
    assert sliding_window_size % 2 == 1, "Sliding window size must be odd (for now)"
    # additional input verification happens in run_pmSATLearn

    if heuristic_function == "simple":
        heuristic_function = simple_heuristic
    elif heuristic_function == "advanced":
        heuristic_function = advanced_heuristic
    else:
        assert callable(heuristic_function), "heuristic_function must be either 'simple'/'advanced' or a callable"

    logger.setLevel({0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG, 2.5: DEBUG_EXT, 3: DEBUG_EXT}[print_level])

    start_time = time.time()
    must_end_by = start_time + timeout if timeout is not None else 100 * 365 * 24 * 60 * 60  # no timeout -> must end in 100 years :)
    learning_rounds = 0
    final_hyps_and_info = None, None, None
    detailed_learning_info = defaultdict(dict)
    all_input_combinations = list(itertools.product(alphabet, repeat=extension_length))

    def do_input_completeness_processing(current_hyp):
        return _do_input_completeness_processing(hyp=current_hyp, sul=sul, alphabet=alphabet,
                                                 all_input_combinations=all_input_combinations)

    def do_glitch_processing(current_hyp: SupportedAutomaton, current_pmsat_info: dict[str, Any],
                             current_hyp_stoc: SupportedAutomaton, current_traces: list[Trace]):
        return _do_glitch_processing(sul=sul, hyp=current_hyp, pmsat_info=current_pmsat_info, hyp_stoc=current_hyp_stoc,
                                     traces_used_to_learn_hyp=current_traces, all_input_combinations=all_input_combinations)

    def do_state_exploration(hyp: SupportedAutomaton):
        return _do_state_exploration(hyp=hyp, sul=sul, alphabet=alphabet, all_input_combinations=all_input_combinations)

    logger.debug("Running the non-MAT learn algorithm...")
    logger.debug("Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    # start with minimum possible number of states
    min_num_states = get_num_outputs(traces)

    while not (timed_out := (time.time() >= must_end_by)):
        learning_rounds += 1
        logger.info(f"Starting learning round {learning_rounds}")
        detailed_learning_info[learning_rounds]["num_traces"] = len(traces)
        detailed_learning_info[learning_rounds]["sliding_window_start"] = min_num_states

        #####################################
        #              LEARNING             #
        #####################################

        hypotheses = learn_sliding_window(sliding_window_size=sliding_window_size,
                                          min_num_states=min_num_states,
                                          traces=traces,
                                          must_end_by=must_end_by,
                                          automata_type=automaton_type,
                                          pm_strategy=pm_strategy,
                                          cost_scheme=cost_scheme,
                                          print_info=print_level == 3)
        assert len(hypotheses) == sliding_window_size

        if input_completeness_processing:
            ### Might look roughly like this:
            # traces_learnt_from = copy.deepcopy(traces)
            # for hyp, _, _ in hypotheses.values():
            #     preprocessing_additional_traces = do_input_completeness_processing(current_hyp=hyp, traces_learnt_from=traces_learnt_from, current_traces=traces)
            #     traces += preprocessing_additional_traces
            # continue
            raise NotImplementedError("Input completeness preprocessing is not implemented yet.")

        scores = {num_states: heuristic_function(hyp, info, traces) for num_states, (hyp, _, info) in hypotheses.items()}
        assert sorted(scores.keys()) == list(scores.keys())
        logger.debug(f"Calculated the following scores: {scores}")

        peak_index = find_index_of_peak(scores)
        peak_num_states = list(scores.keys())[peak_index] if peak_index is not None else None
        detailed_learning_info[learning_rounds]["peak_index"] = peak_index
        detailed_learning_info[learning_rounds]["peak_num_states"] = peak_num_states

        if peak_index is None:
            # we don't have a peak -> we don't know which direction to move in
            # probably: generate more traces, learn same window again, at some point (after 2 such rounds?), simply return highest score?
            # also possible: move in direction of absolute highest value (even if it is not a clear peak)?
            logger.debug("Did not find a clear peak")
            raise NotImplementedError
        else:
            logger.debug(f"Found peak at index {peak_index} of sliding window <{min(scores.keys())}-{max(scores.keys())}>; "
                         f"i.e. at {peak_num_states} states")

            mid_index = sliding_window_size // 2
            if peak_index == mid_index or (peak_index < mid_index and min_num_states == get_num_outputs(traces)):
                if peak_index == mid_index:
                    logger.debug(f"Peak is exactly in the middle; Window is positioned correctly")
                else:
                    logger.debug(f"Peak is in left side of sliding window, but cannot move window further left; Window is positioned correctly")
                peak_hyp, peak_hyp_stoc, peak_pmsat_info = hypotheses[peak_num_states]

                passes_sanity_checks, reason = sanity_checks(peak_hyp)
                if not passes_sanity_checks:
                    logger.debug(f"Peak hypothesis did not pass sanity checks: {reason}. Continuing.")
                else:
                    logger.debug(f"Peak hypothesis passed sanity checks; checking requirements")

                    complete_num_steps = sum(len(trace) for trace in traces)
                    percent_glitches = len(peak_pmsat_info["glitch_steps"]) / complete_num_steps * 100
                    if percent_glitches < 1.0:
                        logger.debug("Peak hypothesis passed requirements, returning")
                        # TODO: maybe run again with same window, compare scores with last round? return if no improvement?
                        final_hyps_and_info = peak_hyp, peak_hyp_stoc, peak_pmsat_info
                        break
                    else:
                        logger.debug(f"Peak hypothesis failed requirements, continuing")

            else:
                diff_to_mid_index = peak_index - mid_index
                min_num_states = min_num_states + diff_to_mid_index
                if min_num_states >= (num_outputs := get_num_outputs(traces)):
                    logger.debug(f"Moving start of sliding window to {min_num_states} states to position peak at the midpoint.")
                else:
                    logger.debug(f"Should move start of sliding window to {min_num_states} states to position peak at the midpoint, "
                                 f"but there are too many outputs in the traces ({num_outputs}); setting min_num_states to {num_outputs}")
                    min_num_states = num_outputs

        #####################################
        #          POST-PROCESSING          #
        #####################################

        # we can only come here if we did not find a hypothesis we want to return
        logger.debug(f"Beginning postprocessing of hypotheses")

        if glitch_processing:
            # we have to do glitch processing first, before appending anything to traces!
            postprocessing_additional_traces_glitch = []
            for hyp, hyp_stoc, pmsat_info in hypotheses.values():
                additional_traces = do_glitch_processing(hyp, pmsat_info, hyp_stoc, traces)
                postprocessing_additional_traces_glitch.extend(additional_traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)  # TODO: does de-duplication affect anything? check!

            logger.debug(f"Produced {len(postprocessing_additional_traces_glitch)} additional traces from glitch processing")
            logger.debug_ext(f"Additional traces from glitch processing: {postprocessing_additional_traces_glitch}")
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_glitch"] = len(postprocessing_additional_traces_glitch)

            traces = traces + postprocessing_additional_traces_glitch

        if random_state_exploration:
            postprocessing_additional_traces_state_expl = []
            for hyp, hyp_stoc, pmsat_info in hypotheses.values():
                additional_traces = do_state_exploration(hyp)
                postprocessing_additional_traces_state_expl.extend(additional_traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_state_expl)  # TODO: does de-duplication affect anything? check!

            logger.debug(f"Produced {len(postprocessing_additional_traces_state_expl)} additional traces from state exploration")
            logger.debug_ext(f"Additional traces from state exploration: {postprocessing_additional_traces_state_expl}")
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_state_exploration"] = len(postprocessing_additional_traces_state_expl)

            traces = traces + postprocessing_additional_traces_state_expl

        # TODO: possible postprocessing idea: do bisimilarity checks between all current hypotheses, and use ARPNI-cex-processing to produce new traces

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            logger.warning(f"No additional traces were produced during this round!")

        if discard_glitched_traces:
            raise NotImplementedError
            # possibly remove traces if too much counter evidence exists
            # probably won't do much though...

            # traces_to_remove = get_incongruent_traces(glitchless_trace=cex_trace, traces=traces)
            # logger.debug(f"Removing {len(traces_to_remove)} traces because they were incongruent with counterexample")
            # detailed_learning_info[learning_rounds]["removed_traces"] = len(traces_to_remove)
            # traces = [t for t_i, t in enumerate(traces) if t_i not in traces_to_remove]

    if timed_out:
        logger.warning(f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")

    hyp, hyp_stoc, pmsat_info = final_hyps_and_info
    if return_data:
        total_time = round(time.time() - start_time, 2)

        active_info = build_info_dict(hyp, sul, learning_rounds, total_time,
                                      pmsat_info, detailed_learning_info, hyp_stoc, timed_out)
        if print_level > 1:
            print_learning_info(active_info)

        return hyp, active_info
    else:
        return hyp


def learn_sliding_window(sliding_window_size: int, min_num_states: int, traces: list[Trace], must_end_by: float = 0.0,
                         **common_pmsatlearn_kwargs) -> dict[int, tuple[PossibleHypothesis, PossibleHypothesis, dict]]:
    logger.debug(f"Running pmSATLearn to learn {sliding_window_size} hypotheses with between {min_num_states} and "
                 f"{min_num_states + sliding_window_size} states from {len(traces)} traces")
    learned = {}
    for i in range(sliding_window_size):
        num_states = min_num_states + i
        assert num_states >= get_num_outputs(traces)
        logger.debug(f"Learning {num_states}-state hypothesis...")
        learned[num_states] = run_pmSATLearn(data=traces,
                                             n_states=num_states,
                                             timeout=max(must_end_by - time.time(), 0),
                                             **common_pmsatlearn_kwargs)
    return learned


def find_index_of_peak(scores: dict[int, float]) -> int | None:
    # TODO: what about e.g. w-shaped peaks? - maybe we can ignore them, as we only want clear peaks
    score_values = list(scores.values())

    for i in range(0, len(score_values)):
        left = score_values[:i]
        right = score_values[i + 1:]
        if all(score_values[i] >= val for val in left) and all(score_values[i] >= val for val in right):
            # return num_states[i]  # returns the number of states at which the peak is
            return i  # returns the index of the peak in our sliding window

    return None


def sanity_checks(hyp: SupportedAutomaton) -> tuple[bool, str]:
    """
    Perform a bunch of sanity checks, without which we don't accept a hypothesis
    :param hyp: hypothesis to check
    :returns a tuple of (passes, reason)
    """

    # hypothesis must be input complete
    hyp.compute_prefixes()
    if not hyp.is_input_complete():
        return False, "Not input complete"

    if any(s.prefix is None for s in hyp.states):
        return False, "Has unreachable states"

    return True, "Passes all checks"



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


def _do_state_exploration(hyp: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input],
                          all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    logger.debug(f"Do state exploration of {len(hyp.states)}-state hypothesis {id(hyp)} to produce new traces...")

    # TODO: this method does not make much (any?) sense currently

    step_probability = 0.5
    hyp.make_input_complete()  #...
    hyp.reset_to_initial()

    while random.random() < step_probability:
        input = random.choice(alphabet)
        hyp.step(input)

    hyp.compute_prefixes()
    prefix = hyp.current_state.prefix
    new_traces = [trace_query(sul, list(prefix) + list(suffix)) for suffix in all_input_combinations]

    logger.debug(f"Produced {len(new_traces)} new traces from state exploration of {len(hyp.states)}-state hypothesis {id(hyp)}")

    return new_traces

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
    logger.debug(f"Use glitched transitions of {len(hyp.states)}-state hypothesis {id(hyp)} produce new traces...")
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

    logger.debug(f"Produced {len(new_traces)} new traces from glitch processing of {len(hyp.states)}-state hypothesis {id(hyp)}")
    return new_traces


def build_info_dict(hyp, sul, learning_rounds, total_time,
                    last_pmsat_info, detailed_learning_info, hyp_stoc, timed_out):
    active_info = {
        'learning_rounds': learning_rounds,
        'learned_automaton_size': hyp.size if hyp is not None else None,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': -1,
        'steps_eq_oracle': -1,
        'eq_oracle_time': -1,
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
    print('Time (in seconds)      : {}'.format(info['total_time']))
    print('Learning Algorithm')
    print(' # Membership Queries  : {}'.format(info['queries_learning']))
    if 'cache_saved' in info.keys():
        print(' # MQ Saved by Caching : {}'.format(info['cache_saved']))
    print(' # Steps               : {}'.format(info['steps_learning']))
    print('-----------------------------------')