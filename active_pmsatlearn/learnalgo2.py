import itertools
import logging
import time
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT
logger = get_logger("APMSL_noMAT")

class Action:
    pass


@dataclass
class Terminate(Action):
    hyp: PossibleHypothesis
    hyp_stoc: PossibleHypothesis
    pmsat_info: PmSatLearningInfo


@dataclass
class Continue(Action):
    next_min_num_states: int = None


class TerminationMode:
    pass


class RequirementsTermination(TerminationMode):
    pass


class GlitchThresholdTermination(RequirementsTermination):
    threshold: float = 1.0


class ImprovementTermination(TerminationMode):
    pass


class ScoreImprovementTermination(ImprovementTermination):
    pass


class GlitchImprovementTermination(ImprovementTermination):
    pass


def simple_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace]):
    logger.debug_ext(f"Calculating simple heuristic...")
    score = 0

    if learned_model is None:
        return -100

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    if len(glitched_delta_freq) > 0:
        score += 1 / np.mean(glitched_delta_freq)
        logger.debug_ext(f"Adding inverse of mean of glitched delta frequencies ({np.mean(glitched_delta_freq)}) to {score=}")
    else:
        score += 1
        logger.debug_ext(f"No glitched delta frequencies; {score=}")

    dominant_delta_freq = learning_info["dominant_delta_freq"]
    dominant_delta_freq = [freq for freq in dominant_delta_freq if freq > 0]  # note: 0-freqs are dominant transitions without evidence -> can't be input complete without these

    score -= 1 / np.min(dominant_delta_freq)
    logger.debug_ext(f"Subtracting inverse of minimum of dominant delta frequencies ({np.min(dominant_delta_freq)}) from {score=}")

    return float(score)  # cast numpy away (nicer output)


def intermediary_heuristic(*args, **kwargs):
    # basically the simple heuristic plus a reward for lower numbers of glitches
    return advanced_heuristic(*args, **kwargs, punish_input_incompleteness=False, punish_unreachable_states=False)


def advanced_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace],
                       *, reward_lower_glitch_mean=True, reward_lower_num_glitches=True, punish_unreachable_states=True,
                       punish_low_dominant_freq=True, punish_input_incompleteness=True):
    score = 0

    if learned_model is None:
        return -100

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    dominant_delta_freq = learning_info["dominant_delta_freq"]

    # reward lower mean of glitched frequencies
    # ADDS between 0 and 1
    if reward_lower_glitch_mean:
        if len(glitched_delta_freq) > 0:
            score += 1 / np.mean(glitched_delta_freq)

    # reward lower numbers of glitches
    # ADDS between 0 and 1
    if reward_lower_num_glitches:
        num_glitches = len(learning_info["glitch_steps"])
        if num_glitches > 0:
            score += 1 / num_glitches
        else:
            score += 1

    # penalize low dominant frequencies
    # SUBTRACTS between 0 and 1
    if punish_low_dominant_freq:
        dominant_delta_freq_without_zero = [freq for freq in dominant_delta_freq if freq > 0]
        score -= 1 / np.min(dominant_delta_freq_without_zero)

    # penalize unreachable states (in absolute numbers!) a LOT
    # SUBTRACTS between 0 and inf
    if punish_unreachable_states:
        learned_model.compute_prefixes()
        num_unreachable = sum(1 for state in learned_model.states if state.prefix is None)
        score -= num_unreachable

    # punish dominant frequencies that are 0 (-> not input complete)
    # SUBTRACTS between 0 and inf
    if punish_input_incompleteness:
        num_dominant_zero_freq = sum(1 for freq in dominant_delta_freq if freq == 0)
        score -= num_dominant_zero_freq

    return float(score)  # cast numpy away (nicer output)


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
    glitch_processing: bool = False,
    counterexample_processing: bool = False,
    discard_glitched_traces: bool = False,
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

    match heuristic_function:
        case "simple":
            heuristic_function = simple_heuristic
        case "intermediary":
            heuristic_function = intermediary_heuristic
        case "advanced":
            heuristic_function = advanced_heuristic
        case _:
            assert callable(heuristic_function), "heuristic_function must be either 'simple'/'advanced' or a callable"

    logger.setLevel({0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG, 2.5: DEBUG_EXT, 3: DEBUG_EXT}[print_level])
    logger.debug("Running the non-MAT learn algorithm...")
    detailed_learning_info = defaultdict(dict)

    start_time = time.time()
    must_end_by = start_time + timeout if timeout is not None else 100 * 365 * 24 * 60 * 60  # no timeout -> must end in 100 years :)
    learning_rounds = 0

    all_input_combinations = list(itertools.product(alphabet, repeat=extension_length))

    final_hyps_and_info = None, None, None
    previous_hypotheses = None
    previous_scores = None

    def do_input_completeness_processing(current_hyp):
        return _do_input_completeness_processing(hyp=current_hyp, sul=sul, alphabet=alphabet,
                                                 all_input_combinations=all_input_combinations)

    def do_glitch_processing(current_hyp: SupportedAutomaton, current_pmsat_info: dict[str, Any],
                             current_hyp_stoc: SupportedAutomaton, current_traces: list[Trace]):
        return _do_glitch_processing(sul=sul, hyp=current_hyp, pmsat_info=current_pmsat_info, hyp_stoc=current_hyp_stoc,
                                     traces_used_to_learn_hyp=current_traces, all_input_combinations=all_input_combinations)

    def do_state_exploration(hyp: SupportedAutomaton):
        return _do_state_exploration(hyp=hyp, sul=sul, alphabet=alphabet, all_input_combinations=all_input_combinations)

    logger.debug("Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    # start with minimum possible number of states
    min_num_states = get_num_outputs(traces)

    while not (timed_out := (time.time() >= must_end_by)):
        logger.info(f"Starting learning round {(learning_rounds := learning_rounds + 1)}")
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
        # assert len(hypotheses) == sliding_window_size  # might not hold with timeouts

        if (num_timed_out := sum(1 if info["timed_out"] else 0 for h, h_s, info in hypotheses.values())) == len(hypotheses):
            timed_out = True
            break
        elif num_timed_out > 0:
            assert num_timed_out == 1, "There should only be one timed out hypothesis (the last one)"
            assert hypotheses[-1][-1]["timed_out"]
            hypotheses.pop(-1)  # remove timed-out hypothesis

        #####################################
        #           PREPROCESSING           #
        #####################################

        if input_completeness_processing:
            # TODO: could this lead to an infinite loop?
            preprocessing_additional_traces = []
            for hyp, _, _ in hypotheses.values():
                additional_traces = do_input_completeness_processing(current_hyp=hyp)
                preprocessing_additional_traces.extend(additional_traces)

            remove_duplicate_traces(traces, preprocessing_additional_traces)  # TODO: does de-duplication affect anything? check!
            logger.debug(f"Produced {len(preprocessing_additional_traces)} additional traces from input completeness processing")
            logger.debug_ext(f"Additional traces from input completeness processing: {preprocessing_additional_traces}")
            detailed_learning_info[learning_rounds]["preprocessing_additional_traces"] = len(preprocessing_additional_traces)

            if preprocessing_additional_traces:
                traces += preprocessing_additional_traces
                continue

        #####################################
        #              SCORING              #
        #####################################

        scores = {num_states: heuristic_function(hyp, info, traces) for num_states, (hyp, _, info) in hypotheses.items()}
        assert sorted(scores.keys()) == list(scores.keys())
        logger.debug(f"Calculated the following scores: {scores}")
        detailed_learning_info[learning_rounds]["heuristic_scores"] = scores

        #####################################
        #            EVALUATING             #
        #####################################

        action = decide_next_action(hypotheses, scores,
                                    previous_hypotheses, previous_scores,
                                    current_min_num_states=min_num_states,
                                    traces_used_to_learn=traces,
                                    termination_mode=GlitchImprovementTermination(),
                                    current_learning_info=detailed_learning_info[learning_rounds])

        match action:
            case Terminate(hyp, hyp_stoc, pmsat_info):
                final_hyps_and_info = hyp, hyp_stoc, pmsat_info
                break
            case Continue(next_min_num_states):
                if next_min_num_states is not None:
                    min_num_states = next_min_num_states
            case _:
                raise ValueError(f"Unexpected action {action}")

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

        previous_hypotheses = hypotheses
        previous_scores = scores
        min_num_states = max(min_num_states, get_num_outputs(traces))

    if timed_out:
        logger.warning(f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")
        if previous_scores:
            previous_peak = find_index_of_absolute_peak(previous_scores)
            previous_peak_num_states = get_num_states_from_scores_index(previous_scores, previous_peak)
            final_hyps_and_info = previous_hypotheses[previous_peak_num_states]

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
                         **common_pmsatlearn_kwargs) -> HypothesesWindow:
    logger.debug(f"Running pmSATLearn to learn {sliding_window_size} hypotheses with between {min_num_states} and "
                 f"{min_num_states + sliding_window_size} states from {len(traces)} traces")
    num_outputs = get_num_outputs(traces)
    learned = {}
    for i in range(sliding_window_size):
        num_states = min_num_states + i
        assert num_states >= num_outputs
        logger.debug(f"Learning {num_states}-state hypothesis...")
        hyp, h_stoc, pmsat_info = run_pmSATLearn(data=traces,
                                                 n_states=num_states,
                                                 timeout=max(must_end_by - time.time(), 0),
                                                 **common_pmsatlearn_kwargs)
        if pmsat_info["timed_out"]:
            break  # the first time learning times out, we end learning the sliding window (everything afterwards will also time out)
        else:
            pmsat_info["percent_glitches"] = get_glitch_percentage(pmsat_info, traces)
        learned[num_states] = hyp, h_stoc, pmsat_info
    return learned

def find_index_of_unimodal_peak(scores: dict[int, float]) -> int | None:
    if scores is None:
        return None
    vals = list(scores.values())
    peak_index = vals.index(max(vals))
    if all(vals[i] <= vals[i + 1] for i in range(0, peak_index)) and all(vals[j] >= vals[j+1] for j in range(peak_index, len(vals) - 1)):
        return peak_index

def find_index_of_absolute_peak(scores: dict[int, float]) -> int:
    if scores is None:
        return None
    vals = list(scores.values())
    return vals.index(max(vals))

def get_num_states_from_scores_index(scores: dict[int, float], index: int | None) -> int | None:
    if index is None:
        return None
    return list(scores.keys())[index]



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


def decide_next_action(current_hypotheses: HypothesesWindow, current_scores: dict[int, float],
                       previous_hypotheses: HypothesesWindow | None, previous_scores: dict[int, float] | None,
                       current_min_num_states: int, traces_used_to_learn: list[Trace],
                       termination_mode: TerminationMode,
                       current_learning_info: dict[str, Any]) -> Action:
    peak_index = current_learning_info["peak_index"] = find_index_of_unimodal_peak(current_scores)
    peak_num_states = current_learning_info["peak_num_states"] = get_num_states_from_scores_index(current_scores, peak_index)

    prev_peak_index = find_index_of_unimodal_peak(previous_scores)
    prev_peak_num_states = get_num_states_from_scores_index(previous_scores, prev_peak_index)

    unimodal_peak = bool(peak_index is not None)
    sliding_window_size = len(current_scores)
    min_allowed_num_states = get_num_outputs(traces_used_to_learn)

    if not unimodal_peak:
        match previous_hypotheses, prev_peak_index:
            case None, None:
                logger.debug("No unimodal peak found in initial round; continuing with same window after collecting more traces.")
                return Continue()
            case _, None:
                logger.debug("No unimodal peak found in current and previous round; using absolute peak.")
                peak_index = current_learning_info["absolute_peak_index"] = find_index_of_absolute_peak(current_scores)
                peak_num_states = current_learning_info["absolute_peak_num_states"] = get_num_states_from_scores_index(current_scores, peak_index)
            case _, _:
                logger.debug("No unimodal peak found in current round; continuing with same window after collecting more traces")
                return Continue()
            case _:
                raise ValueError(f"Unhandled case: {previous_hypotheses=}, {prev_peak_index=}")

    logger.debug(f"Found peak at index {peak_index} of sliding window <{min(current_scores.keys())}-{max(current_scores.keys())}>; "
                 f"i.e. at {peak_num_states} states")
    if previous_hypotheses is not None:
        logger.debug(f"Previous peak was at index {prev_peak_index} of sliding window <{min(previous_scores.keys())}-{max(previous_scores.keys())}>; "
                     f"i.e. at {prev_peak_num_states} states")

    if unimodal_peak:
        currently_positioned_correctly = is_positioned_correctly(sliding_window_size, peak_index, current_min_num_states, min_allowed_num_states)
    else:
        # if we don't have an unimodal peak, we don't move the window -> assume that we are positioned correctly
        currently_positioned_correctly = True

    if not currently_positioned_correctly:
        mid_index = sliding_window_size // 2
        diff_to_mid_index = peak_index - mid_index
        min_num_states = current_min_num_states + diff_to_mid_index
        if min_num_states >= min_allowed_num_states:
            logger.debug(f"Moving start of sliding window to {min_num_states} states to position peak at the midpoint.")
        else:
            logger.debug(
                f"Should move start of sliding window to {min_num_states} states to position peak at the midpoint, "
                f"but there are too many outputs in the traces ({num_outputs}); setting min_num_states to {num_outputs}")
            min_num_states = min_allowed_num_states

        return Continue(next_min_num_states=min_num_states)

    if prev_peak_num_states != peak_num_states:
        logger.debug(f"Continue learning with the same window, since the last peak was at a different number of states ({prev_peak_num_states})")
        return Continue()

    peak_hyp, peak_hyp_stoc, peak_pmsat_info = current_hypotheses[peak_num_states]
    prev_peak_hyp, prev_peak_hyp_stoc, prev_peak_pmsat_info = previous_hypotheses[prev_peak_num_states]

    passes_sanity_checks, reason = sanity_checks(peak_hyp)  # TODO not entirely sure about the sanity checks
    if not passes_sanity_checks:
        logger.debug(f"Peak hypothesis did not pass sanity checks: {reason}. Continuing.")
        return Continue()
    logger.debug(f"Peak hypothesis passed sanity checks; checking requirements")

    match termination_mode:
        case GlitchThresholdTermination(threshold):
            percent_glitches = peak_pmsat_info["percent_glitches"]
            msg = (f"Peak hypothesis has {{}} glitches ({percent_glitches}%) than the threshold allowed "
                   f"by the termination mode ({threshold}%). ")
            if percent_glitches <= threshold:
                logger.debug(msg.format("less") + "Terminating.")
                return Terminate(peak_hyp, peak_hyp_stoc, peak_pmsat_info)
            else:
                logger.debug(msg.format("more"))
                return Continue()

        case GlitchImprovementTermination():
            curr_percent_glitches = peak_pmsat_info["percent_glitches"]
            prev_percent_glitches = prev_peak_pmsat_info["percent_glitches"]
            msg = f"Peak hypothesis has {curr_percent_glitches}% glitches, while previous peak hypothesis had {prev_percent_glitches}% glitches. "
            if curr_percent_glitches <= prev_percent_glitches:
                logger.debug(msg + "Terminating.")
                return Terminate(peak_hyp, peak_hyp_stoc, peak_pmsat_info)
            else:
                logger.debug(msg + "Continuing with additional traces.")
                return Continue()

        case _:
            raise NotImplementedError(f"Termination mode {type(termination_mode).__name__} not implemented!")


def get_glitch_percentage(pmsat_info, traces_used_to_learn):
    complete_num_steps = sum(len(trace) for trace in traces_used_to_learn)
    percent_glitches = len(pmsat_info["glitch_steps"]) / complete_num_steps * 100
    return percent_glitches


def is_positioned_correctly(sliding_window_size, peak_index, current_min_num_states, min_allowed_num_states):
    mid_index = sliding_window_size // 2
    if peak_index == mid_index or (peak_index < mid_index and current_min_num_states == min_allowed_num_states):
        logger.debug(f"Window is positioned correctly. Peak is "
                     f"{'exactly in the middle' if peak_index == mid_index else 'on the left side but cannot move further left'}.")
        return True
    return False

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

    if hyp.is_input_complete():
        assert not new_traces, "Should not create new traces if already input complete!"

    return new_traces


def _do_passive_counterexample_processing(sul: SupportedSUL, hypotheses: HypothesesWindow, all_input_combinations: list[tuple[Input, ...]]):
    combinations = itertools.combinations(hypotheses.values(), 2)
    new_traces = []
    for hyp_a, hyp_b in combinations:
        cex = hyp_a.find_distinguishing_seq(hyp_a.initial_state, hyp_b.initial_state)
        new_traces.append(active_pmsatlearn.learnalgo._do_cex_processing(sul, cex, all_input_combinations))

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