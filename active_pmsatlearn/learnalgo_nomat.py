import copy
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Literal

from active_pmsatlearn.heuristics import simple_heuristic, intermediary_heuristic, advanced_heuristic
from pmsatlearn import run_pmSATLearn
from active_pmsatlearn.common import do_cex_processing, do_glitch_processing, do_input_completeness_processing
from active_pmsatlearn.defs import *
from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT
logger = get_logger("APMSL")


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


@dataclass
class GlitchThresholdTermination(RequirementsTermination):
    threshold: float = 1.0


class ImprovementTermination(TerminationMode):
    pass


class ScoreImprovementTermination(ImprovementTermination):
    pass


class GlitchImprovementTermination(ImprovementTermination):
    pass


def run_activePmSATLearn(
    alphabet: list,
    sul: SupportedSUL,
    automaton_type: Literal["mealy", "moore"],
    extension_length: int = 2,
    sliding_window_size: int = 5,
    heuristic_function: str | HeuristicFunction = "intermediary",
    pm_strategy: Literal["lsu", "fm", "rc2"] = "rc2",
    timeout: int | float | None = None,
    cost_scheme: Literal["per_step", "per_transition"] = "per_step",
    input_completeness_processing: bool = False,
    glitch_processing: bool = False,
    replay_glitches: bool = True,
    counterexample_processing: bool = True,
    discard_glitched_traces: bool = False,
    random_state_exploration: bool = False,
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
    assert sliding_window_size >= 3, "Sliding window must have a size of at least 3"
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

    result = None, None, None
    previous_hypotheses = None
    previous_scores = None

    common_processing_kwargs = dict(
        sul=sul,
        alphabet=alphabet,
        all_input_combinations=all_input_combinations,
    )

    logger.debug("Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    min_num_states = get_num_outputs(traces)  # start with minimum possible number of states

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
        assert not any((info["timed_out"] or (h is None)) for h, h_s, info in hypotheses.values()), "Sliding window should not contained timed-out hypotheses"
        traces_used_to_learn = copy.deepcopy(traces)

        if len(hypotheses) != sliding_window_size:
            # if we have a timeout during learning of the sliding window, no further hypotheses will be learned;
            # the returned hypotheses window contains only those that were successfully learnt
            last_learnt_num_states = max(hypotheses.keys()) if len(hypotheses) != 0 else min_num_states - 1
            logger.debug(f"Learning of all hypotheses after {last_learnt_num_states} states timed out. ")
            detailed_learning_info[learning_rounds]["timed_out"] = True
            detailed_learning_info[learning_rounds]["last_learnt"] = last_learnt_num_states

            if (last_learnt_num_states + 1) <= (min_num_states + (sliding_window_size // 2)):
                logger.debug(f"First timed-out hypothesis was in left side of sliding window. Returning peak hypothesis from previous round.")
                timed_out = True
                break
            else:
                logger.debug(f"First timed-out hypothesis was in right side of sliding window. Continuing with truncated window.")

        #####################################
        #           PREPROCESSING           #
        #####################################

        if input_completeness_processing:
            # TODO: could this lead to an infinite loop?
            preprocessing_additional_traces = []
            for hyp, _, _ in hypotheses.values():
                additional_traces = do_input_completeness_processing(hyp=hyp, **common_processing_kwargs)
                preprocessing_additional_traces.extend(additional_traces)

            remove_duplicate_traces(traces, preprocessing_additional_traces)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(preprocessing_additional_traces, detailed_learning_info[learning_rounds],
                                            "input completeness", "preprocessing_additional_traces")

            if preprocessing_additional_traces:
                traces += preprocessing_additional_traces
                continue
        else:
            # make input complete anyways, such that we don't have to check whether transitions exist during postprocessing
            for hyp, _, _ in hypotheses.values():
                if not hyp.is_input_complete():
                    logger.debug(f"Input completeness processing is deactivated, but {len(hyp.states)}-state hypothesis "
                                 f"{id(hyp)} was not input complete. Force input completeness via self-loops.")
                    hyp.make_input_complete()

        #####################################
        #              SCORING              #
        #####################################

        scores = {num_states: heuristic_function(hyp, info, traces) for num_states, (hyp, _, info) in hypotheses.items()}
        assert sorted(scores.keys()) == list(scores.keys())
        logger.debug(f"Calculated the following scores via heuristic function '{heuristic_function.__name__}': {scores}")
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
                result = hyp, hyp_stoc, pmsat_info
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
            postprocessing_additional_traces_glitch = []
            for hyp, hyp_stoc, pmsat_info in hypotheses.values():
                additional_traces = do_glitch_processing(hyp=hyp, pmsat_info=pmsat_info, hyp_stoc=hyp_stoc,
                                                         traces_used_to_learn_hyp=traces_used_to_learn, **common_processing_kwargs)
                postprocessing_additional_traces_glitch.extend(additional_traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(postprocessing_additional_traces_glitch, detailed_learning_info[learning_rounds], "glitch")

            traces = traces + postprocessing_additional_traces_glitch

        if counterexample_processing:
            postprocessing_additional_traces_counterexample = do_passive_counterexample_processing(hypotheses, **common_processing_kwargs)

            remove_duplicate_traces(traces, postprocessing_additional_traces_counterexample)
            log_and_store_additional_traces(postprocessing_additional_traces_counterexample, detailed_learning_info[learning_rounds], "counterexample")

            traces = traces + postprocessing_additional_traces_counterexample

        if replay_glitches:
            postprocessing_additional_traces_replay = []
            for hyp, hyp_stoc, pmsat_info in hypotheses.values():
                additional_traces = do_replay_glitches(pmsat_info=pmsat_info, traces_used_to_learn=traces_used_to_learn, hyp=hyp, **common_processing_kwargs)
                postprocessing_additional_traces_replay.extend(additional_traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_replay)
            log_and_store_additional_traces(postprocessing_additional_traces_replay, detailed_learning_info[learning_rounds], "replay")

            traces = traces + postprocessing_additional_traces_replay

        if random_state_exploration:
            postprocessing_additional_traces_state_expl = []
            for hyp, hyp_stoc, pmsat_info in hypotheses.values():
                additional_traces = do_state_exploration(hyp=hyp, **common_processing_kwargs)
                postprocessing_additional_traces_state_expl.extend(additional_traces)

            remove_duplicate_traces(traces, postprocessing_additional_traces_state_expl)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(postprocessing_additional_traces_state_expl, detailed_learning_info[learning_rounds], "state exploration")

            traces = traces + postprocessing_additional_traces_state_expl

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            logger.warning(f"No additional traces were produced during this round!")

        if discard_glitched_traces:
            raise NotImplementedError

        previous_hypotheses = hypotheses
        previous_scores = scores
        min_num_states = max(min_num_states, get_num_outputs(traces))

    #####################################
    #          RETURN RESULT            #
    #####################################

    if timed_out:
        logger.warning(f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")
        if previous_scores:
            logger.debug(f"Returning peak hypothesis of previous round")
            previous_peak = find_index_of_absolute_peak(previous_scores)
            previous_peak_num_states = get_num_states_from_scores_index(previous_scores, previous_peak)
            result = previous_hypotheses[previous_peak_num_states]

    hyp, hyp_stoc, pmsat_info = result
    if return_data:
        total_time = round(time.time() - start_time, 2)

        active_info = build_info_dict(hyp, sul, learning_rounds, total_time,
                                      pmsat_info, detailed_learning_info, hyp_stoc, timed_out)
        if print_level > 1:
            print_learning_info(active_info)

        return hyp, active_info
    else:
        return hyp


def log_and_store_additional_traces(additional_traces: list[Trace], current_learning_info: dict, name: str, key: str = None):
    logger.debug(f"Produced {len(additional_traces)} additional traces from {name} processing")
    logger.debug_ext(f"Additional traces from {name} processing: {additional_traces}")
    if key is None:
        key = f"postprocessing_additional_traces_{'_'.join(name)}"
    current_learning_info[key] = additional_traces


def learn_sliding_window(sliding_window_size: int, min_num_states: int, traces: list[Trace], must_end_by: float = 0.0,
                         **common_pmsatlearn_kwargs) -> HypothesesWindow:
    logger.debug(f"Running pmSATLearn to learn {sliding_window_size} hypotheses with between {min_num_states} and "
                 f"{min_num_states + sliding_window_size - 1} states from {len(traces)} traces")
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
                prev_peak_index = current_learning_info["absolute_prev_peak_index"] = find_index_of_absolute_peak(previous_scores)
                prev_peak_num_states = current_learning_info["absolute_prev_peak_num_states"] = get_num_states_from_scores_index(previous_scores, prev_peak_index)
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


def do_state_exploration(hyp: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input],
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


def do_passive_counterexample_processing(hypotheses: HypothesesWindow, sul: SupportedSUL, alphabet: list[Input],
                                         all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    logger.debug(f"Do passive counterexample processing for hypotheses window of size {len(hypotheses)}...")
    hyps: list[SupportedAutomaton] = [h for h, h_s, i in hypotheses.values()]
    combinations = itertools.combinations(hyps, 2)
    new_traces = []
    for hyp_a, hyp_b in combinations:
        cex = hyp_a.find_distinguishing_seq(hyp_a.initial_state, hyp_b.initial_state, alphabet)
        if cex is not None:
            logger.debug_ext(f"CEX between {len(hyp_a.states)}-state hypothesis {id(hyp_a)} and {len(hyp_b.states)}-state hypothesis {id(hyp_b)}: {cex}")
            new_traces.extend(do_cex_processing(cex=cex, sul=sul, alphabet=alphabet, all_input_combinations=all_input_combinations))

    return new_traces


def do_replay_glitches(pmsat_info: PmSatLearningInfo, traces_used_to_learn: list[Trace], hyp: SupportedAutomaton,
                       sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    num_glitch_traces = len(set(trace_index for trace_index, step_index in pmsat_info["glitch_steps"]))
    logger.debug(f"Replaying {num_glitch_traces} traces with glitched transitions of {len(hyp.states)}-state hypothesis {id(hyp)}...")

    new_traces = []
    for trace_index, step_index in pmsat_info["glitch_steps"]:
        glitch_trace = traces_used_to_learn[trace_index]
        new_traces.append(trace_query(sul, [i for i, o in glitch_trace[1:]]))

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