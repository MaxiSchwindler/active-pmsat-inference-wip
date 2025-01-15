import copy
import itertools
import logging
from collections import defaultdict
import random
from typing import Literal

from active_pmsatlearn.heuristics import simple_heuristic, intermediary_heuristic, advanced_heuristic
from pmsatlearn import run_pmSATLearn
from active_pmsatlearn.common import *
from active_pmsatlearn.defs import *
from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT
logger = get_logger("APMSL")

INPUT_COMPLETENESS_PROCESSING_DEFAULT = 'random_suffix'
GLITCH_PROCESSING_DEFAULT = 'random_suffix'
REPLAY_GLITCHES_DEFAULT = ('original_suffix', 'random_suffix')


def run_activePmSATLearn(
    # general args
    alphabet: list,
    sul: SupportedSUL,
    automaton_type: Literal["mealy", "moore"],

    # algorithm-specific args
    sliding_window_size: int = 5,
    extension_length: int = 2,
    heuristic_function: str | HeuristicFunction = "intermediary",
    timeout: float | None = None,

    # pmsat args
    pm_strategy: Literal["lsu", "fm", "rc2"] = "rc2",
    cost_scheme: Literal["per_step", "per_transition"] = "per_step",

    # preprocessing
    input_completeness_preprocessing: bool | str = INPUT_COMPLETENESS_PROCESSING_DEFAULT,

    # termination mode
    termination_mode: str | TerminationMode = GlitchImprovementTermination(),

    # postprocessing
    glitch_processing: bool | str = GLITCH_PROCESSING_DEFAULT,
    replay_glitches: bool | Sequence[str] = REPLAY_GLITCHES_DEFAULT,
    window_cex_processing: bool = True,
    random_state_exploration: bool = False,

    # postprocessing, only relevant if termination_mode = EqOracleTermination(...)
    cex_processing: bool = True,
    discard_glitched_traces: bool = True,
    add_cex_as_hard_clauses: bool = True,

    # return/print/debug
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
    :param timeout: timeout for the entire call (in seconds). This currently effectively only times the solver calls,
                    i.e. equivalence or membership queries could be still performed after the timeout has been reached.
                    Also, interrupting the solver may not immediately work, and might thus lead to longer wait times.

    :param pm_strategy: strategy to be used by the solver
    :param cost_scheme: cost scheme for pmsat-learn

    :param input_completeness_preprocessing: whether input completeness processing should be performed before continuing
                                             with the learnt hypotheses

    :param termination_mode: when to terminate learning: if a

    :param glitch_processing: whether glitch processing should be performed, and what type of suffix to append
    :param window_cex_processing: whether window counterexample processing should be performed.
    :param random_state_exploration:
    :param replay_glitches:

    :param cex_processing: whether counterexample processing should be performed; only relevant if eq_oracle is given
    :param discard_glitched_traces: whether to discard traces which clash with robust counterexamples received by the eq_oracle.
    :param add_cex_as_hard_clauses: whether to give robust counterexamples from the eq_oracle as hard clauses to the solver,
                                    forcing them to be included in the learnt hypothesis. However, if the hypothesis to
                                    be learned has not enough states to satisfy all hard clauses (or if two hard clauses
                                    cannot be satisfied at the same time), learning might fail with UNSAT.  # TODO: handle UNSAT properly


    :param return_data: whether to return a dictionary containing learning information
    :param print_level: 0 - None,
                        1 - just results,
                        2 - current round information,
                        3 - pmsat-learn output and logging of traces
    """

    #####################################
    #   INPUT VERIFICATION/DEFAULTS     #
    #####################################
    assert sliding_window_size >= 1, "Sliding window must have a size of at least 1"

    if input_completeness_preprocessing is True:
        input_completeness_preprocessing = INPUT_COMPLETENESS_PROCESSING_DEFAULT
    if glitch_processing is True:
        glitch_processing = GLITCH_PROCESSING_DEFAULT
    if replay_glitches is True:
        replay_glitches = REPLAY_GLITCHES_DEFAULT

    assert glitch_processing in ('random_suffix', 'all_suffixes') or not glitch_processing
    assert (isinstance(replay_glitches, Sequence) and all(g in ('original_suffix', 'random_suffix') for g in replay_glitches)) or not replay_glitches

    if discard_glitched_traces:
        # implementation detail: we store glitchless traces inside a kwargs dict we pass to pmsat-learn.
        # if we want to be able to discard glitched traces WITHOUT passing glitchless_data=[...] to pmsat-learn,
        # we would need to change this.
        assert add_cex_as_hard_clauses, "discard_glitched_traces is currently only supported if also specifying add_cex_as_hard_clauses"

    uses_eq_oracle = isinstance(termination_mode, EqOracleTermination)
    for var in ('cex_processing', 'discard_glitched_traces', 'add_cex_as_hard_clauses'):
        if uses_eq_oracle and not locals()[var]:
            logger.warning(f"Using termination_mode=EqOracleTermination() without {var}. Consider using {var}, "
                           f"as it should lead to considerable performance improvements.")
        elif locals()[var] and not uses_eq_oracle:
            logger.warning(f"{var} is only relevant when using termination_mode=EqOracleTermination(...).")

    match heuristic_function:
        case "simple":
            heuristic_function = simple_heuristic
        case "intermediary":
            heuristic_function = intermediary_heuristic
        case "advanced":
            heuristic_function = advanced_heuristic
        case _:
            assert callable(heuristic_function), "heuristic_function must be either 'simple'/'advanced' or a callable"

    # additional input verification happens in run_pmSATLearn

    #####################################
    #              SETUP                #
    #####################################

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

    common_pmsatlearn_kwargs = dict(
        automata_type=automaton_type,
        pm_strategy=pm_strategy,
        cost_scheme=cost_scheme,
        print_info=print_level == 3,
    )

    if uses_eq_oracle and add_cex_as_hard_clauses:
        common_pmsatlearn_kwargs["glitchless_data"] = []

    common_processing_kwargs = dict(
        sul=sul,
        alphabet=alphabet,
        all_input_combinations=all_input_combinations,
    )

    logger.debug("Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    min_num_states = get_num_outputs(traces)  # start with minimum possible number of states

    #####################################
    #             MAIN LOOP             #
    #####################################

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
                                          **common_pmsatlearn_kwargs
                                          )
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

        if input_completeness_preprocessing:  # TODO: could this lead to an infinite loop?
            preprocessing_additional_traces = do_input_completeness_preprocessing(hyps=hypotheses, suffix_mode=input_completeness_preprocessing,
                                                                    **common_processing_kwargs)

            remove_duplicate_traces(traces, preprocessing_additional_traces)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(preprocessing_additional_traces, detailed_learning_info[learning_rounds],
                                            "input completeness", processing_step="pre")

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
                                    termination_mode=termination_mode,
                                    current_learning_info=detailed_learning_info[learning_rounds])

        match action:
            case Terminate(hyp, hyp_stoc, pmsat_info):
                result = hyp, hyp_stoc, pmsat_info
                break
            case Continue():
                if action.next_min_num_states is not None:
                    min_num_states = action.next_min_num_states
            case _:
                raise ValueError(f"Unexpected action {action}")

        #####################################
        #          POST-PROCESSING          #
        #####################################

        # we can only come here if we did not find a hypothesis we want to return
        logger.debug(f"Beginning postprocessing of hypotheses")

        if glitch_processing:
            postprocessing_additional_traces_glitch = do_glitch_processing(hyps=hypotheses,
                                                                           suffix_mode=glitch_processing,
                                                                           traces_used_to_learn_hyp=traces_used_to_learn,
                                                                           **common_processing_kwargs)

            remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(postprocessing_additional_traces_glitch, detailed_learning_info[learning_rounds], "glitch")

            traces = traces + postprocessing_additional_traces_glitch

        if window_cex_processing:
            postprocessing_additional_traces_window_cex = do_window_counterexample_processing(hypotheses, **common_processing_kwargs)

            remove_duplicate_traces(traces, postprocessing_additional_traces_window_cex)
            log_and_store_additional_traces(postprocessing_additional_traces_window_cex, detailed_learning_info[learning_rounds], "window cex")

            traces = traces + postprocessing_additional_traces_window_cex

        if replay_glitches:
            postprocessing_additional_traces_replay = do_replay_glitches(hyps=hypotheses,
                                                                         traces_used_to_learn=traces_used_to_learn,
                                                                         suffix_modes=replay_glitches,
                                                                         **common_processing_kwargs)

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

        if uses_eq_oracle:
            eq_oracle_cex = action.additional_data["cex"]
            eq_oracle_cex_outputs = action.additional_data["cex_outputs"]
            eq_oracle_cex_trace = traces[0][0], *zip(eq_oracle_cex, eq_oracle_cex_outputs)

            assert eq_oracle_cex is not None, f"Must have cex during postprocessing when using EqOracleTermination!"
            assert eq_oracle_cex_outputs is not None, f"Must have cex outputs during postprocessing when using EqOracleTermination!"

            if cex_processing:
                postprocessing_additional_traces_cex = do_cex_processing(eq_oracle_cex, **common_processing_kwargs)

                remove_duplicate_traces(traces, postprocessing_additional_traces_cex)
                log_and_store_additional_traces(postprocessing_additional_traces_cex, detailed_learning_info[learning_rounds], "cex")

                traces = traces + postprocessing_additional_traces_cex

            if add_cex_as_hard_clauses:
                logger.debug(f"Treating counterexample as a hard clause")
                common_pmsatlearn_kwargs["glitchless_data"].append(eq_oracle_cex_trace)
            else:
                logger.debug(f"Treating counterexample as a soft clause")
                traces.append(eq_oracle_cex_trace)

            if discard_glitched_traces:
                assert add_cex_as_hard_clauses, "discard_glitched_traces works only in combination with add_cex_as_hard_clauses"

                traces_to_remove = set()
                for glitchless_trace in common_pmsatlearn_kwargs["glitchless_data"]:
                    traces_to_remove.update(get_incongruent_traces(glitchless_trace=glitchless_trace, traces=traces))

                logger.debug(f"Removing {len(traces_to_remove)} traces because they were incongruent with counterexample")
                detailed_learning_info[learning_rounds]["removed_traces"] = len(traces_to_remove)
                traces = [t for t_i, t in enumerate(traces) if t_i not in traces_to_remove]

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            logger.warning(f"No additional traces were produced during this round!")

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
            result = get_absolute_peak_hypothesis(previous_hypotheses, previous_scores)

    hyp, hyp_stoc, pmsat_info = result
    if return_data:
        total_time = round(time.time() - start_time, 2)

        active_info = build_info_dict(hyp, sul, learning_rounds, total_time, termination_mode,
                                      pmsat_info, detailed_learning_info, hyp_stoc, timed_out)
        if print_level > 1:
            print_learning_info(active_info)

        return hyp, active_info
    else:
        return hyp


def log_and_store_additional_traces(additional_traces: list[Trace], current_learning_info: dict, name: str, *,
                                    processing_step: str = 'post'):
    logger.debug(f"Produced {len(additional_traces)} additional traces from {name} processing")
    logger.debug_ext(f"Additional traces from {name} processing: {additional_traces}")
    key = f"additional_traces_{processing_step}processing_{'_'.join(name.split(' '))}"
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
        elif not pmsat_info["is_sat"]:
            assert "glitchless_data" in common_pmsatlearn_kwargs, ("Without specifying 'glitchless_data', we should "
                                                                   "never get UNSAT!")
            assert len(common_pmsatlearn_kwargs["glitchless_data"]) > 0
            continue  # TODO: handle unsat! -> learn again without hard clauses?

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


def get_absolute_peak_hypothesis(hypotheses: HypothesesWindow, scores: dict[int, float]) -> PmSatReturnTuple:
    peak_index = find_index_of_absolute_peak(scores)
    peak_num_states = get_num_states_from_scores_index(scores, peak_index)
    return hypotheses[peak_num_states]


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

    # special case for MAT-mode (eq oracle): ask directly whether current peak hypothesis is correct
    continue_kwargs = dict()
    if isinstance(termination_mode, EqOracleTermination):
        eq_oracle = termination_mode.eq_oracle
        logger.debug(f"Asking eq oracle {type(eq_oracle).__name__} for a counterexample...")

        abs_peak_hyp = get_absolute_peak_hypothesis(current_hypotheses, current_scores)
        cex, cex_outputs = eq_oracle.find_cex(abs_peak_hyp[0])
        if cex is None:
            logger.debug(f"Oracle did not find counterexample. Terminating.")
            return Terminate(*abs_peak_hyp)
        else:
            logger.debug(f"Oracle found counterexample. Continuing.")
            logger.debug_ext(f"Counterexample: {cex}")
            continue_kwargs["additional_data"] = dict(cex=cex, cex_outputs=cex_outputs)

    if not unimodal_peak:
        match previous_hypotheses, prev_peak_index:
            case None, None:
                logger.debug("No unimodal peak found in initial round; continuing with same window after collecting more traces.")
                return Continue(**continue_kwargs)
            case _, None:
                logger.debug("No unimodal peak found in current and previous round; using absolute peak.")
                peak_index = current_learning_info["absolute_peak_index"] = find_index_of_absolute_peak(current_scores)
                peak_num_states = current_learning_info["absolute_peak_num_states"] = get_num_states_from_scores_index(current_scores, peak_index)
                prev_peak_index = current_learning_info["absolute_prev_peak_index"] = find_index_of_absolute_peak(previous_scores)
                prev_peak_num_states = current_learning_info["absolute_prev_peak_num_states"] = get_num_states_from_scores_index(previous_scores, prev_peak_index)
            case _, _:
                logger.debug("No unimodal peak found in current round; continuing with same window after collecting more traces")
                return Continue(**continue_kwargs)
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
        # TODO: we could also check for is_positioned_correctly and move the window such that the absolute peak is in the middle
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

        return Continue(next_min_num_states=min_num_states, **continue_kwargs)

    if prev_peak_num_states != peak_num_states:
        logger.debug(f"Continue learning with the same window, since the last peak was at a different number of states ({prev_peak_num_states})")
        return Continue(**continue_kwargs)

    peak_hyp, peak_hyp_stoc, peak_pmsat_info = current_hypotheses[peak_num_states]
    prev_peak_hyp, prev_peak_hyp_stoc, prev_peak_pmsat_info = previous_hypotheses[prev_peak_num_states]

    # passes_sanity_checks, reason = sanity_checks(peak_hyp)  # TODO not entirely sure about the sanity checks
    # if not passes_sanity_checks:
    #     logger.debug(f"Peak hypothesis did not pass sanity checks: {reason}. Continuing.")
    #     return Continue()
    # logger.debug(f"Peak hypothesis passed sanity checks; checking requirements")

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
                return Continue(**continue_kwargs)

        case GlitchImprovementTermination():
            curr_percent_glitches = peak_pmsat_info["percent_glitches"]
            prev_percent_glitches = prev_peak_pmsat_info["percent_glitches"]
            msg = f"Peak hypothesis has {curr_percent_glitches}% glitches, while previous peak hypothesis had {prev_percent_glitches}% glitches. "
            if prev_percent_glitches <= curr_percent_glitches:
                logger.debug(msg + "Terminating.")
                return Terminate(peak_hyp, peak_hyp_stoc, peak_pmsat_info)
            else:
                logger.debug(msg + "Continuing with additional traces.")
                return Continue(**continue_kwargs)

        case EqOracleTermination(eq_oracle):
            # if the eq oracle had no counterexample, we would have already returned
            assert 'cex' in continue_kwargs.get('additional_data', {})
            return Continue(**continue_kwargs)

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


def do_window_counterexample_processing(hypotheses: HypothesesWindow, sul: SupportedSUL, alphabet: list[Input],
                                        all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    'Counterexample processing' on a window of hypotheses.
    For each pair of 'adjacent' hypotheses (i.e. they have n and n+1 states), get the distinguishing sequence
    and perform ARPNI counterexample processing on it.
    :param hypotheses: Hypotheses window
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: input combinations
    """
    logger.debug(f"Do window counterexample processing for hypotheses window of size {len(hypotheses)}...")
    num_states = sorted(hypotheses.keys())
    combinations = [(hypotheses[num_states[i]][0], hypotheses[num_states[i + 1]][0]) for i in range(len(num_states) - 1)]

    new_traces = []
    performed_prefixes = set()

    for hyp_a, hyp_b in combinations:
        assert len(hyp_a.states) + 1 == len(hyp_b.states)  # only compare adjacent hypotheses
        cex = hyp_a.find_distinguishing_seq(hyp_a.initial_state, hyp_b.initial_state, alphabet)
        if cex is not None and tuple(cex) not in performed_prefixes:
            logger.debug_ext(f"CEX between {len(hyp_a.states)}-state hypothesis {id(hyp_a)} and {len(hyp_b.states)}-state hypothesis {id(hyp_b)}: {cex}")
            new_traces.extend(do_cex_processing(cex=cex, sul=sul, alphabet=alphabet, all_input_combinations=all_input_combinations))
            for p in get_prefixes(cex):
                performed_prefixes.add(tuple(p))

    return new_traces


def do_replay_glitches(hyps: HypothesesWindow, traces_used_to_learn: list[Trace], suffix_modes: Sequence[str],
                       sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Replay the traces which pmsat_learn marked as containing glitches.
    :param hyps: hypotheses window
    :param pmsat_info: info dict returned from pmsat_learn call when learning hyp
    :param traces_used_to_learn: traces used to learn hyp
    :param suffix_modes: which suffixes to append after a glitch step. Choose from:
                     'original_suffix': continue with the original trace; i.e. each trace containing a glitch is replayed as a whole
                     'random_suffix': after a glitch step, continue with a random input sequence from alphabet^extension_length
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all input combinations, alphabet^extension_length
    :return: list of traces
    """
    new_traces = []
    replayed_trace_indices = set()
    replayed_trace_step_pairs = set()

    for hyp, hyp_stoc, pmsat_info in hyps.values():
        hyp_str = f"{len(hyp.states)}-state hypothesis {id(hyp)}"
        logger.debug(f"Replaying traces with glitched transitions of {hyp_str}. {suffix_modes=}")

        for trace_index, step_index in pmsat_info["glitch_steps"]:
            glitch_trace = traces_used_to_learn[trace_index]

            if 'original_suffix' in suffix_modes:
                if trace_index not in replayed_trace_indices:
                    # replay whole trace
                    new_traces.append(trace_query(sul, [i for i, o in glitch_trace[1:]]))
                    replayed_trace_indices.add(trace_index)

            if 'random_suffix' in suffix_modes:
                if (trace_index, step_index) not in replayed_trace_step_pairs:
                    # replay trace until glitch, then append random suffix
                    random_suffix = random.choice(all_input_combinations)
                    new_traces.append(trace_query(sul, [i for i, o in glitch_trace[1:step_index+1]] + list(random_suffix)))
                    replayed_trace_step_pairs.add((trace_index, step_index))

    return new_traces


def build_info_dict(hyp, sul, learning_rounds, total_time, termination_mode,
                    last_pmsat_info, detailed_learning_info, hyp_stoc, timed_out):
    active_info = {
        'learning_rounds': learning_rounds,
        'learned_automaton_size': hyp.size if hyp is not None else None,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': termination_mode.eq_oracle.num_queries if isinstance(termination_mode, EqOracleTermination) else None,
        'steps_eq_oracle': termination_mode.eq_oracle.num_queries if isinstance(termination_mode, EqOracleTermination) else None,
        'eq_oracle_time': termination_mode.eq_oracle.eq_query_time if isinstance(termination_mode, EqOracleTermination) else None,
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
    print('Pre/Postprocessing:')
    for method_name, total_num in get_total_num_additional_traces(info['detailed_learning_info']).items():
        print(f"  {method_name}: {total_num} additional traces")
    print('-----------------------------------')


def get_total_num_additional_traces(detailed_learning_info: dict[int, dict]):
    sums = defaultdict(int)
    for info in detailed_learning_info.values():
        for key, value in info.items():
            if not key.startswith("additional_traces_"):
                continue
            key = key[len("additional_traces_"):]
            sums[key] += len(value)
    return sums