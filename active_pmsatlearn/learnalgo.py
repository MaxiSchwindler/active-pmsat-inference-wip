import copy
import itertools
import logging
import multiprocessing
from typing import Literal, Optional

from pebble import concurrent

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.defs import *
from active_pmsatlearn.heuristics import *
from active_pmsatlearn.processing import *
from active_pmsatlearn.utils import *
from active_pmsatlearn.log import get_logger, DEBUG_EXT, set_current_process_name

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
    deduplicate_traces: bool = True,
    only_replay_peak_glitches: bool = False,  #TODO: combine with replay_glitches (dict?)
    only_add_incongruent_traces: bool = False,

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
    window_cex_processing: bool = False,
    random_steps_per_round: int = 0,
    transition_coverage_steps: int = 0,

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
    :param random_steps_per_round: either False or a tuple of integers (num_walks, min_walk_len, max_walk_len)
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

    assert input_completeness_preprocessing in ('random_suffix', 'all_suffixes') or not input_completeness_preprocessing
    assert glitch_processing in ('random_suffix', 'all_suffixes') or not glitch_processing
    assert (isinstance(replay_glitches, Sequence) and all(g in ('original_suffix', 'random_suffix') for g in replay_glitches)) or not replay_glitches
    assert isinstance(random_steps_per_round, int)
    assert isinstance(transition_coverage_steps, int)

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
    logger.debug("Running ActivePMSATLearn...")
    detailed_learning_info = defaultdict(dict)

    start_time = time.time()
    must_end_by = start_time + timeout if timeout is not None else 100 * 365 * 24 * 60 * 60  # no timeout -> must end in 100 years :)
    timed_out = False
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

    while True:
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
        assert len(hypotheses) == sliding_window_size
        traces_used_to_learn = copy.deepcopy(traces)
        detailed_learning_info[learning_rounds]["traces_used_to_learn"] = traces_used_to_learn

        # remove timed out hyps from hypotheses window
        timed_out_num_states = [n for n, r in hypotheses.items() if r == "timed_out"]
        if timed_out_num_states:
            timed_out = True
            for n in timed_out_num_states:
                del hypotheses[n]

        # remove unsat hyps from hypotheses window (can only have unsat hyps if using glitchless_data)
        unsat_num_states = [n for n, r in hypotheses.items() if r == "unsat"]
        if unsat_num_states:
            assert "glitchless_data" in common_pmsatlearn_kwargs, (
                "Without specifying 'glitchless_data', we should never get UNSAT!"
            )
            assert len(common_pmsatlearn_kwargs["glitchless_data"]) > 0

            for n in unsat_num_states:
                del hypotheses[n]

        detailed_learning_info[learning_rounds]["pmsat_info"] = {ns: info for ns, (h, hs, info) in hypotheses.items()}
        detailed_learning_info[learning_rounds]["hyp"] = {ns: str(h) for ns, (h, hs, info) in hypotheses.items()}
        detailed_learning_info[learning_rounds]["hyp_stoc"] = {ns: str(hs) for ns, (h, hs, info) in hypotheses.items()}

        #####################################
        #           PREPROCESSING           #
        #####################################

        if input_completeness_preprocessing and not timed_out:  # TODO: could this lead to an infinite loop?
            preprocessing_additional_traces = do_input_completeness_preprocessing(hyps=hypotheses, suffix_mode=input_completeness_preprocessing,
                                                                                  **common_processing_kwargs)

            if deduplicate_traces:
                remove_duplicate_traces(traces, preprocessing_additional_traces)  # TODO: does de-duplication affect anything? check!
            log_and_store_additional_traces(preprocessing_additional_traces, detailed_learning_info[learning_rounds],
                                            "input completeness", processing_step="pre")

            if preprocessing_additional_traces:
                traces += preprocessing_additional_traces
                min_num_states = max(min_num_states, get_num_outputs(traces))
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

        terminate, hyp, additional_data = should_terminate(hypotheses, scores,
                                                           previous_hypotheses, previous_scores,
                                                           current_min_num_states=min_num_states,
                                                           traces_used_to_learn=traces,
                                                           termination_mode=termination_mode,
                                                           current_learning_info=detailed_learning_info[learning_rounds],
                                                           timed_out=timed_out)

        if terminate:
            result = hyp
            break

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

            if deduplicate_traces:
                remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)  # TODO: does de-duplication affect anything? check!

            if only_add_incongruent_traces:
                peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                remove_congruent_traces(peak_hyp, postprocessing_additional_traces_glitch)

            log_and_store_additional_traces(postprocessing_additional_traces_glitch, detailed_learning_info[learning_rounds], "glitch")

            traces = traces + postprocessing_additional_traces_glitch

        if window_cex_processing:
            postprocessing_additional_traces_window_cex = do_window_counterexample_processing(hypotheses, **common_processing_kwargs)

            if deduplicate_traces:
                remove_duplicate_traces(traces, postprocessing_additional_traces_window_cex)

            if only_add_incongruent_traces:
                peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                remove_congruent_traces(peak_hyp, postprocessing_additional_traces_window_cex)

            log_and_store_additional_traces(postprocessing_additional_traces_window_cex, detailed_learning_info[learning_rounds], "window cex")

            traces = traces + postprocessing_additional_traces_window_cex

        if replay_glitches:
            if only_replay_peak_glitches:
                peak_index = find_index_of_absolute_peak(scores)
                peak_num_states = get_num_states_from_scores_index(scores, peak_index)
                fake_hyp_window = {peak_num_states: hypotheses[peak_num_states]}  # used to only replay glitches of peak
            else:
                fake_hyp_window = hypotheses
            postprocessing_additional_traces_replay = do_replay_glitches(hyps=fake_hyp_window,
                                                                         traces_used_to_learn=traces_used_to_learn,
                                                                         suffix_modes=replay_glitches,
                                                                         **common_processing_kwargs)

            if deduplicate_traces:
                remove_duplicate_traces(traces, postprocessing_additional_traces_replay)

            if only_add_incongruent_traces:
                peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                remove_congruent_traces(peak_hyp, postprocessing_additional_traces_replay)

            log_and_store_additional_traces(postprocessing_additional_traces_replay, detailed_learning_info[learning_rounds], "replay")

            traces = traces + postprocessing_additional_traces_replay

        if random_steps_per_round:
            postprocessing_additional_traces_random_walks = do_random_walks(random_steps_per_round, **common_processing_kwargs)

            if deduplicate_traces:
                remove_duplicate_traces(traces, postprocessing_additional_traces_random_walks)  # TODO: does de-duplication affect anything? check!

            if only_add_incongruent_traces:
                peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                remove_congruent_traces(peak_hyp, postprocessing_additional_traces_random_walks)

            log_and_store_additional_traces(postprocessing_additional_traces_random_walks, detailed_learning_info[learning_rounds], "random walks")

            traces = traces + postprocessing_additional_traces_random_walks

        if transition_coverage_steps:
            peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
            postprocessing_additional_traces_state_coverage = do_transition_coverage(peak_hyp, transition_coverage_steps, **common_processing_kwargs)

            if deduplicate_traces:
                remove_duplicate_traces(traces, postprocessing_additional_traces_state_coverage)

            if only_add_incongruent_traces:
                peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                remove_congruent_traces(peak_hyp, postprocessing_additional_traces_state_coverage)

            log_and_store_additional_traces(postprocessing_additional_traces_state_coverage, detailed_learning_info[learning_rounds], "transition coverage")

            traces = traces + postprocessing_additional_traces_state_coverage

        if uses_eq_oracle:
            eq_oracle_cex = additional_data["cex"]
            eq_oracle_cex_outputs = additional_data["cex_outputs"]
            eq_oracle_cex_trace = traces[0][0], *zip(eq_oracle_cex, eq_oracle_cex_outputs)

            assert eq_oracle_cex is not None, f"Must have cex during postprocessing when using EqOracleTermination!"
            assert eq_oracle_cex_outputs is not None, f"Must have cex outputs during postprocessing when using EqOracleTermination!"

            if cex_processing:
                postprocessing_additional_traces_cex = do_cex_processing(eq_oracle_cex, **common_processing_kwargs)

                if deduplicate_traces:
                    remove_duplicate_traces(traces, postprocessing_additional_traces_cex)

                if only_add_incongruent_traces:
                    peak_hyp = get_absolute_peak_hypothesis(hypotheses, scores)[0]
                    remove_congruent_traces(peak_hyp, postprocessing_additional_traces_cex)

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

        min_num_states = calculate_next_min_num_states(hypotheses, scores, get_num_outputs(traces))
        assert min_num_states >= get_num_outputs(traces)
        previous_hypotheses = hypotheses
        previous_scores = scores

    #####################################
    #          RETURN RESULT            #
    #####################################

    if timed_out:
        logger.warning(f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")

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
                                    processing_step: str = 'post', store: bool = True):
    logger.debug(f"Produced {len(additional_traces)} additional traces from {name} processing")
    logger.debug_ext(f"Additional traces from {name} processing: {additional_traces}")
    key = f"additional_traces_{processing_step}processing_{'_'.join(name.split(' '))}"
    current_learning_info[f"num_{key}"] = len(additional_traces)
    if store:
        current_learning_info[key] = additional_traces


@concurrent.process(daemon=False)
def learn_single_hypothesis(num_states, traces, must_end_by, **common_pmsatlearn_kwargs):
    set_current_process_name(f"LEARN_{num_states}s_HYP_{multiprocessing.current_process().pid}")
    logger.debug(f"Learning {num_states}-state hypothesis...")

    hyp, h_stoc, pmsat_info = run_pmSATLearn(
        data=traces,
        n_states=num_states,
        timeout=max(must_end_by - time.time(), 0),
        **common_pmsatlearn_kwargs
    )

    if pmsat_info["timed_out"]:
        return num_states, "timed_out"
    elif not pmsat_info["is_sat"]:
        assert "glitchless_data" in common_pmsatlearn_kwargs, (
            "Without specifying 'glitchless_data', we should never get UNSAT!"
        )
        assert len(common_pmsatlearn_kwargs["glitchless_data"]) > 0
        return num_states, "unsat"

    pmsat_info["percent_glitches"] = get_glitch_percentage(pmsat_info, traces)

    logger.debug(f"Finished learning {num_states}-state hypothesis.")
    return num_states, (hyp, h_stoc, pmsat_info)


def learn_sliding_window(sliding_window_size: int, min_num_states: int, traces: list[Trace],
                         must_end_by: float = 0.0, **common_pmsatlearn_kwargs) -> HypothesesWindow:
    logger.debug(f"Running pmSATLearn to learn {sliding_window_size} hypotheses with between {min_num_states} and "
                 f"{min_num_states + sliding_window_size - 1} states from {len(traces)} traces")

    num_outputs = get_num_outputs(traces)
    assert min_num_states >= num_outputs
    learned = {}

    futures = [
        learn_single_hypothesis(min_num_states + i, traces, must_end_by, **common_pmsatlearn_kwargs)
        for i in range(sliding_window_size)
    ]

    for future in futures:
        num_states, result = future.result()
        learned[num_states] = result

    return learned


def calculate_next_min_num_states(hypotheses, scores):
    sliding_window_size = len(hypotheses)
    peak_index = find_index_of_absolute_peak(scores)
    peak_num_states = get_num_states_from_scores_index(scores, peak_index)

    return peak_num_states - (sliding_window_size // 2)


def find_index_of_unimodal_peak(scores: dict[int, float]) -> int | None:
    if scores is None:
        return None
    vals = list(scores.values())
    peak_index = vals.index(max(vals))
    if all(vals[i] <= vals[i + 1] for i in range(0, peak_index)) and all(vals[j] >= vals[j+1] for j in range(peak_index, len(vals) - 1)):
        return peak_index


def find_index_of_absolute_peak(scores: dict[int, float]) -> int | None:
    if not scores:
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


def should_terminate(current_hypotheses: HypothesesWindow, current_scores: dict[int, float],
                     previous_hypotheses: HypothesesWindow | None, previous_scores: dict[int, float] | None,
                     current_min_num_states: int, traces_used_to_learn: list[Trace],
                     termination_mode: TerminationMode,
                     current_learning_info: dict[str, Any],
                     timed_out: bool) -> tuple[bool, Optional[PmSatReturnTuple], Optional[dict]]:
    peak_index = current_learning_info["peak_index"] = find_index_of_absolute_peak(current_scores)
    peak_num_states = current_learning_info["peak_num_states"] = get_num_states_from_scores_index(current_scores, peak_index)

    prev_peak_index = find_index_of_absolute_peak(previous_scores)
    prev_peak_num_states = get_num_states_from_scores_index(previous_scores, prev_peak_index)

    sliding_window_size = len(current_scores)
    min_allowed_num_states = get_num_outputs(traces_used_to_learn)

    # special case for timeouts:
    if timed_out:
        if previous_hypotheses and prev_peak_num_states not in current_hypotheses.keys():
            logger.debug(f"Since the PMSAT-LEARN call for the previous peak num states ({prev_peak_num_states}) timed out, "
                         f"a comparison of the remaining (not timed-out) current hypotheses does not make sense. "
                         f"Terminating with previous peak hypothesis.")
            hyp = previous_hypotheses[prev_peak_num_states]
        else:
            logger.debug(f"Some PMSAT-LEARN calls timed out, but the previous peak num states hypothesis ({prev_peak_num_states} states) "
                         f"was successfully learnt again with more data; therefore, return the peak hypothesis "
                         f"({peak_num_states} states) of the current window.")
            hyp = current_hypotheses[peak_num_states]
        return True, hyp, None

    # special case for MAT-mode (eq oracle): ask directly whether current peak hypothesis is correct
    additional_data = dict()
    if isinstance(termination_mode, EqOracleTermination):
        eq_oracle = termination_mode.eq_oracle
        logger.debug(f"Asking eq oracle {type(eq_oracle).__name__} for a counterexample...")

        abs_peak_hyp = get_absolute_peak_hypothesis(current_hypotheses, current_scores)
        cex, cex_outputs = eq_oracle.find_cex(abs_peak_hyp[0])
        if cex is None:
            logger.debug(f"Oracle did not find counterexample. Terminating.")
            return True, abs_peak_hyp, None
        else:
            logger.debug(f"Oracle found counterexample. Continuing.")
            logger.debug_ext(f"Counterexample: {cex}")
            additional_data = dict(cex=cex, cex_outputs=cex_outputs)

    if previous_hypotheses is None:
        logger.debug("No previous hypotheses; learn again with same window.")
        return False, None, additional_data

    logger.debug(f"Found peak at index {peak_index} of sliding window <{min(current_scores.keys())}-{max(current_scores.keys())}>; "
                 f"i.e. at {peak_num_states} states")
    logger.debug(f"Previous peak was at index {prev_peak_index} of sliding window <{min(previous_scores.keys())}-{max(previous_scores.keys())}>; "
                 f"i.e. at {prev_peak_num_states} states")

    if not is_positioned_correctly(sliding_window_size, peak_index, current_min_num_states, min_allowed_num_states):
        return False, None, additional_data

    # TODO do we want this?
    if prev_peak_num_states != peak_num_states:
        logger.debug(f"Continue learning with the same window, since the last peak was at a different number of states ({prev_peak_num_states})")
        return False, None, additional_data

    peak_hyp, peak_hyp_stoc, peak_pmsat_info = current_hypotheses[peak_num_states]
    prev_peak_hyp, prev_peak_hyp_stoc, prev_peak_pmsat_info = previous_hypotheses[prev_peak_num_states]

    match termination_mode:
        case GlitchThresholdTermination(threshold):
            percent_glitches = peak_pmsat_info["percent_glitches"]
            msg = (f"Peak hypothesis has {{}} glitches ({percent_glitches}%) than the threshold allowed "
                   f"by the termination mode ({threshold}%). ")
            if percent_glitches <= threshold:
                if termination_mode.first_time:
                    logger.debug(msg.format("less") + "Terminating the first time we are below threshold.")
                    return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
                else:
                    prev_percent_glitches = prev_peak_pmsat_info["percent_glitches"]
                    msg = msg.format("less") + f"Last peak hypothesis had {prev_percent_glitches}%. "
                    if prev_percent_glitches <= threshold:
                        logger.debug(msg + "Terminating.")
                        return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
                    else:
                        logger.debug(msg + "Continuing.")
                        return False, None, additional_data
            else:
                logger.debug(msg.format("more"))
                return False, None, additional_data

        case GlitchImprovementTermination():
            curr_percent_glitches = peak_pmsat_info["percent_glitches"]
            prev_percent_glitches = prev_peak_pmsat_info["percent_glitches"]
            msg = f"Peak hypothesis has {curr_percent_glitches}% glitches, while previous peak hypothesis had {prev_percent_glitches}% glitches. "
            if prev_percent_glitches <= curr_percent_glitches:
                logger.debug(msg + "Terminating.")
                return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
            else:
                logger.debug(msg + "Continuing with additional traces.")
                return False, None, additional_data

        case ApproximateScoreImprovementTermination():
            curr_approx_score = approximate_advanced_heuristic(peak_hyp, peak_pmsat_info, traces_used_to_learn)
            prev_approx_score = approximate_advanced_heuristic(prev_peak_hyp, peak_pmsat_info, traces_used_to_learn)
            msg = f"Peak hypothesis has approximate score of {curr_approx_score}  while previous peak hypothesis has {prev_approx_score}. "
            if prev_approx_score >= curr_approx_score:
                logger.debug(msg + "Terminating.")
                return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
            else:
                logger.debug(msg + "Continuing with additional traces.")
                return False, None, additional_data

        case ScoreImprovementTermination():
            curr_score = current_scores[peak_num_states]
            prev_score = previous_scores[prev_peak_num_states]
            msg = f"Peak hypothesis has score of {curr_score} while previous peak hypothesis has {prev_score}. "
            if prev_score >= curr_score:
                logger.debug(msg + "Terminating.")
                return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
            else:
                logger.debug(msg + "Continuing with additional traces.")
                return False, None, additional_data

        case HypothesisDoesNotChangeTermination():
            dis = peak_hyp.find_distinguishing_seq(peak_hyp.initial_state, prev_peak_hyp.initial_state, peak_hyp.get_input_alphabet())
            if dis is None:
                logger.debug(f"No distinguishing sequence between current hypothesis and previous hypothesis found. Terminating")
                return True, (peak_hyp, peak_hyp_stoc, peak_pmsat_info), None
            else:
                logger.debug(f"Found distinguishing sequence between current and previous hypothesis. Continue.")
                return False, None, additional_data

        case EqOracleTermination(eq_oracle):
            # if the eq oracle had no counterexample, we would have already returned
            assert 'cex' in additional_data.get('additional_data', {})
            return False, None, additional_data

        case _:
            raise NotImplementedError(f"Termination mode {type(termination_mode).__name__} not implemented!")


def get_glitch_percentage(pmsat_info, traces_used_to_learn):
    complete_num_steps = sum(len(trace[1:]) for trace in traces_used_to_learn)  # !!!
    percent_glitches = len(pmsat_info["glitch_steps"]) / complete_num_steps * 100
    return percent_glitches


def is_positioned_correctly(sliding_window_size, peak_index, current_min_num_states, min_allowed_num_states):
    mid_index = sliding_window_size // 2
    if peak_index == mid_index or (peak_index < mid_index and current_min_num_states == min_allowed_num_states):
        logger.debug(f"Window is positioned correctly. Peak is "
                     f"{'exactly in the middle' if peak_index == mid_index else 'on the left side but cannot move further left'}.")
        return True
    return False


def build_info_dict(hyp, sul, learning_rounds, total_time, termination_mode,
                    last_pmsat_info, detailed_learning_info, hyp_stoc, timed_out):
    active_info = {
        'learning_rounds': learning_rounds,
        'learned_automaton_size': hyp.size if hyp is not None else 0,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': termination_mode.eq_oracle.num_queries if isinstance(termination_mode, EqOracleTermination) else 0,
        'steps_eq_oracle': termination_mode.eq_oracle.num_steps if isinstance(termination_mode, EqOracleTermination) else 0,
        'eq_oracle_time': termination_mode.eq_oracle.eq_query_time if isinstance(termination_mode, EqOracleTermination) else 0,
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