import copy
import logging
import time
import itertools
from collections import defaultdict
from typing import Literal, Any, Sequence, TypeAlias

from aalpy import SUL, MealyMachine, MooreMachine, bisimilar
from active_pmsatlearn import RobustEqOracleMixin
from active_pmsatlearn.learnalgo import log_and_store_additional_traces, get_total_num_additional_traces

from active_pmsatlearn.log import get_logger, DEBUG_EXT
from active_pmsatlearn.processing import do_input_completeness_preprocessing, do_random_walks, do_cex_processing
from active_pmsatlearn.utils import trace_query
from . import gsm

logger = get_logger("ActiveGSM")

SupportedAutomaton: TypeAlias = MooreMachine | MealyMachine
INPUT_COMPLETENESS_PROCESSING_DEFAULT = 'random_suffix'


def run_activeGSM(
    alphabet: list,
    sul: SUL,
    automaton_type: Literal["mealy", "moore"],

    failure_rate: float,
    certainty: float,
    purge_mismatches=False,
    instrument=None,

    eq_oracle: RobustEqOracleMixin = None,
    use_dis_as_cex: bool = False,
    input_completeness_preprocessing: bool | str = False,
    extension_length: int = 2,
    random_steps_per_round: int = 200,

    timeout: float | None = None,
    return_data: bool = True,
    print_level: int | None = 2,
) -> (
        SupportedAutomaton | None | tuple[SupportedAutomaton | None, dict[str, Any]]
):
    """
    Run GSM with GlitchyDeterministicScore as active learning algorithm.
    If no eq oracle is given, do this by:
     1) generate data (input_alphabet^extension_length)
     2) learn with passive GSM
     3) compare with previously learnt model (if any)
        - if the models are not bisimilar:
            4) generate random traces
            5) continue to step 2)
        - otherwise, terminate

    If an eq oracle is given, check for counterexample in step 3 instead.
    """
    if input_completeness_preprocessing is True:
        input_completeness_preprocessing = INPUT_COMPLETENESS_PROCESSING_DEFAULT
    assert input_completeness_preprocessing in ('random_suffix', 'all_suffixes') or not input_completeness_preprocessing
    if eq_oracle is not None:
        assert isinstance(eq_oracle, RobustEqOracleMixin)
        assert not use_dis_as_cex
    if use_dis_as_cex:
        assert eq_oracle is None

    logger.setLevel({0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG, 2.5: DEBUG_EXT, 3: DEBUG_EXT}[print_level])
    logger.debug("Running activeGSM with GlitchyDeterministicScore algorithm...")
    detailed_learning_info = defaultdict(dict)

    start_time = time.time()
    must_end_by = start_time + timeout if timeout is not None else 100 * 365 * 24 * 60 * 60  # no timeout -> must end in 100 years :)
    timed_out = False
    learning_rounds = 0

    all_input_combinations = list(itertools.product(alphabet, repeat=extension_length))

    result = None
    previous_hypotheses = []

    common_gsm_kwargs = dict(
        failure_rate=failure_rate,
        output_behavior=automaton_type,
        certainty=certainty,
        purge_mismatches=purge_mismatches,
        instrument=instrument
    )

    common_processing_kwargs = dict(
        sul=sul,
        alphabet=alphabet,
        all_input_combinations=all_input_combinations,
    )

    logger.debug("Creating initial traces...")
    traces = [trace_query(sul, input_combination) for input_combination in all_input_combinations]
    logger.debug_ext(f"Initial traces: {traces}")

    #####################################
    #             MAIN LOOP             #
    #####################################

    while True:
        logger.info(f"Starting learning round {(learning_rounds := learning_rounds + 1)}")

        detailed_learning_info[learning_rounds]["num_traces"] = len(traces)

        #####################################
        #              LEARNING             #
        #####################################

        hyp: SupportedAutomaton = gsm.run(traces, **common_gsm_kwargs)
        traces_used_to_learn = copy.deepcopy(traces)
        detailed_learning_info[learning_rounds]["traces_used_to_learn"] = traces_used_to_learn

        detailed_learning_info[learning_rounds]["hyp"] = str(hyp)

        #####################################
        #           PREPROCESSING           #
        #####################################

        if input_completeness_preprocessing and not timed_out:
            preprocessing_additional_traces = do_input_completeness_preprocessing(hyps={hyp.size: (hyp, None, None)},
                                                                                  suffix_mode=input_completeness_preprocessing,
                                                                                  **common_processing_kwargs)

            log_and_store_additional_traces(preprocessing_additional_traces, detailed_learning_info[learning_rounds],
                                            "input completeness", processing_step="pre")

            if preprocessing_additional_traces:
                traces += preprocessing_additional_traces
                continue
        else:
            # make input complete anyways, such that we don't have to check whether transitions exist during postprocessing
            if not hyp.is_input_complete():
                logger.debug(
                    f"Input completeness processing is deactivated, but {len(hyp.states)}-state hypothesis "
                    f"{id(hyp)} was not input complete. Force input completeness via self-loops.")
                hyp.make_input_complete()

        #####################################
        #            EVALUATING             #
        #####################################

        prev_hyp = previous_hypotheses[-1] if previous_hypotheses else None
        terminate = False

        if eq_oracle is not None:
            logger.debug("Checking for counterexample in eq oracle...")
            cex = eq_oracle.find_cex(hypothesis=hyp, return_outputs=False)
            terminate = cex is not None
            logger.debug(f"Eq oracle found{'' if cex else ' no'} counterexample - {'Terminate' if terminate else 'Continue'}!")
            if cex:
                logger.debug_ext(f"Counterexample: {cex}")

        elif use_dis_as_cex and prev_hyp:
            logger.debug("Using distinguishing sequence with previous hyp as cex. Finding distinguishing sequence...")
            cex = hyp.find_distinguishing_seq(hyp.initial_state, prev_hyp.initial_state, alphabet)
            terminate = cex is not None
            logger.debug(f"{'D' if cex else 'No d'}istinguishing sequence found - {'Terminate' if terminate else 'Continue'}!")
            if cex:
                logger.debug_ext(f"Counterexample: {cex}")

        elif prev_hyp:
            logger.debug("Checking for bisimilarity with previous hyp")
            cex = None
            terminate = bisimilar(prev_hyp, hyp)
            logger.debug(f"Previous and current hyp are{'' if terminate else ' not'} bisimilar - {'Terminate' if terminate else 'Continue'}!")

        else:
            assert learning_rounds == 1
            logger.debug("No previous hypothesis, no eq oracle; continue")
            cex = None
            terminate = False

        if terminate:
            result = hyp
            break

        #####################################
        #          POST-PROCESSING          #
        #####################################

        # we can only come here if we did not find a hypothesis we want to return
        logger.debug(f"Beginning postprocessing of hypotheses")

        if random_steps_per_round:
            postprocessing_additional_traces_random_walks = do_random_walks(random_steps_per_round,
                                                                            **common_processing_kwargs)

            log_and_store_additional_traces(postprocessing_additional_traces_random_walks,
                                            detailed_learning_info[learning_rounds], "random walks")

            traces = traces + postprocessing_additional_traces_random_walks

        if cex is not None:
            postprocessing_additional_traces_cex = do_cex_processing(cex, **common_processing_kwargs)

            log_and_store_additional_traces(postprocessing_additional_traces_cex,
                                            detailed_learning_info[learning_rounds], "cex")

            traces = traces + postprocessing_additional_traces_cex

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            logger.warning(f"No additional traces were produced during this round!")

        previous_hypotheses.append(hyp)

    #####################################
    #          RETURN RESULT            #
    #####################################

    if timed_out:
        logger.warning(
            f"Aborted learning after reaching timeout (start: {start_time:.2f}, now: {time.time():.2f}, timeout: {timeout})")

    hyp = result
    if return_data:
        total_time = round(time.time() - start_time, 2)

        active_info = build_info_dict(hyp, sul, learning_rounds, total_time, eq_oracle,
                                      detailed_learning_info, timed_out)
        if print_level > 1:
            print_learning_info(active_info)

        return hyp, active_info
    else:
        return hyp


def build_info_dict(hyp, sul, learning_rounds, total_time, eq_oracle,
                    detailed_learning_info, timed_out):
    active_info = {
        'learning_rounds': learning_rounds,
        'learned_automaton_size': hyp.size if hyp is not None else 0,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries if eq_oracle is not None else 0,
        'steps_eq_oracle': eq_oracle.num_steps if eq_oracle is not None else 0,
        'eq_oracle_time': eq_oracle.eq_query_time if eq_oracle is not None else 0,
        'total_time': total_time,
        'cache_saved': sul.num_cached_queries,
        'detailed_learning_info': detailed_learning_info,
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