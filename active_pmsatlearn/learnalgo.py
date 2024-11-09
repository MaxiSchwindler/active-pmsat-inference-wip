import itertools
import time
import math
from collections import defaultdict
from pprint import pprint

from aalpy.SULs import MooreSUL, MealySUL
from aalpy.automata import MooreMachine, MealyMachine
from aalpy.base import Oracle

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.utils import *


class PmSATLearn:
    """ Small wrapper class around the pmSATLearn algorithm to keep track of the number of calls"""
    def __init__(self, automata_type: str, pm_strategy: str, timeout: float | None, cost_scheme: str, print_info: bool,
                 logging_func: callable = print, max_calls_per_round: int = 1):
        self.automata_type = automata_type
        self.pm_strategy = pm_strategy
        self.timeout = timeout
        self.cost_scheme = cost_scheme
        self.print_info = print_info
        self.num_calls_to_pmsat_learn = 0
        self.log = logging_func
        self.max_calls_per_round = max_calls_per_round

    def run_once(self, traces: Sequence[Trace], num_states: int):
        self.num_calls_to_pmsat_learn += 1
        self.log(f"Running pmSATLearn to learn a {num_states}-state automaton from {len(traces)} traces "
                 f"({self.num_calls_to_pmsat_learn}. call)", level=2)
        hyp, hyp_stoc, info = run_pmSATLearn(data=traces,
                                             n_states=num_states,
                                             automata_type=self.automata_type,
                                             pm_strategy=self.pm_strategy,
                                             timeout=self.timeout,
                                             cost_scheme=self.cost_scheme,
                                             print_info=self.print_info)
        if hyp:
            info['num_glitches'] = len(info['glitch_steps'])
            self.log(f"pmSATLearn learned a hypothesis with {info['num_glitches']} glitches.", level=2)
        else:
            self.log(f"pmSATLearn could not learn a hypothesis.", level=2)
            info['num_glitches'] = math.inf

        return hyp, hyp_stoc, info

    def run_multiple(self,
                     traces: Sequence[Trace],
                     min_states: int | None = None, max_states: int = 10, initial_num_states: int | None = None,
                     target_glitch_percentage: float = 0.0,
                     ):
        min_states = get_num_outputs(traces) if min_states is None else min_states
        initial_num_states = max(min_states, initial_num_states) if initial_num_states is not None else min_states

        assert min_states <= max_states, f"Minimum number of states {min_states} must be smaller than maximum {max_states}"

        def num_states_generator():
            if initial_num_states != min_states:
                yield initial_num_states
            yield from range(min_states, max_states+1)

        best_solution = None
        least_glitches = math.inf
        complete_num_steps = sum(len(trace) for trace in traces)

        self.log(f"Running pmSATLearn to try and find a solution with less than {target_glitch_percentage}% glitches", level=1)
        for num_states in num_states_generator():
            solution = self.run_once(traces, num_states)
            num_glitches = len(solution[2]["glitch_steps"])
            percentage = num_glitches / complete_num_steps * 100

            if percentage <= target_glitch_percentage:
                self.log(f"Found {num_states}-state solution with {percentage}% glitches!", level=1)
                return solution

            if num_glitches < least_glitches:
                least_glitches = num_glitches
                best_solution = solution

        self.log(f"After {num_states-initial_num_states} calls with num_states ranging from {initial_num_states} to {num_states}, "
                 f"no solution with less than {least_glitches / complete_num_steps * 100}% glitches was found. "
                 f"Returning solution with least glitches.", level=1)
        return best_solution


def run_activePmSATLearn(
    alphabet: list,
    sul: SupportedSUL,
    eq_oracle: Oracle,
    automaton_type: Literal["mealy", "moore"],
    extension_length: int = 2,
    pm_strategy: Literal["lsu", "fm", "rc2"] = "rc2",
    timeout: int | float | None = None,
    cost_scheme: Literal["per_step", "per_transition"] = "per_step",
    input_completeness_processing: bool = True,
    cex_processing: bool = True,
    glitch_processing: bool = True,
    allowed_glitch_percentage: float = 0.0,
    min_states: int | None = None,
    max_states: int = 10,
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
    :param timeout: timeout for a single solver call (in seconds)
    :param cost_scheme: cost scheme for pmsat-learn
    :param input_completeness_processing: whether input completeness processing should be performed
                                          before querying the oracle for a counterexample
    :param cex_processing: whether counterexample processing should be performed
    :param glitch_processing: whether glitch processing should be performed
    :param additional_states_threshold: how many additional states more than the number of outputs
                                        in the traces are allowed.
    :param return_data: whether to return a dictionary containing learning information
    :param print_level: 0 - None,
                        1 - just results,
                        2 - current round information,
                        3 - pmsat-learn output and logging of traces
    """
    # Input verification happens in run_pmSATLearn

    def log(message: Any, level: int):
        if print_level >= level:
            if isinstance(message, str):
                print(message)
            else:
                pprint(message)

    all_input_combinations = list(itertools.product(alphabet, repeat=extension_length))

    def do_input_completeness_processing(current_hyp):
        return _do_input_completeness_processing(hyp=current_hyp, sul=sul, alphabet=alphabet,
                                                 all_input_combinations=all_input_combinations, log=log)

    def do_cex_processing(current_cex):
        return _do_cex_processing(sul=sul, cex=current_cex, all_input_combinations=all_input_combinations, log=log)

    def do_glitch_processing(current_pmsat_info: dict[str, Any], current_hyp_stoc: SupportedAutomaton, current_traces: list[Trace]):
        return _do_glitch_processing(sul=sul, pmsat_info=current_pmsat_info, hyp_stoc=current_hyp_stoc,
                                     traces_used_to_learn_hyp=current_traces, all_input_combinations=all_input_combinations, log=log)

    log(f"Creating initial traces...", level=2)
    traces = []
    for input_combination in all_input_combinations:
        traces.append(trace_query(sul, input_combination))
    log(f"Initial traces: {traces}", level=3)

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0
    detailed_learning_info = defaultdict(dict)

    hyp = None
    pmsat_learn = PmSATLearn(automata_type=automaton_type,
                             pm_strategy=pm_strategy,
                             timeout=timeout,
                             cost_scheme=cost_scheme,
                             print_info=print_level > 2,
                             logging_func=log)

    while True:
        learning_rounds += 1
        detailed_learning_info[learning_rounds]["num_traces"] = len(traces)
        hyp, hyp_stoc, pmsat_info = pmsat_learn.run_multiple(traces=traces,
                                                             min_states=min_states,
                                                             max_states=max_states,
                                                             initial_num_states=len(hyp.states) if hyp is not None else None,
                                                             target_glitch_percentage=allowed_glitch_percentage)

        #####################################
        #           PRE-PROCESSING          #
        #####################################

        if hyp is not None and input_completeness_processing:
            preprocessing_additional_traces = do_input_completeness_processing(hyp)

            # TODO for now, we only add _new_ traces, to avoid huge number of traces
            #      however, having the same trace multiple times (or weighting it)
            #      would 'tell' the solver that it is 'more important' (i.e. it is less likely
            #      to be a glitch) in Partial MaxSAT
            remove_duplicate_traces(traces, preprocessing_additional_traces)

            log(f"Produced {len(preprocessing_additional_traces)} additional traces from counterexample", level=2)
            detailed_learning_info[learning_rounds]["preprocessing_additional_traces"] = len(preprocessing_additional_traces)

            if preprocessing_additional_traces:
                log(f"Learning again after preprocessing with passive pmSATLearn (round {learning_rounds})...", level=2)
                traces = traces + preprocessing_additional_traces
                # TODO should this count as learning_round? we track the number of solver calls seperately
                #  however, this would also overwrite this rounds detailed_learning_info (i.e. "preprocessing_additional_traces") - would needa a special case
                continue  # jump back up and learn again

        #####################################
        #              EQ-CHECK             #
        #####################################

        detailed_learning_info[learning_rounds]["num_states"] = len(hyp.states) if hyp is not None else None
        detailed_learning_info[learning_rounds]["num_glitches"] = len(pmsat_info["glitch_steps"])
        detailed_learning_info[learning_rounds]["is_sat"] = pmsat_info["is_sat"]
        detailed_learning_info[learning_rounds]["timed_out"] = pmsat_info["timed_out"]

        if hyp is not None and pmsat_info["is_sat"]:
            log(f"pmSATLearn learned hypothesis with {len(hyp.states)} states", level=1)

            if not hyp.is_input_complete():
                hyp.make_input_complete()  # oracles (randomwalk, perfect) assume input completeness

            eq_query_start = time.time()
            cex = eq_oracle.find_cex(hyp)
            eq_query_time += time.time() - eq_query_start
        else:
            # UNSAT - set cex to None to enter next if
            cex = None
            hyp = None
            hyp_stoc = None

        if cex is None:
            if pmsat_info["is_sat"]:
                log("No counterexample found. Returning hypothesis", level=1)
            else:
                log("UNSAT! Returning None", level=1)

            if return_data:
                total_time = round(time.time() - start_time, 2)
                eq_query_time = round(eq_query_time, 2)
                learning_time = round(total_time - eq_query_time, 2)

                active_info = build_info_dict(hyp, sul, eq_oracle, pmsat_learn, learning_rounds, total_time, learning_time,
                                              eq_query_time, pmsat_info, detailed_learning_info, hyp_stoc)
                log(active_info, level=2)

                return hyp, active_info
            else:
                return hyp

        #####################################
        #          POST-PROCESSING          #
        #####################################

        log("Counterexample found - process and continue learning", level=1)

        if glitch_processing:
            # we have to do glitch processing first, before appending anything to traces!
            postprocessing_additional_traces_glitch = do_glitch_processing(pmsat_info, hyp_stoc, traces)

            # TODO for now, we only add _new_ traces, to avoid huge number of traces
            #      however, having the same trace multiple times (or weighting it)
            #      would 'tell' the solver that it is 'more important' (i.e. it is less likely
            #      to be a glitch) in Partial MaxSAT
            remove_duplicate_traces(traces, postprocessing_additional_traces_glitch)

            log(f"Produced {len(postprocessing_additional_traces_glitch)} additional traces from glitch processing", level=2)
            log(f"Additional traces from glitch processing: {postprocessing_additional_traces_glitch}", level=3)
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_glitch"] = len(postprocessing_additional_traces_glitch)

            traces = traces + postprocessing_additional_traces_glitch

        if cex_processing:
            postprocessing_additional_traces_cex = do_cex_processing(cex)

            # TODO for now, we only add _new_ traces, to avoid huge number of traces
            #      however, having the same trace multiple times (or weighting it)
            #      would 'tell' the solver that it is 'more important' (i.e. it is less likely
            #      to be a glitch) in Partial MaxSAT
            remove_duplicate_traces(traces, postprocessing_additional_traces_cex)

            log(f"Produced {len(postprocessing_additional_traces_cex)} additional traces from counterexample", level=2)
            log(f"Additional traces from counterexample: {postprocessing_additional_traces_cex}", level=3)
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_cex"] = len(postprocessing_additional_traces_cex)

            traces = traces + postprocessing_additional_traces_cex

        if len(traces) == detailed_learning_info[learning_rounds]["num_traces"]:
            log(f"No additional traces were produced during this round", level=2)


def _do_input_completeness_processing(hyp: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input],
                                      all_input_combinations: list[tuple[Input, ...]], log=log_all) -> list[Trace]:
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
    log("Try to force input completeness in hypothesis to produce new traces...", level=2)
    if hyp.is_input_complete():
        log("Hypothesis is already input complete. In the current implementation, this step won't be useful.", level=2)

    hyp.compute_prefixes()
    new_traces = []
    for state in hyp.states:
        if state.prefix is None:
            continue  # ignore unreachable states (reachable only through glitches)

        for inp in alphabet:
            if inp not in state.transitions:
                log(f"Hypothesis didn't have a transition from state {state.state_id} "
                    f"(output='{state.output}', prefix={state.prefix}) with input '{inp}' - create traces!", level=2)

                for suffix in all_input_combinations:
                    trace = trace_query(sul, list(state.prefix) + [inp] + list(suffix))
                    new_traces.append(trace)

    return new_traces


def _do_cex_processing(sul: SupportedSUL, cex: Trace, all_input_combinations: list[tuple[Input, ...]], log=log_all) -> list[Trace]:
    """
    Counterexample processing, like in Active RPNI
    :param sul: system under learning
    :param cex: counterexample
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :param log: a function to log info to
    :return: list of new traces
    """
    log("Processing counterexample to produce new traces...", level=2)
    new_traces = []
    for prefix in get_prefixes(cex):
        for suffix in all_input_combinations:
            trace = trace_query(sul, list(prefix) + list(suffix))
            new_traces.append(trace)

    return new_traces


def _do_glitch_processing(sul: SupportedSUL, pmsat_info: dict[str, Any], hyp_stoc: SupportedAutomaton,
                         traces_used_to_learn_hyp: list[Trace], all_input_combinations: list[tuple[Input, ...]],
                         log=log_all) -> list[Trace]:
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
    log("Use glitched transitions to produce new traces...", level=2)
    hyp_stoc.compute_prefixes()
    new_traces = []
    all_glitched_steps = [traces_used_to_learn_hyp[traces_index][trace_index] for (traces_index, trace_index) in
                          pmsat_info["glitch_steps"]]

    for state in hyp_stoc.states:
        for inp, next_state in state.transitions.items():
            if inp.startswith("!"):  # glitched transition
                state_prefix = [get_input_from_stoc_trans(i) for i in state.prefix]
                glitched_input = get_input_from_stoc_trans(inp)
                assert (glitched_input, next_state.output) in all_glitched_steps, (f"The tuple {(glitched_input, next_state.output)}" 
                                                                                   f"was not found in {all_glitched_steps=}")
                log(f"Hypothesis contained a glitched transition from state {state.state_id} (output='{state.output}', "
                    f"prefix={state_prefix}) with input '{glitched_input}' to state {next_state.state_id} - create traces!",
                    level=2)

                for suffix in all_input_combinations:
                    trace = trace_query(sul, state_prefix + [glitched_input] + list(suffix))
                    new_traces.append(trace)

    return new_traces


def build_info_dict(hyp, sul, eq_oracle, pmsat_learn, learning_rounds, total_time, learning_time, eq_query_time,
                    last_pmsat_info, detailed_learning_info, hyp_stoc, abort_reason=None):
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
        'solver_calls': pmsat_learn.num_calls_to_pmsat_learn,
        'detailed_learning_info': detailed_learning_info,
        'hyp_stoc': str(hyp_stoc) if hyp is not None else None,
        'abort_reason': abort_reason,
    }
    return active_info
