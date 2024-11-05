import itertools
import time
from collections import defaultdict
from pprint import pprint

from aalpy.SULs import MooreSUL, MealySUL
from aalpy.automata import MooreMachine, MealyMachine
from aalpy.base import Oracle

from pmsatlearn import run_pmSATLearn

from active_pmsatlearn.utils import *


class PmSATLearn:
    """ Small wrapper class around the pmSATLearn algorithm to keep track of the number of calls"""
    def __init__(self, automata_type, pm_strategy, timeout, cost_scheme, print_info):
        self.automata_type = automata_type
        self.pm_strategy = pm_strategy
        self.timeout = timeout
        self.cost_scheme = cost_scheme
        self.print_info = print_info
        self.num_calls_to_pmsat_learn = 0

    def run(self, traces: Sequence[Trace], num_states: int):
        self.num_calls_to_pmsat_learn += 1
        hyp, hyp_stoc, info = run_pmSATLearn(data=traces,
                                             n_states=num_states,
                                             automata_type=self.automata_type,
                                             pm_strategy=self.pm_strategy,
                                             timeout=self.timeout,
                                             cost_scheme=self.cost_scheme,
                                             print_info=self.print_info)
        return hyp, hyp_stoc, info


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
    additional_states_threshold: int = 10,
    return_data: bool = True,
    print_level: int | None = 2,
) -> (
        SupportedAutomaton | tuple[SupportedAutomaton, dict[str, Any]]
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

    log(f"Creating initial traces...", level=2)
    traces = []
    for input_combination in all_input_combinations:
        traces.append(trace_query(sul, input_combination))
    log(f"Initial traces: {traces}", level=3)

    num_states_before_receiving_cex = {}
    num_states = get_num_outputs(traces)  # start with minimum number of states in first round

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0
    detailed_learning_info = defaultdict(dict)

    pmsat_learn = PmSATLearn(automata_type=automaton_type,
                             pm_strategy=pm_strategy,
                             timeout=timeout,
                             cost_scheme=cost_scheme,
                             print_info=print_level > 2)

    while True:
        learning_rounds += 1
        log(f"Learning with passive pmSATLearn (round {learning_rounds}). "
            f"Learning a {num_states}-state automaton from {len(traces)} traces.", level=2)
        hyp, hyp_stoc, pmsat_info = pmsat_learn.run(traces, num_states)

        #####################################
        #           PRE-PROCESSING          #
        #####################################

        if hyp is not None and input_completeness_processing:
            # visit every state in the hypothesis and check if we have a transition for every input
            # if we don't have a transition there, add additional traces of state.prefix + input + (inp_alph ^ ext_len)
            log("Try to force input completeness in hypothesis to produce new traces...", level=2)
            if hyp.is_input_complete():
                log("Hypothesis is already input complete. In the current implementation, this step won't be useful.", level=2)

            hyp.compute_prefixes()
            preprocessing_additional_traces = []
            for state in hyp.states:
                if state.prefix is None:
                    continue  # ignore unreachable states (reachable only through glitches)

                for inp in alphabet:
                    if inp not in state.transitions:
                        log(f"Hypothesis didn't have a transition from state {state.state_id} "
                            f"(output='{state.output}', prefix={state.prefix}) with input '{inp}' - create traces!", level=2)

                        for suffix in all_input_combinations:
                            trace = trace_query(sul, list(state.prefix) + [inp] + list(suffix))
                            if trace not in traces:
                                # TODO for now, we only add _new_ traces, to avoid huge number of traces
                                #      however, having the same trace multiple times (or weighting it)
                                #      would 'tell' the solver that it is 'more important' (i.e. it is less likely
                                #      to be a glitch) in Partial MaxSAT
                                preprocessing_additional_traces.append(trace)

            if preprocessing_additional_traces:
                traces = traces + preprocessing_additional_traces
                num_states = max(num_states, get_num_outputs(traces))

                log(f"Produced {len(preprocessing_additional_traces)} additional traces from counterexample", level=2)
                detailed_learning_info[learning_rounds]["preprocessing_additional_traces"] = len(preprocessing_additional_traces)

                log(f"Learning again after preprocessing with passive pmSATLearn (round {learning_rounds})...", level=2)
                # TODO should this count as learning_round? we track the number of solver calls seperately
                learning_rounds -= 1  # decrease before it is increased again at start of loop - this does not count as full round
                continue  # jump back up and learn again

        #####################################
        #              EQ-CHECK             #
        #####################################

        if hyp is not None and pmsat_info["is_sat"]:
            log(f"pmSATLearn learned hypothesis with {len(hyp.states)} states", level=1)

            detailed_learning_info[learning_rounds]["num_traces"] = len(traces)
            detailed_learning_info[learning_rounds]["num_states"] = len(hyp.states)  # TODO: see TODO in num_states calculation at bottom
            detailed_learning_info[learning_rounds]["num_glitches"] = len(pmsat_info["glitch_steps"])
            detailed_learning_info[learning_rounds]["is_sat"] = True

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
            detailed_learning_info[learning_rounds]["is_sat"] = False

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

        if cex_processing:
            # use counterexample, same as ActiveRPNI
            log("Processing counterexample to produce new traces...", level=2)
            postprocessing_additional_traces_cex = []
            for prefix in get_prefixes(cex):
                for suffix in all_input_combinations:
                    trace = trace_query(sul, list(prefix) + list(suffix))
                    if trace not in traces:
                        # TODO for now, we only add _new_ traces, to avoid huge number of traces
                        #      however, having the same trace multiple times (or weighting it)
                        #      would 'tell' the solver that it is 'more important' (i.e. it is less likely
                        #      to be a glitch) in Partial MaxSAT
                     postprocessing_additional_traces_cex.append(trace)

            log(f"Produced {len(postprocessing_additional_traces_cex)} additional traces from counterexample", level=2)
            log(f"Additional traces from counterexample: {postprocessing_additional_traces_cex}", level=3)
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_cex"] = len(postprocessing_additional_traces_cex)

            traces = traces + postprocessing_additional_traces_cex

        if glitch_processing:
            # look at glitches:
            # for every glitched transition from state s with input i, add traces for s.prefix + i + (inp_alph ^ ext_len)
            log("Use glitched transitions to produce new traces...", level=2)
            hyp_stoc.compute_prefixes()
            postprocessing_additional_traces_glitch = []
            all_glitched_steps = [traces[traces_index][trace_index] for (traces_index, trace_index) in pmsat_info["glitch_steps"]]
            for state in hyp_stoc.states:
                for inp, next_state in state.transitions.items():
                    if inp.startswith("!"):  # glitched transition
                        state_prefix = [get_input_from_stoc_trans(i) for i in state.prefix]
                        glitched_input = get_input_from_stoc_trans(inp)
                        assert (glitched_input, next_state.output) in all_glitched_steps, (f"The tuple {(glitched_input, next_state.output)}"
                                                                                           f"was not found in {all_glitched_steps=}")
                        log(f"Hypothesis contained a glitched transition from state {state.state_id} (output='{state.output}', "
                            f"prefix={state_prefix}) with input '{glitched_input}' to state {next_state.state_id} - create traces!", level=2)

                        for suffix in all_input_combinations:
                            trace = trace_query(sul, state_prefix + [glitched_input] + list(suffix))
                            if trace not in traces:
                                # only add traces we don't already have - TODO does this make a difference for the solver? (*max* sat - if multiple times in it, more important. Maybe mark as hard clause? maybe weight somehow?)
                                postprocessing_additional_traces_glitch.append(trace)

            log(f"Produced {len(postprocessing_additional_traces_glitch)} additional traces from glitch processing", level=2)
            log(f"Additional traces from glitch processing: {postprocessing_additional_traces_glitch}", level=3)
            detailed_learning_info[learning_rounds]["postprocessing_additional_traces_glitch"] = len(postprocessing_additional_traces_glitch)

            traces = traces + postprocessing_additional_traces_glitch

        #####################################
        #          CALC NUM_STATES          #
        #####################################

        # set number of states for next iteration
        cex_as_tuple = tuple(cex)
        num_outputs = get_num_outputs(traces)
        if cex_as_tuple in num_states_before_receiving_cex:
            # we already saw and processed this cex once before
            # increase the number of states from last time by one (if number of outputs isn't anyways higher)
            # TODO: the number of states could already be higher now, this way decreases the num_states again
            #  (i.e. we learned with 3 states, received cex, learned some more rounds, now we learned with 10 states,
            #   then we received the same cex again - now we start over at 4 states). Does that make sense?
            #   Does this whole thing make sense for non-consecutive counter examples?
            num_states_last_time = num_states_before_receiving_cex[cex_as_tuple]
            num_states = max(num_states_last_time + 1, num_outputs)
            log(f"This counterexample ({cex}) was already seen before with {num_states_last_time} states, "
                f"increase number of states by at least 1 to {num_states}", level=2)
        else:
            # this is the first time we handled this cex
            # try again with the same number of states (if num_outputs hasn't increased)
            num_states = max(num_states, num_outputs)
            log(f"This is the first time we received counterexample {cex}. After processing, "
                f"run pmSATLearn again with the same number of states ({num_states}) if possible.", level=2)

        if num_states > num_outputs + additional_states_threshold:
            raise ValueError(f"Number of states is much larger than number of outputs")

        num_states_before_receiving_cex[cex_as_tuple] = num_states


def build_info_dict(hyp, sul, eq_oracle, pmsat_learn, learning_rounds, total_time, learning_time, eq_query_time,
                    last_pmsat_info, detailed_learning_info, hyp_stoc):
    active_info = {
        'learning_rounds': learning_rounds,
        'automaton_size': hyp.size if hyp is not None else None,
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
        'hyp_stoc': str(hyp_stoc),
    }
    return active_info
