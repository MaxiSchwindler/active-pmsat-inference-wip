from typing import Any

from active_pmsatlearn.defs import Input, Trace, SupportedAutomaton, SupportedSUL
from active_pmsatlearn.log import get_logger
from active_pmsatlearn.utils import trace_query, get_prefixes, get_input_from_stoc_trans

logger = get_logger("APMSL")


def do_input_completeness_processing(hyp: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input],
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

def do_cex_processing(sul: SupportedSUL, cex: Trace, all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
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


def do_glitch_processing(sul: SupportedSUL, hyp: SupportedAutomaton, pmsat_info: dict[str, Any], hyp_stoc: SupportedAutomaton,
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


