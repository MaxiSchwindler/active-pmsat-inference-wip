import random

from typing import Any, Sequence, Literal

from aalpy import MooreMachine, KWayTransitionCoverageEqOracle, MooreState
from active_pmsatlearn.defs import *
from active_pmsatlearn.log import get_logger
from active_pmsatlearn.utils import trace_query, get_prefixes, get_input_from_stoc_trans, get_prefix_suffix_pairs
from evaluation.utils import TracedMooreSUL
from pmsatlearn.learnalgo import Input

logger = get_logger("APMSL")


def do_input_completeness_preprocessing(hyps: HypothesesWindow, suffix_mode: str, sul: SupportedSUL, alphabet: list[Input],
                                        all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Visit every state in the hypothesis and check if we have a transition for every input.
    If we don't have a transition there, query the SUL for additional traces of
    (state.prefix + input + (alphabet^extension_length)
    :param hyps: hypotheses window
    :param suffix_mode: which suffixes to append to (state.prefix + missing_input). One of:
                 'all_suffixes': all input combinations, i.e. alphabet^extension_length
                 'random_suffix': one random sequence of input combinations from all_input_combinations, i.e. random.choice(alphabet^extension_length)
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :return: a list of additional traces
    """
    new_traces = []
    already_performed_prefixes = set()

    for hyp, _, _ in hyps.values():
        hyp_str = f"{len(hyp.states)}-state hypothesis {id(hyp)}"
        if hyp.is_input_complete():
            logger.debug(f"{hyp_str} is already input complete.")
            continue

        logger.debug(f"Try to force input completeness in {hyp_str} to produce new traces...")

        hyp.compute_prefixes()
        for state in hyp.states:
            if state.prefix is None:
                continue  # ignore unreachable states (reachable only through glitches)

            for inp in alphabet:
                if inp not in state.transitions:
                    prefix = list(state.prefix) + [inp]
                    msg = (f"{hyp_str} didn't have a transition from state {state.state_id} "
                           f"(output='{state.output}', prefix={state.prefix}) with input '{inp}'")

                    if tuple(prefix) in already_performed_prefixes:
                        logger.debug(f"{msg}, but the prefix {prefix} was already performed this round - continue. ")
                        continue

                    logger.debug(f"{msg}. Perform queries with {prefix=} and {suffix_mode=}")

                    if suffix_mode == 'all_suffixes':
                        suffixes = all_input_combinations
                    elif suffix_mode == 'random_suffix':
                        suffixes = [random.choice(all_input_combinations)]
                    else:
                        assert False

                    for suffix in suffixes:
                        trace = trace_query(sul, prefix + list(suffix))
                        new_traces.append(trace)

                    already_performed_prefixes.add(tuple(prefix))

    return new_traces


def do_cex_processing(cex: Sequence[Input], sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                      suffix_mode: str) -> list[Trace]:
    """
    Counterexample processing, like in Active RPNI
    :param sul: system under learning
    :param cex: counterexample
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :param suffix_mode: which suffixes to append to each prefix. One of:
                 'all_suffixes': all input combinations, i.e. alphabet^extension_length
                 'random_suffix': one random suffix from all_input_combinations, i.e. random.choice(alphabet^extension_length)
    :return: list of new traces
    """
    logger.debug("Processing counterexample to produce new traces...")

    assert suffix_mode in ('all_suffixes', 'random_suffix')

    queries = []

    if suffix_mode == 'all_suffixes':
        for prefix in get_prefixes(cex):
            for suffix in all_input_combinations:
                queries.append(tuple(list(prefix) + list(suffix)))

    elif suffix_mode == 'random_suffix':
        assert len(alphabet) > 1

        for prefix, orig_suffix in get_prefix_suffix_pairs(cex):
            if len(orig_suffix) == 0:
                suffix = random.choice(all_input_combinations)
            else:
                suffix = random.choice([a for a in all_input_combinations if a[0] != orig_suffix[0]])

            queries.append(
                list(prefix) + list(suffix)
            )

    new_traces = []
    for query in queries:
        trace = trace_query(sul, query)
        new_traces.append(trace)

    return new_traces


def do_glitch_processing(hyps: HypothesesWindow, traces_used_to_learn_hyp: list[Trace], suffix_mode: str,
                         sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                         already_reproduced: set = None) -> list[Trace]:
    """
    Glitch processing. For every glitched transition from state s with input i, query traces for
    s.prefix + i + <specified suffixes>. Note that only one prefix is queried, the one of the state with
    the glitched transition.
    :param hyps: hypotheses window
    :param traces_used_to_learn_hyp: list of traces used to learn the given hypothesis. Currently only used for verification.
    :param suffix_mode: which suffixes to append to (state.prefix + glitched_input). One of:
                     'all_suffixes': all input combinations, i.e. alphabet^extension_length
                     'random_suffix': one random sequence of input combinations from all_input_combinations, i.e. random.choice(alphabet^extension_length)
    :param sul: system under learning
    :param alphabet: input alphabet
    :param all_input_combinations: all combinations of length <extension length> of the input alphabet
    :return: list of new traces
    """
    assert suffix_mode in ('all_suffixes', 'random_suffix')
    new_traces = []

    if already_reproduced is not None:
        already_performed_prefixes = already_reproduced
    else:
        already_performed_prefixes = set()

    for hyp, hyp_stoc, pmsat_info in hyps.values():
        hyp_str = f"{len(hyp.states)}-state hypothesis {id(hyp)}"
        logger.debug(f"Use glitched transitions of {hyp_str} produce new traces...")
        all_glitched_steps = [traces_used_to_learn_hyp[traces_index][trace_index]
                              for (traces_index, trace_index) in pmsat_info["glitch_steps"]]

        all_glitched_transitions = pmsat_info["glitch_trans"]  # [(s1, i, s2), ...]

        hyp.compute_prefixes()
        for s1_id, glitched_input, s2_id in all_glitched_transitions:
            s1 = hyp.get_state_by_id(s1_id)
            s2 = hyp.get_state_by_id(s2_id)
            assert (glitched_input, s2.output) in all_glitched_steps, (
                f"The tuple {(glitched_input, s2.output)} "
                f"was not found in {all_glitched_steps=}")

            msg = (f"{hyp_str} contained a glitched transition from state {s1.state_id} (output='{s1.output}', "
                   f"prefix={s1.prefix}) with input '{glitched_input}' to state {s2.state_id}")

            if s1.prefix is None:
                logger.debug(f"{msg}, but since state {s1.state_id} is unreachable (prefix is None), continue.")
                continue

            prefix = list(s1.prefix) + [glitched_input]

            if tuple(prefix) in already_performed_prefixes:
                logger.debug(f"{msg}, but the prefix {prefix} was already performed this round - continue. ")
                continue

            logger.debug(f"{msg}. Perform queries with {prefix=} and {suffix_mode=}")

            if suffix_mode == 'all_suffixes':
                suffixes = all_input_combinations
            elif suffix_mode == 'random_suffix':
                suffixes = [random.choice(all_input_combinations)]
            else:
                assert False

            for suffix in suffixes:
                trace = trace_query(sul, prefix + list(suffix))
                new_traces.append(trace)

            already_performed_prefixes.add(tuple(prefix))

    return new_traces


def get_incongruent_traces(glitchless_trace, traces) -> set[int]:
    """ Get incongruent traces: traces which exist in @traces, but are incompatible with
    traces in @glitchless_traces. E.g. :
        glitchless_traces = [(o1, (i1, o2))]
        traces = [(o1, (i1, o3)]
    @returns a set of indices to incongruent traces in @traces.
    """
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
            new_traces.extend(
                do_cex_processing(cex=cex, sul=sul, alphabet=alphabet, all_input_combinations=all_input_combinations, suffix_mode="random_suffix")
            )
            for p in get_prefixes(cex):
                performed_prefixes.add(tuple(p))

    return new_traces


def do_replay_glitches(hyps: HypothesesWindow, traces_used_to_learn: list[Trace], suffix_modes: Sequence[str],
                       sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                       already_replayed: set = None) -> list[Trace]:
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
    if already_replayed is not None:
        replayed_trace_indices = already_replayed
        replayed_trace_step_pairs = already_replayed
    else:
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


def do_transition_coverage(hyp: SupportedAutomaton, num_steps: int, sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Try to ensure transition coverage in the given hypothesis.
    Create a TracedMooreSUL with a copy of hyp as 'ground truth', create a KWayTransitionCoverageEqOracle wrapping this face SUL,
    and then ask this oracle for a counterexample with (another copy of) the hypothesis.
    This will of course not lead to a useful counterexample (in fact, since the SUL and the hyp are identical, no counterexample),
    but we can trace which queries are performed by the oracle; since the oracle tries to ensure transition coverage,
    we can then replay the queries it chose on the *actual* SUL to try to achieve actual transition coverage.

    Args:
        hyp: Hypothesis to ensure transition coverage on
        num_steps: Maximum number of steps to perform
        sul: system under learning
        alphabet: input alphabet
        all_input_combinations: all input combinations, alphabet^extension_length

    Returns:
        a list of new traces

    """
    logger.debug(f"Performing {num_steps} for transition coverage")
    hyp_sul = TracedMooreSUL(mm=MooreMachine(initial_state=hyp.initial_state, states=[s for s in hyp.states if s.prefix is not None]))
    oracle = KWayTransitionCoverageEqOracle(alphabet=alphabet, sul=hyp_sul, max_number_of_steps=num_steps)
    oracle.find_cex(MooreMachine(initial_state=hyp.initial_state, states=[s for s in hyp.states if s.prefix is not None]))
    assert hyp_sul.num_steps <= num_steps

    new_traces = []
    for trace in hyp_sul.traces:
        inputs = [i for (i, o) in trace[1:]]
        new_traces.append(trace_query(sul, inputs))

    return new_traces


def do_random_walks(num_steps: int, sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]]) -> list[Trace]:
    """
    Do random walks on the SUL

    Args:
        num_steps: exact number of steps all walks should add up to
        sul: system under learning
        alphabet: input alphabet
        all_input_combinations: all input combinations, alphabet^extension_length

    Returns:
        a list of new traces
    """
    logger.debug(f"Performing random walks with a total of {num_steps} steps")
    new_traces = []
    remaining_steps = num_steps
    while remaining_steps > 0:
        walk_len = random.randint(1, remaining_steps)
        random_walk = tuple(random.choice(alphabet) for _ in range(walk_len))

        new_traces.append(trace_query(sul, random_walk))
        remaining_steps -= walk_len

    return new_traces


def do_random_walks_with_reset_prob(num_steps: int, sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                                    reset_prob: float = 0.09) -> list[Trace]:
    """
    Do random walks on the SUL

    Args:
        num_steps: exact number of steps all walks should add up to
        sul: system under learning
        alphabet: input alphabet
        all_input_combinations: all input combinations, alphabet^extension_length

    Returns:
        a list of new traces
    """
    logger.debug(f"Performing random walks with a total of {num_steps} steps")

    queries = []
    remaining_steps = num_steps
    while remaining_steps > 0:
        query = []
        while remaining_steps > 0:
            inp = random.choice(alphabet)
            remaining_steps -= 1
            query.append(inp)
            if random.random() < reset_prob:
                break
        queries.append(query)

    new_traces = []
    for query in queries:
        new_traces.append(trace_query(sul, query))
    return new_traces


def do_state_prefix_coverage_on_hyp(num_steps: int, hypothesis: SupportedAutomaton, sul: SupportedSUL, alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                             reset_prob: float = 0.09, mode: Literal["random", "depth_first", "balanced_traces", "balanced_steps"] = "balanced_traces",
                             states_to_steps: dict[tuple[Input, ...], int] = None, states_to_traces: dict[tuple[Input, ...], int] = None) -> list[Trace]:
    hypothesis.compute_prefixes()
    states_to_cover = [s for s in hypothesis.states if s.prefix is not None]  # filter out unreachable states
    return do_state_prefix_coverage(num_steps=num_steps, states_to_cover=states_to_cover, sul=sul,
                                    alphabet=alphabet, all_input_combinations=all_input_combinations,
                                    reset_prob=reset_prob, mode=mode,
                                    states_to_traces=states_to_traces, states_to_steps=states_to_steps)

def do_state_prefix_coverage(num_steps: int, states_to_cover: list[MooreState], sul: SupportedSUL,
                             alphabet: list[Input], all_input_combinations: list[tuple[Input, ...]],
                             reset_prob: float = 0.09, mode: Literal[
            "random", "depth_first", "balanced_traces", "balanced_steps"] = "balanced_traces",
                             states_to_steps: dict[tuple[Input, ...], int] = None,
                             states_to_traces: dict[tuple[Input, ...], int] = None) -> list[Trace]:
    assert mode in ("random", "depth_first", "balanced_traces", "balanced_steps")
    if mode == "balanced_traces":
        assert states_to_traces is not None
    if mode == "balanced_steps":
        assert states_to_steps is not None

    states_to_cover = [s for s in states_to_cover if s.prefix is not None]  # filter out unreachable states
    assert len(states_to_cover) > 0, f"No prefixes!"

    if mode == "depth_first":
        # reverse sort the states by length of their access sequences
        # first do the random walk on the state with the longest access sequence
        states_to_cover.sort(key=lambda s: len(s.prefix), reverse=True)
    elif mode == "balanced_traces":
        if len(states_to_steps) == 0:
            random.shuffle(states_to_cover)
        else:
            states_to_cover.sort(key=lambda s: states_to_traces.get(s.prefix, 0), reverse=False)
    elif mode == "balanced_steps":
        if len(states_to_steps) == 0:
            random.shuffle(states_to_cover)
        else:
            states_to_cover.sort(key=lambda s: states_to_steps.get(s.prefix, 0), reverse=False)
    else:
        random.shuffle(states_to_cover)

    queries = []
    remaining_steps = num_steps

    while len(states_to_cover) > 0 and remaining_steps > 0:
        state = states_to_cover.pop(0)
        if len(state.prefix) > remaining_steps:
            # this state cannot be covered anymore in this round - remove
            continue

        query = [*state.prefix]
        remaining_steps -= len(state.prefix)
        while remaining_steps > 0:
            inp = random.choice(alphabet)
            remaining_steps -= 1
            query.append(inp)
            if random.random() < reset_prob:
                break
        queries.append(query)
        states_to_steps[state.prefix] += (len(query) - len(state.prefix))
        states_to_traces[state.prefix] += 1
        states_to_cover.append(state)

    # print(states_to_steps)
    # print(states_to_traces)
    new_traces = []
    for query in queries:
        new_traces.append(trace_query(sul, query))
    return new_traces
