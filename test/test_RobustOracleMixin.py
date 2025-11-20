import itertools
import random
import sys
import os

import numpy as np
import pytest
from aalpy import MooreMachine
from aalpy.SULs import MooreSUL
from active_pmsatlearn import RobustEqOracleMixin
from active_pmsatlearn.oracles import RobustRandomWalkEqOracle

from active_pmsatlearn.utils import trace_query
from evaluation.generate_automata import generate_automaton
from evaluation.learn_automata import setup_sul

from evaluation.utils import GlitchingSUL, TracedMooreSUL
from utils import MOORE_CONFIGS, mm, compare_states


@pytest.mark.parametrize("mm", MOORE_CONFIGS, indirect=True, ids=lambda c: f"{c['num_states']}s{c['num_inputs']}i{c['num_outputs']}o")
@pytest.mark.parametrize("percent_glitches", [0.0, 0.5, 1.0, 5.0, 100.0], ids=lambda g: f"{g}%".replace('.', ','))
@pytest.mark.parametrize("traced", [True, False], ids=lambda t: "traced" if t else "non-traced")
def test_validate_cex(mm: MooreMachine, percent_glitches: float, traced: bool, fault_mode: str = "enter_random_state"):
    sul_mm = MooreMachine(initial_state=mm.initial_state, states=mm.states)
    underlying_sul = TracedMooreSUL(sul_mm) if traced else MooreSUL(sul_mm)
    sul = GlitchingSUL(sul=underlying_sul, glitch_percentage=percent_glitches, fault_type="enter_random_state")

    assert hasattr(sul, "traces") == traced
    assert isinstance(sul, GlitchingSUL)
    assert sul.fault_type == fault_mode
    assert sul.glitch_percentage == percent_glitches

    input_alphabet = mm.get_input_alphabet()
    extension_length = 10  # len(mm.states) + 1
    all_input_combinations = list(itertools.product(input_alphabet, repeat=extension_length))
    total_num_steps = len(all_input_combinations) * extension_length

    robust_oracle = RobustRandomWalkEqOracle(
        alphabet=input_alphabet,
        sul=sul,
        num_steps=50,
        reset_prob=0.09,
        reset_after_cex=True,

        perform_n_times=20,
        validity_threshold=0.51,
        max_num_tries=5,
    )

    hyp_mm = mm.copy()
    hyp_mm.
    robust_oracle.validate_counterexample()

    # quick test to show that the states are the same objects, but steps in the sul don't influence the mm:
    # sul.automaton.reset_to_initial()
    # mm.reset_to_initial()
    # assert sul.automaton.current_state == sul.automaton.initial_state == mm.current_state == mm.initial_state
    # while sul.automaton.current_state == sul.automaton.initial_state:
    #     sul.step(random.choice(input_alphabet))
    # assert sul.automaton.current_state != mm.current_state

    num_glitches = 0
    for input_combination in all_input_combinations:
        mm.reset_to_initial()
        sul.pre()  # needed to set up new trace in TracedMooreSUL -> usually called from query()
        assert sul.automaton.current_state == mm.current_state == mm.initial_state
        for i in input_combination:
            mm_out = mm.step(i)
            mm_state = mm.current_state
            sul_out = sul.step(i)
            sul_state = sul.automaton.current_state

            if mm_state != sul_state:
                num_glitches += 1
                mm.current_state = sul_state

            if traced:
                assert sul.traces[-1][-1] == (i, sul_out)
                assert sul.current_trace[-1] == (i, sul_out)

        if traced:
            assert len(sul.traces[-1][1:]) == len(input_combination)
            assert tuple(i for i, o in sul.traces[-1][1:]) == tuple(input_combination)
            assert tuple(i for i, o in sul.current_trace[1:]) == tuple(input_combination)

    # assert sul.num_steps == total_num_steps  # since we always do manual .step() in this test, num_steps is always zero (because only .query() increases it)
    assert sul.num_steps == 0
    assert sul.num_queries == 0

    assert num_glitches == sul.num_glitched_steps
    actual_percentage_glitches = num_glitches / total_num_steps * 100
    if percent_glitches == 0.0:
        assert actual_percentage_glitches == 0
    else:
        assert percent_glitches - 0.1 <= actual_percentage_glitches <= percent_glitches + 0.1
        assert np.isclose(actual_percentage_glitches, percent_glitches, atol=0.05, rtol=0.01)