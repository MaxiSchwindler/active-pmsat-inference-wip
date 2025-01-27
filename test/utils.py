import pytest
from aalpy import generate_random_deterministic_automata, MooreMachine, MooreState

MOORE_CONFIGS = [
    {"num_states": 3, "num_inputs": 3, "num_outputs": 3},
    {"num_states": 5, "num_inputs": 3, "num_outputs": 3},
    {"num_states": 8, "num_inputs": 3, "num_outputs": 3},
    {"num_states": 8, "num_inputs": 3, "num_outputs": 5},
    {"num_states": 8, "num_inputs": 3, "num_outputs": 8},
]


@pytest.fixture
def mm(request) -> MooreMachine:
    num_states = request.param.get("num_states", 8)
    num_inputs = request.param.get("num_inputs", 3)
    num_outputs = request.param.get("num_outputs", 3)
    return generate_random_deterministic_automata("moore",
                                                  num_states,
                                                  num_inputs,
                                                  num_outputs)


def compare_states(state_1: MooreState, state_2: MooreState, should_be_same_object=False):
    """ Compare two MooreStates. If they should not be the same object, compare by id and output."""
    if should_be_same_object:
        return state_1 is state_2
    return state_1.state_id == state_2.state_id and state_1.output == state_2.output