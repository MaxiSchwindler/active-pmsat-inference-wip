from collections import defaultdict
from typing_extensions import Generic

from aalpy.SULs import MooreSUL
from aalpy.automata import MooreMachine, MooreState
from aalpy.base.Automaton import Automaton, AutomatonState, InputType, OutputType, AutomatonStateType


class NondeterministicMooreState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output: OutputType = output
        self.transitions: dict[InputType, set[NondeterministicMooreState]] = defaultdict(set)

    @staticmethod
    def from_moore_state(moore_state: MooreState):
        return NondeterministicMooreState(moore_state.state_id, moore_state.output)


class NondeterministicMooreMachine(Automaton[NondeterministicMooreState[InputType, OutputType]]):
    def __init__(self, initial_state: NondeterministicMooreState, states: list[NondeterministicMooreState]):
        super().__init__(initial_state, states)
        self.current_state: set[NondeterministicMooreState] = {initial_state}

    def reset_to_initial(self):
        self.current_state = {self.initial_state}

    def step(self, letter):
        next_states = set()
        for state in self.current_state:
            if letter in state.transitions:
                next_states.update(state.transitions[letter])
        self.current_state = next_states
        return [s.output for s in self.current_state]

    def make_input_complete(self, missing_transition_go_to='self_loop'):
        assert missing_transition_go_to in {'self_loop', 'sink_state'}

        input_al = self.get_input_alphabet()

        if self.is_input_complete():
            return

        if missing_transition_go_to == 'sink_state':
            sink_state = NondeterministicMooreState(state_id='sink', output='sink_state')
            self.states.append(sink_state)

        for state in self.states:
            for i in input_al:
                if i not in state.transitions.keys():
                    if missing_transition_go_to == 'sink_state':
                        state.transitions[i].add(sink_state)
                    else:
                        state.transitions[i].add(state)

    def execute_sequence(self, origin_state, seq):
        return super().execute_sequence(origin_state, seq)

    def save(self, file_path='LearnedModel', file_type='dot'):
        # Changes in aalpy/utils/FileHandler.py
        super().save(file_path, file_type)

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Automaton':
        """
        First state in the state setup is the initial state.
        Example state setup:
        state_setup = {
                "a": ("a", {"x": {"b1"}, "y": {"a", "b1"}}),
                "b1": ("b", {"x": {"b2"}, "y": {"a"}}),
                "b2": ("b", {"x": {"b3"}, "y": {"a"}}),
                "b3": ("b", {"x": {"b4"}, "y": {"a"}}),
                "b4": ("b", {"x": {"c"}, "y": {"a"}}),
                "c": ("c", {"x": {"a"}, "y": {"a"}}),
            }

        Args:

            state_setup: map from state_id to tuple(output and transitions_dict)

        Returns:

            Nondeterministic Moore machine
        """

        # build states with state_id and output
        states = {key: NondeterministicMooreState(key, val[0]) for key, val in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, target_state_id in state_setup[state_id][1].items():
                state.transitions[_input].add(states[target_state_id])

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        mm = NondeterministicMooreMachine(states[0], states)

        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

        return mm

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.output, {i: [s.state_id for s in states] for i, states in s.transitions.items()})

        return state_setup_dict


def hyp_stoc_to_nondet_mm(hyp_stoc: MooreMachine) -> NondeterministicMooreMachine:
    from .utils import get_input_from_stoc_trans
    nondet_states = [NondeterministicMooreState.from_moore_state(state) for state in hyp_stoc.states]
    initial_state = [s for s in nondet_states if s.state_id == hyp_stoc.initial_state.state_id][0]
    nondet_mm = NondeterministicMooreMachine(initial_state, nondet_states)

    for state in hyp_stoc.states:
        nondet_state = nondet_mm.get_state_by_id(state.state_id)
        for inp, next_state in state.transitions.items():
            clean_inp = get_input_from_stoc_trans(inp)
            nondet_next_state = nondet_mm.get_state_by_id(next_state.state_id)
            nondet_state.transitions[clean_inp].add(nondet_next_state)

    return nondet_mm