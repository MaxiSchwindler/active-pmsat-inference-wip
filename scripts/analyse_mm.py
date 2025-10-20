import argparse

from aalpy import load_automaton_from_file, MooreMachine, MooreState


def get_target_states_of_input(mm: MooreMachine, input):
    target_states = set()
    for start_state in mm.states:
        if input in start_state.transitions:
            target_state = start_state.transitions[input]
            target_states.add(target_state)
        else:
            print(f"Warning: no transition for input {input} from state {start_state.state_id}")
    return target_states


def get_target_states_per_input(mm: MooreMachine):
    input_to_target_states = dict()
    for input in mm.get_input_alphabet():
        input_to_target_states[input] = get_target_states_of_input(mm, input)
    return input_to_target_states


def get_inputs_reaching_state(mm: MooreMachine, state: MooreState):
    assert state in mm.states
    inputs_reaching_state = set()
    for start_state in mm.states:
        for input, target_state in start_state.transitions.items():
            if target_state == state:
                inputs_reaching_state.add(input)
    return inputs_reaching_state


def get_inputs_reaching_state_per_state(mm: MooreMachine):
    state_to_inputs_reaching_it = dict()
    for state in mm.states:
        state_to_inputs_reaching_it[state] = get_inputs_reaching_state(mm, state)
    return state_to_inputs_reaching_it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    def sep():
        print("---------")


    mm: MooreMachine = load_automaton_from_file(args.file, "moore")

    print(f"Loaded automaton from {args.file}")
    print(f"{len(mm.states)} states")
    print(f"{len(mm.get_input_alphabet())} inputs: {mm.get_input_alphabet()}")
    print(f"{len(set(s.output for s in mm.states))} outputs: {set(s.output for s in mm.states)}")
    print(f"Input complete: {mm.is_input_complete()}")
    sep()

    # inputs to states
    input_to_target_states = get_target_states_per_input(mm)
    for input, target_states in input_to_target_states.items():
        if len(target_states) == 0:
            print(f"No transitions for input {input}")
        elif len(target_states) == 1:
            print(f"Input {input} always transitions to state {next(iter(target_states)).state_id}")
        else:
            print(f"Input {input} has {len(target_states)} targets ({list(t.state_id for t in target_states)})")
    sep()

    state_to_inputs_reaching_it = get_inputs_reaching_state_per_state(mm)
    for state, inputs_reaching_it in state_to_inputs_reaching_it.items():
        if len(inputs_reaching_it) == 0:
            print(f"No inputs reaching state {state.state_id}")
        elif len(inputs_reaching_it) == 1:
            print(f"State {state.state_id} is only reached via input {next(iter(inputs_reaching_it))}")
        else:
            print(f"State {state.state_id} is reached via {len(inputs_reaching_it)} inputs ({inputs_reaching_it})")

if __name__ == '__main__':
    main()