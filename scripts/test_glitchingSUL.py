import itertools
import sys
import os

from aalpy import load_automaton_from_file

from active_pmsatlearn.utils import trace_query
from evaluation.learn_automata import setup_sul


def get_sul_from_file(file: str, glitch_percent: float = 0):
    mm = load_automaton_from_file(file, "moore")
    return setup_sul(mm, glitch_percent=glitch_percent)


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("USAGE: python test_glitchingSUL.py AUTOMATA_FILE GLITCH_PERCENT")
        exit(1)

    automaton_file = args[0]
    if not os.path.exists(automaton_file):
        print(f"{automaton_file} does not exist")
        exit(2)

    if not os.path.splitext(automaton_file)[1] == '.dot':
        print(f"{automaton_file} is not a .dot file")
        exit(3)

    glitch_percent = float(args[1]) if len(args) > 1 else 1.0
    if not (1 <= glitch_percent <= 100):
        print(f"{glitch_percent} must be between 1 and 100")

    sul = get_sul_from_file(automaton_file, glitch_percent)
    print(sul)

    all_input_combinations = list(itertools.product(sul.automaton.get_input_alphabet(), repeat=10))
    traces = []
    for input_combination in all_input_combinations:
        traces.append(trace_query(sul, input_combination))

    complete_num_steps = sum(len(trace) for trace in traces)

    # check what the SUL itself tells us
    print(f"SUL: In {len(traces)} traces, for a total of {complete_num_steps} steps, there were {sul.num_glitched_steps} glitch steps "
          f"({sul.num_glitched_steps / complete_num_steps * 100:.2f}%)")

    # replay on original automaton
    mm = sul.automaton
    num_glitches = 0
    for trace in traces:
        mm.reset_to_initial()
        assert mm.step(None) == trace[0]
        for (input, output) in trace[1:]:
            state_before = mm.current_state
            if (actual_output := mm.step(input)) != output:
                num_glitches += 1
                glitch_state = sul.glitch_transitions[state_before][input]
                assert glitch_state.output == output  # this fails sometimes - why???
                mm.current_state = glitch_state

    print(f"MM: In {len(traces)} traces, for a total of {complete_num_steps} steps, there were {num_glitches} glitch steps "
          f"({num_glitches / complete_num_steps * 100:.2f}%)")




if __name__ == '__main__':
    main()
