import itertools
import sys
import os

from aalpy import load_automaton_from_file

from active_pmsatlearn.utils import trace_query
from evaluation.learn_automata import setup_sul

from evaluation.utils import GlitchingSUL


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

    print(f"loading SUL from file {automaton_file}...")
    sul = get_sul_from_file(automaton_file, glitch_percent)
    # sul.fault_type = "enter_random_state"
    print(f"SUL: {sul}")
    assert isinstance(sul, GlitchingSUL)

    input_alphabet = sul.automaton.get_input_alphabet()
    extension_length = len(sul.automaton.states) + 1
    print(
        f"Performing all possible traces of the original automaton "
        f"== (I^(num_statest+1)"
        f"== ({input_alphabet})^({len(sul.automaton.states)} + 1) "
        f"== ({len(input_alphabet)})^({extension_length})) "
        f"== {len(input_alphabet)**extension_length} traces of length {extension_length} "
        f"== {(len(input_alphabet)**extension_length) * extension_length} steps"
    )
    all_input_combinations = list(itertools.product(input_alphabet, repeat=extension_length))
    traces = []
    for input_combination in all_input_combinations:
        traces.append(trace_query(sul, input_combination))
        assert len(traces[-1][1:]) == extension_length, f"Trace {traces[-1]} for input_combination {input_combination} has length {len(traces[-1])}"

    complete_num_steps = sum(len(trace[1:]) for trace in traces)  # !!!

    # check what the SUL itself tells us
    print(f"SUL: In {len(traces)} traces, for a total of {complete_num_steps} steps, there were {sul.num_glitched_steps} glitch steps "
          f"({sul.num_glitched_steps / complete_num_steps * 100:.2f}%)")

    # replay on original automaton
    mm = sul.automaton
    num_glitches = 0
    for trace in traces:
        mm.reset_to_initial()
        assert mm.step(None) == trace[0], f"Initial step of trace should never glitch!"

        for (i, o) in trace[1:]:
            actual_output = mm.step(i)
            # rough approximation, we don't know which state the sul entered next
            if actual_output != o:
                num_glitches += 1

    print(f"MM : In {len(traces)} traces, for a total of {complete_num_steps} steps, there were {num_glitches} glitch steps "
          f"({num_glitches / complete_num_steps * 100:.2f}%)")




if __name__ == '__main__':
    main()
