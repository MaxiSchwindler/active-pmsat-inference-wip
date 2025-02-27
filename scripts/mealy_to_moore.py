""" Copied from pmsat-inference/scripts and adapted for newer AALpy version """

import copy
import sys
from pathlib import Path

from aalpy.automata import MealyMachine, MooreMachine
from aalpy.base import SUL, Oracle
from aalpy.learning_algs import run_Lstar
from aalpy.SULs import MooreSUL
from aalpy.utils import load_automaton_from_file


class PerfectMooreEqOracle(Oracle):
    def __init__(self, alphabet: list, sul: SUL, mm: MooreMachine):
        super().__init__(alphabet, sul)
        self.mm = mm

    def find_cex(self, hypothesis):
        """
        Return a counterexample (inputs) that displays different behavior on system under learning and
        current hypothesis.

        Args:

          hypothesis: current hypothesis

        Returns:

            tuple or list containing counterexample inputs, None if no counterexample is found
        """
        self.reset_hyp_and_sul(hypothesis)
        # print(hypothesis)
        mm = self.mm  # mm: MooreMachine = copy.deepcopy(self.mm)
        assert set(mm.get_input_alphabet()) == set(
            self.alphabet
        ), f"{mm.get_input_alphabet()} != {self.alphabet}"
        dis = mm.find_distinguishing_seq(mm.current_state, hypothesis.current_state, alphabet=self.alphabet)
        if dis is None:
            return None  # no CEX found
        # print("Found cex:", dis)
        # return dis
        assert self.sul.step(None) == hypothesis.step(None)
        for index, inp in enumerate(dis):
            out1 = self.sul.step(inp)
            out2 = hypothesis.step(inp)
            if out1 != out2:
                assert (
                    index == len(dis) - 1
                ), f"Difference in output not on last index? {index} != {len(dis)-1}"
                return dis
                # print("Found difference in output without exception??")
        assert False, "Did not find difference in output on performing CEX??"


def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv
    if len(args) not in [2, 3]:
        print(f"USAGE: python3 {Path(__file__).name} MEALY_FILE.DOT [MOORE_FILE_OR_DIR]")
        exit(1)

    file = args[1]
    mealy: MealyMachine = load_automaton_from_file(file, "mealy", True)
    alphabet = mealy.get_input_alphabet()

    file = Path(file)
    outfilename = file.stem + "_moore" + file.suffix
    print(file)
    print(outfilename)
    if len(args) == 3:
        arg2 = Path(args[2])
        if not arg2.exists():
            # is file

            outfile = arg2.parent / outfilename
        elif arg2.is_dir():
            outfile = arg2 / outfilename
        else:
            # overwrite file
            outfile = arg2
    else:
        outfile = file.parent / outfilename
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # mealy_setup = {
    #     "q1": {"x": ("a", "q1"), "y": ("a", "q2")},
    #     "q2": {"x": ("b", "q2"), "y": ("a", "q3")},
    #     "q3": {"x": ("a", "q2"), "y": ("b", "q3")},
    # }

    moore_setup = {}
    mealy_setup = mealy.to_state_setup()
    all_outputs = sorted(set(o for trans in mealy_setup.values() for o, _ in trans.values()))

    for sid, t in mealy_setup.items():
        outs = sorted(set(o for o, _ in t.values()))
        for o in outs:
            moore_setup[f"{sid}_{o}"] = (
                o,
                {letter: f"{t[letter][1]}_{t[letter][0]}" for letter in alphabet},
            )
        for o in all_outputs:
            key = f"{sid}_{o}"
            if key not in moore_setup:
                moore_setup[key] = (
                    o,
                    {letter: f"{t[letter][1]}_{t[letter][0]}" for letter in alphabet},
                )

    # pp(moore_setup)

    moore = MooreMachine.from_state_setup(moore_setup)
    # moore.visualize()
    new_initial = copy.copy(moore.initial_state)
    new_initial.state_id = "INIT"
    new_initial.output = "INIT"
    moore.states.insert(0, new_initial)
    moore.initial_state = new_initial

    sul = MooreSUL(moore)
    eq = PerfectMooreEqOracle(alphabet, sul, moore)
    moore: MooreMachine = run_Lstar(alphabet, sul, eq, "moore", print_level=0)

    assert moore.states[0] == moore.initial_state
    for state_index in range(1, len(moore.states)):
        moore_starting = copy.deepcopy(moore)
        moore_starting.states.pop(0)
        moore_starting.initial_state = moore_starting.states[state_index]
        dis = moore_starting.find_distinguishing_seq(
            moore_starting.initial_state, moore.initial_state, alphabet
        )
        if dis is None:
            moore = moore_starting
            break

    # moore.visualize()
    moore.save(outfile)
    assert moore.is_minimal(), "Moore machine learned by lstar is not minimal!"


if __name__ == "__main__":
    main()
