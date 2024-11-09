from aalpy import MooreState, MooreMachine
from aalpy.SULs import MooreSUL

from active_pmsatlearn.RandomWalkEqOracle import RandomWalkEqOracle
from active_pmsatlearn.NondeterministicMooreMachine import NondeterministicMooreMachine, NondeterministicMooreState


if __name__ == '__main__':

    nondet_states = [
        s1 := NondeterministicMooreState("s0", "a"),
        s2 := NondeterministicMooreState("s1", "b"),
    ]
    s1.transitions["i"].append(s1)
    s1.transitions["i"].append(s2)
    s2.transitions["i"].append(s2)

    nondet_mm = NondeterministicMooreMachine(s1, nondet_states)

    det_states = [
        d1 := MooreState("d1", "a"),
        d2 := MooreState("d2", "b"),
    ]
    d1.transitions["i"] = d2
    d2.transitions["i"] = d2

    det_mm = MooreMachine(d1, det_states)
    det_sul = MooreSUL(det_mm)
    eq_oracle = RandomWalkEqOracle(alphabet=det_sul.automaton.get_input_alphabet(),
                                   sul=det_sul,
                                   num_steps=det_mm.size * 15_000,
                                   reset_after_cex=True,
                                   reset_prob=0.25
                                   )

    cex = eq_oracle.find_cex(nondet_mm)

    if cex:
        print(f"Found cex: {cex}")
    else:
        print(f"No counterexample found!")

    nondet_mm.visualize()