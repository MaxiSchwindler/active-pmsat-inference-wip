import contextlib

from aalpy.base import Oracle
from aalpy.base.SUL import CacheSUL
from aalpy.SULs import MooreSUL

from .RobustEqOracleMixin import RobustEqOracleMixin


class RobustPerfectMooreEqOracle(Oracle, RobustEqOracleMixin):
    def __init__(self, sul: MooreSUL):
        alphabet = sul.automaton.get_input_alphabet()
        Oracle.__init__(self, alphabet, sul)
        RobustEqOracleMixin.__init__(self)

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
        if isinstance(self.sul, CacheSUL):
            mm = self.sul.sul.automaton
        else:
            mm = self.sul.automaton

        from evaluation.utils import GlitchingSUL
        cm = self.sul.dont_glitch if isinstance(self.sul, GlitchingSUL) else contextlib.nullcontext

        assert set(mm.get_input_alphabet()) == set(self.alphabet), f"{mm.get_input_alphabet()} != {self.alphabet}"

        with cm():
            if (dis := mm.find_distinguishing_seq(mm.current_state, hypothesis.current_state, self.alphabet)) is None:
                return None, []  # no CEX found

            assert self.sul.step(None) == hypothesis.step(None)

            outputs_sul = []
            for index, inp in enumerate(dis):
                out_sul = self.sul.step(inp)
                out_hyp = hypothesis.step(inp)
                outputs_sul.append(out_sul)

                if out_sul != out_hyp:
                    assert (
                            index == len(dis) - 1
                    ), f"Difference in output not on last index? {index} != {len(dis) - 1}"
                    return dis, outputs_sul

            assert False, "Did not find difference in output on performing CEX?"

    def validate_counterexample(self, inputs, outputs_sul, outputs_hyp):
        return True, inputs, outputs_sul

