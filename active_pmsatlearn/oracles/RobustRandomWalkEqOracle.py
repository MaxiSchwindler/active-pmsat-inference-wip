import random

from aalpy.base import Oracle, SUL

from active_pmsatlearn.oracles.RobustEqOracleMixin import RobustEqOracleMixin


###
# The code used in this file is copied from the AALpy project:
# https://github.com/DES-Lab/AALpy
#
# Following file/class has been copied:
# -- aalpy/oracles/RandomWalkEqOracle.py
#
# Adaptions to the existing code have been made:
# -- removed handling of onfwm, mdp and smm as it is not relevant for us
# -- inherit from RobustEqOracleMixin, perform validity check and add ability to return outputs
#
###


class RobustRandomWalkEqOracle(Oracle, RobustEqOracleMixin):
    """
    Equivalence oracle where queries contain random inputs. After every step, 'reset_prob' determines the probability
    that the system will reset and a new query asked.
    """

    def __init__(self, alphabet: list, sul: SUL, num_steps=5000, reset_after_cex=True, reset_prob=0.09,
                 perform_n_times=10, validity_threshold=0.7):
        """

        Args:
            alphabet: input alphabet

            sul: system under learning

            num_steps: number of steps to be preformed

            reset_after_cex: if true, num_steps will be preformed after every counter example, else the total number
                or steps will equal to num_steps

            reset_prob: probability that the new query will be asked

            perform_n_times: number of times a possible counterexample will be performed on the SUL

            validity_threshold: if a counterexample has been performed n times, n * validity_threshold traces
                must be identical for the counterexample to be accepted (assumed to be glitchless). Must be > 0.5
                        """

        assert validity_threshold > 0.5
        Oracle.__init__(self, alphabet, sul)
        RobustEqOracleMixin.__init__(self, perform_n_times, validity_threshold)

        self.step_limit = num_steps
        self.reset_after_cex = reset_after_cex
        self.reset_prob = reset_prob
        self.random_steps_done = 0
        self.automata_type = 'det'

    def find_cex(self, hypothesis, return_outputs=True):
        inputs = []
        outputs_sul = []
        outputs_hyp = []
        self.reset_hyp_and_sul(hypothesis)

        while self.random_steps_done < self.step_limit:
            self.num_steps += 1
            self.random_steps_done += 1

            if random.random() <= self.reset_prob:
                self.reset_hyp_and_sul(hypothesis)
                inputs.clear()
                outputs_sul.clear()
                outputs_hyp.clear()

            inputs.append(random.choice(self.alphabet))

            out_sul = self.sul.step(inputs[-1])
            outputs_hyp.append(out_sul)

            out_hyp = hypothesis.step(inputs[-1])
            outputs_sul.append(out_hyp)

            if out_sul != out_hyp:

                valid_cex, cex_inputs, cex_outputs = self.validate_counterexample(inputs, outputs_sul, outputs_hyp)
                if not valid_cex:
                    continue

                if self.reset_after_cex:
                    self.random_steps_done = 0
                self.sul.post()

                if return_outputs:
                    return cex_inputs, cex_outputs
                return cex_inputs

        return None

    def reset_counter(self):
        if self.reset_after_cex:
            self.random_steps_done = 0
