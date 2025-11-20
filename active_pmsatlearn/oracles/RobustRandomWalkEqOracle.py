import random

from aalpy.base import Oracle, SUL

from active_pmsatlearn.oracles.RobustEqOracleMixin import RobustEqOracleMixin, NoMajorityTrace


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

    def __init__(self, alphabet: list, sul: SUL, num_steps=50, reset_after_cex=True, reset_prob=0.09,
                 perform_n_times=20, validity_threshold=0.51, max_num_tries=5):
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

            max_num_tries: how many times finding a majority trace is attempted before returning "no counterexample"
                           if None, don't abort if no majority trace is found any number of times
                        """

        assert validity_threshold > 0.5
        Oracle.__init__(self, alphabet, sul)
        RobustEqOracleMixin.__init__(self, perform_n_times, validity_threshold)

        self.step_limit = num_steps
        self.reset_after_cex = reset_after_cex
        self.reset_prob = reset_prob
        self.random_steps_done = 0
        self.automata_type = 'det'
        self.max_num_tries = max_num_tries if max_num_tries is not None else float("inf")

        self.info["detailed_oracle_info"]["params"]["max_num_tries"] = self.max_num_tries
        self.info["detailed_oracle_info"]["params"]["reset_prob"] = self.reset_prob
        self.info["detailed_oracle_info"]["params"]["reset_after_cex"] = self.reset_after_cex
        self.info["detailed_oracle_info"]["params"]["step_limit"] = self.step_limit

    def find_cex(self, hypothesis, return_outputs=True):
        inputs = []
        outputs_sul = []
        outputs_hyp = []
        num_tries = 0
        self.reset_hyp_and_sul(hypothesis)
        self.info["find_cex_called"] += 1

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
            outputs_sul.append(out_sul)

            out_hyp = hypothesis.step(inputs[-1])
            outputs_hyp.append(out_hyp)

            if out_sul != out_hyp:

                try:
                    valid_cex, cex_inputs, cex_outputs = self.validate_counterexample(inputs, outputs_sul, outputs_hyp)
                except NoMajorityTrace:
                    num_tries += 1
                    if num_tries >= self.max_num_tries:
                        self.info["aborted_after_no_majority_trace"] += 1
                        if return_outputs:
                            return None, []
                        return None
                    else:
                        self.info["continued_after_no_majority_trace"] += 1
                        self.reset_hyp_and_sul(hypothesis)
                        inputs.clear()
                        outputs_sul.clear()
                        outputs_hyp.clear()
                        continue

                if not valid_cex:
                    outputs_sul[:] = cex_outputs[:]
                    continue

                if self.reset_after_cex:
                    self.random_steps_done = 0
                self.sul.post()

                self.info["counterexamples_found"] += 1
                if return_outputs:
                    return cex_inputs, cex_outputs
                return cex_inputs

        self.info["no_counterexamples_found"] += 1
        if return_outputs:
            return None, []
        return None

    def reset_counter(self):
        if self.reset_after_cex:
            self.random_steps_done = 0
