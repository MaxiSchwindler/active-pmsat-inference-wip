from random import shuffle, choice, randint

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import product

from active_pmsatlearn.oracles.RobustEqOracleMixin import RobustEqOracleMixin, NoMajorityTrace


class RobustWMethodEqOracle(Oracle, RobustEqOracleMixin):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states, shuffle_test_set=True,
                 perform_n_times: int = 20, validity_threshold: float = 0.51, max_num_tries=float("inf")):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
            shuffle_test_set: if True, test cases will be shuffled
        """

        assert validity_threshold > 0.5
        Oracle.__init__(self, alphabet, sul)
        RobustEqOracleMixin.__init__(self, perform_n_times, validity_threshold)

        self.m = max_number_of_states
        self.shuffle = shuffle_test_set

        self.max_num_tries = max_num_tries

        self.info["detailed_oracle_info"]["params"]["max_num_tries"] = self.max_num_tries
        self.info["detailed_oracle_info"]["params"]["max_number_of_states"] = self.m
        self.info["detailed_oracle_info"]["params"]["shuffle_test_set"] = self.shuffle

    def test_suite(self, cover, depth, char_set):
        """
        Construct the test suite for the W Method using
        the provided state cover and characterization set,
        exploring up to a given depth.
        Args:

            cover: list of states to cover
            depth: maximum length of middle part
            char_set: characterization set
        """
        # fix the length of the middle part per loop
        # to avoid generating large sequences early on
        char_set = char_set or [()]
        for d in range(depth):
            middle = product(self.alphabet, repeat=d)
            for m in middle:
                for (s, c) in product(cover, char_set):
                    yield s + m + c

    def find_cex(self, hypothesis, return_outputs=False):
        self.info["find_cex_called"] += 1
        num_tries = 0
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # hypothesis.compute_prefixes()  # TODO needed?

        # covers every transition of the specification at least once.
        transition_cover = [
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        ]

        depth = self.m + 1 - len(hypothesis.states)
        for seq in self.test_suite(transition_cover, depth, hypothesis.characterization_set):
            self.reset_hyp_and_sul(hypothesis)
            inputs = []
            outputs_sul = []
            outputs_hyp = []

            for ind, letter in enumerate(seq):
                self.num_steps += 1
                inputs.append(letter)

                out_sul = self.sul.step(letter)
                outputs_sul.append(out_sul)

                out_hyp = hypothesis.step(letter)
                outputs_hyp.append(out_hyp)

                if out_hyp != out_sul:
                    try:
                        valid_cex, cex_inputs, cex_outputs = self.validate_counterexample(inputs,
                                                                                          outputs_sul,
                                                                                          outputs_hyp)
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

                            # no majority trace - go to next sequence as we can't validate OR invalidate this one
                            break

                    if not valid_cex:
                        outputs_sul[:] = cex_outputs[:]
                        continue

                    self.sul.post()

                    self.info["counterexamples_found"] += 1
                    if return_outputs:
                        return cex_inputs, cex_outputs
                    return cex_inputs

        self.info["no_counterexamples_found"] += 1
        if return_outputs:
            return None, []
        return None


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, walks_per_state=12, walk_len=12):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state: number of random walks that should start from each state

            walk_len: length of random walk
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()

    def find_cex(self, hypothesis):

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()
            # fix for non-minimal intermediate hypothesis that can occur in KV
            if not hypothesis.characterization_set:
                hypothesis.characterization_set = [(a,) for a in hypothesis.get_input_alphabet()]

        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        shuffle(states_to_cover)

        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            random_walk = tuple(choice(self.alphabet) for _ in range(randint(1, self.random_walk_len)))

            test_case = prefix + random_walk + choice(hypothesis.characterization_set)

            for ind, i in enumerate(test_case):
                output_hyp = hypothesis.step(i)
                output_sul = self.sul.step(i)
                self.num_steps += 1

                if output_sul != output_hyp:
                    self.sul.post()
                    return test_case[:ind + 1]

        return None
