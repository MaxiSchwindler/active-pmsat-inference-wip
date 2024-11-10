from functools import partial

import aalpy.learning_algs
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.base import Oracle, SUL
from aalpy.base.SUL import CacheSUL
from aalpy.oracles import RandomWMethodEqOracle

from active_pmsatlearn.learnalgo import run_activePmSATLearn
from active_pmsatlearn.RandomWalkEqOracle import RandomWalkEqOracle
from evaluation.utils import dict_product

SECOND = SECONDS = 1
MINUTE = MINUTES = 60
HOUR = HOURS = MINUTE * 60

common_args = dict(automaton_type='moore',
                   return_data=True,
                   )

run_kv = partial(aalpy.learning_algs.run_KV, cex_processing=None, **common_args)
run_kv_rs = partial(aalpy.learning_algs.run_KV, cex_processing='rs', **common_args)
run_lstar = partial(aalpy.learning_algs.run_Lstar, cex_processing=None, **common_args)
run_lstar_rs = partial(aalpy.learning_algs.run_Lstar, cex_processing='rs', **common_args)

algorithms = {
    "KV": run_kv,
    "KV (RS)": run_kv_rs,
    "L*": run_lstar,
    "L* (RS)": run_lstar_rs,
    # plus e.g.:
    # "ActivePMSL(2)
    # "ActivePMSL(2)_no_glitch_processing
    # "ActivePMSL(2)_no_cex_processing_no_glitch_processing
    # "ActivePMSL(2)_no_cex_processing_no_input_completeness_processing_no_glitch_processing
    # "ActivePMSL(2)_only_cex_processing
}

# keywords for APML
apml_choices = dict(
    extension_length=(2, 3, 4),
    input_completeness_processing=(True, False),
    cex_processing=(True, False),
    glitch_processing=(True, False),
)

# add all ActivePMSL(n)_no_<> combinations
for combination in dict_product(apml_choices):
    alg_name = f"ActivePMSL({combination['extension_length']})"
    for k, v in combination.items():
        if not v:
            alg_name += f"_no_{k}"
    run_apml_config = partial(run_activePmSATLearn, pm_strategy='rc2', timeout=10*MINUTES, allowed_glitch_percentage=2, **combination, **common_args)
    algorithms[alg_name] = run_apml_config

# add ActivePMSL(n)_only_<> combinations
for el in apml_choices['extension_length']:
    alg_name = f"ActivePMSL({el})"
    algorithms[f"{alg_name}_only_input_completeness_processing"] = algorithms[f"{alg_name}_no_cex_processing_no_glitch_processing"]
    algorithms[f"{alg_name}_only_cex_processing"] = algorithms[f"{alg_name}_no_input_completeness_processing_no_glitch_processing"]
    algorithms[f"{alg_name}_only_glitch_processing"] = algorithms[f"{alg_name}_no_input_completeness_processing_no_cex_processing"]

# create .unique_keywords attribute
for alg in algorithms.values():
    alg.unique_keywords = {k: alg.keywords[k] for k in alg.keywords if k not in common_args}


class PerfectMooreOracle(Oracle):
    def __init__(self, sul: MooreSUL):
        alphabet = sul.automaton.get_input_alphabet()
        super().__init__(alphabet, sul)

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
        assert set(mm.get_input_alphabet()) == set(self.alphabet), f"{mm.get_input_alphabet()} != {self.alphabet}"

        if (dis := mm.find_distinguishing_seq(mm.current_state, hypothesis.current_state, self.alphabet)) is None:
            return None  # no CEX found

        assert self.sul.step(None) == hypothesis.step(None)

        for index, inp in enumerate(dis):
            out_sul = self.sul.step(inp)
            out_hyp = hypothesis.step(inp)
            if out_sul != out_hyp:
                assert (
                    index == len(dis) - 1
                ), f"Difference in output not on last index? {index} != {len(dis)-1}"
                return dis

        assert False, "Did not find difference in output on performing CEX??"


class RandomWalkOracle(RandomWalkEqOracle):
    def __init__(self, sul: MooreSUL | MealySUL):
        super().__init__(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            num_steps=sul.automaton.size * 5_000,
            reset_after_cex=True,
            reset_prob=0.25
        )


class RandomWMethodOracle(RandomWMethodEqOracle):
    def __init__(self, sul: MooreSUL | MealySUL):
        super().__init__(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            walks_per_state=sul.automaton.size * 5,
            walk_len=sul.automaton.size * 5
        )


oracles = {
    "Perfect": PerfectMooreOracle,
    "Random": RandomWalkOracle,
    "Random WMethod": RandomWMethodOracle,
}

