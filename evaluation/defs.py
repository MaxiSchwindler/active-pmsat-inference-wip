from functools import partial

import aalpy.learning_algs
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.oracles import RandomWMethodEqOracle

from active_pmsatlearn.learnalgo import run_activePmSATLearn
from active_pmsatlearn.oracles import RobustRandomWalkEqOracle, RobustPerfectMooreEqOracle
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
    run_apml_config = partial(run_activePmSATLearn, pm_strategy='rc2', timeout=10*MINUTES, **combination, **common_args)
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


class RobustPerfectMooreOracle(RobustPerfectMooreEqOracle):
    def __init__(self, sul: MooreSUL):
        super().__init__(sul)


class RobustRandomWalkOracle(RobustRandomWalkEqOracle):
    def __init__(self, sul: MooreSUL | MealySUL):
        super().__init__(
            alphabet=sul.automaton.get_input_alphabet(),
            sul=sul,
            num_steps=sul.automaton.size * 5_000,
            reset_after_cex=True,
            reset_prob=0.09
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
    "Perfect": RobustPerfectMooreOracle,
    "Random": RobustRandomWalkOracle,
    #"Random WMethod": RandomWMethodOracle,
}

