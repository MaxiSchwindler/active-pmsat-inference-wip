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
    # "APMSL(2)
    # "APMSL(2)_no_glitch_processing
    # "APMSL(2)_no_cex_processing_no_glitch_processing
    # "APMSL(2)_no_cex_processing_no_input_completeness_processing_no_glitch_processing
    # "APMSL(2)_only_cex_processing
}

apmsl_common_args = dict(
    pm_strategy='rc2',
    timeout=10*MINUTES,
)

apmsl_choices = dict(
    extension_length=(2,),
    # cex_processing=(True, False),
    glitch_processing=(True, False),
    discard_glitched_traces=(True, False),
    add_cex_as_hard_clauses=(True, False),
)

# add all APMSL(n)_no_<> combinations
for combination in dict_product(apmsl_choices):
    alg_name = f"APMSL({combination['extension_length']})"
    for k, v in combination.items():
        if not v:
            alg_name += f"_no_{''.join(a[0] for a in k.split('_'))}"
    run_apml_config = partial(run_activePmSATLearn, **apmsl_common_args, **combination, **common_args)
    algorithms[alg_name] = run_apml_config

# create .unique_keywords attribute
for alg in algorithms.values():
    alg.unique_keywords = {k: alg.keywords[k] for k in alg.keywords if k not in common_args and k not in apmsl_common_args}


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


oracles = {
    "Perfect": RobustPerfectMooreOracle,
    "Random": RobustRandomWalkOracle,
}

