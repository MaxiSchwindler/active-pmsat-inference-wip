from functools import partial

import aalpy.learning_algs
from aalpy.SULs import MooreSUL, MealySUL
from aalpy.oracles import RandomWMethodEqOracle

import active_pmsatlearn.learnalgo_nomat
from active_pmsatlearn.learnalgo_mat import run_activePmSATLearn
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
    extension_length=(2,3),
    # cex_processing=(True, False),
    glitch_processing=(True, False),
    discard_glitched_traces=(True, False),
    add_cex_as_hard_clauses=(True, False),
)

# add all APMSL(n)_no_<> combinations
for combination in dict_product(apmsl_choices):
    for name in ("APMSL",): #"NAPMSL"):
        alg_name = f"{name}({combination['extension_length']})"
        for k, v in combination.items():
            if not v:
                alg_name += f"_no_{''.join(a[0] for a in k.split('_'))}"
        run_apml_config = partial(run_activePmSATLearn, **apmsl_common_args, **combination, **common_args)
        algorithms[alg_name] = run_apml_config

mat_kwargs = dict(
    cex_processing=True,
    add_cex_as_hard_clauses=True,  # does not work due to unsat currently
    discard_glitched_traces=True,
)

algorithms["NAPMSL(3)_wcp"] = partial(active_pmsatlearn.learnalgo_nomat.run_activePmSATLearn,
                                      extension_length=3,
                                      window_cex_processing=True,
                                      **apmsl_common_args,
                                      **mat_kwargs,
                                      **common_args)
algorithms["NAPMSL(3)_wcp_w3"] = partial(active_pmsatlearn.learnalgo_nomat.run_activePmSATLearn,
                                         sliding_window_size=3,
                                         extension_length=3,
                                         window_cex_processing=True,
                                         **apmsl_common_args,
                                         **mat_kwargs,
                                         **common_args)
algorithms["NAPMSL(3)_no_wcp"] = partial(active_pmsatlearn.learnalgo_nomat.run_activePmSATLearn,
                                         extension_length=3,
                                         window_cex_processing=False,
                                         **apmsl_common_args,
                                         **mat_kwargs,
                                         **common_args)
algorithms["NAPMSL(3)_no_wcp_w3"] = partial(active_pmsatlearn.learnalgo_nomat.run_activePmSATLearn,
                                            extension_length=3,
                                            sliding_window_size=3,
                                            window_cex_processing=False,
                                            **apmsl_common_args,
                                            **mat_kwargs,
                                            **common_args)


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
    "None": lambda sul: None,
    "Perfect": RobustPerfectMooreOracle,
    "Random": RobustRandomWalkOracle,
}

