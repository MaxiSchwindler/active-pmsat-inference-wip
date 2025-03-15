"""
Written by Benjamin von Berg
"""

from typing import Any

from scipy.stats import binom

from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.GsmNode import GsmNode as Node
from aalpy.learning_algs.general_passive.GsmNode import TransitionInfo
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation


def get_dominant_output(transitions: dict[Any, TransitionInfo]):
    dominant_output, _ = max(transitions.items(), key=lambda x: x[1].count)
    return dominant_output


class GlitchyDeterministicScore(ScoreCalculation):
    def __init__(self, failure_rate: float, certainty: float, purge_mismatches=False):
        super().__init__()
        if not 0 <= failure_rate <= 1:
            raise ValueError("Failure rate must be between 0 and 1")
        self.failure_rate = failure_rate

        if not 0 <= certainty <= 1:
            raise ValueError("Certainty must be between 0 and 1")
        self.certainty = certainty

        self.purge_mismatches = purge_mismatches

    def score_function(self, part: dict[Node, Node]):
        mismatches = 0
        evidence = 0
        for node in set(part.values()):
            for in_sym, trans in node.transitions.items():
                dominant_output = get_dominant_output(trans)
                for out_sym, info in trans.items():
                    if out_sym != dominant_output:
                        mismatches += info.count
                    evidence += info.count
                if self.purge_mismatches:
                    node.transitions[in_sym] = {dominant_output: trans[dominant_output]}

        # Probability of observing at least as many mismatches as we did
        prob = 1 - binom.cdf(mismatches - 1, evidence, self.failure_rate)
        if prob < self.certainty:
            return False
        return prob, evidence


def postprocess(root: Node):
    for node in root.get_all_nodes():
        for in_sym, trans in node.transitions.items():
            dominant_output = get_dominant_output(trans)
            node.transitions[in_sym] = {dominant_output: trans[dominant_output]}
    return root


def run(
    data,
    output_behavior,
    failure_rate: float,
    certainty: float,
    purge_mismatches=False,
    instrument=None,
):
    score = GlitchyDeterministicScore(failure_rate, certainty, purge_mismatches)
    internal = run_GSM(
        data,
        transition_behavior="nondeterministic",
        output_behavior=output_behavior,
        convert=False,
        score_calc=score,
        postprocessing=postprocess,
        instrumentation=instrument,
    )
    return internal.to_automaton(output_behavior, "deterministic")


if __name__ == "__main__":
    pass
    # traces = [
    #   [o, (i, o), ...],
    #   ...
    # ]
    # pmsat_learned = run_pmsat...
    # gsm_learned = run(traces, "moore", failure_rate=0.01, certainty=0.05)