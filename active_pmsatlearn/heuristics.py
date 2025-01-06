import numpy as np

from active_pmsatlearn.defs import SupportedAutomaton, PmSatLearningInfo, Trace
from active_pmsatlearn.log import get_logger
logger = get_logger("APMSL")


def simple_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace]):
    logger.debug_ext(f"Calculating simple heuristic...")
    score = 0

    if learned_model is None:
        return -100

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    if len(glitched_delta_freq) > 0:
        score += 1 / np.mean(glitched_delta_freq)
        logger.debug_ext(f"Adding inverse of mean of glitched delta frequencies ({np.mean(glitched_delta_freq)}) to {score=}")
    else:
        score += 1
        logger.debug_ext(f"No glitched delta frequencies; {score=}")

    dominant_delta_freq = learning_info["dominant_delta_freq"]
    dominant_delta_freq = [freq for freq in dominant_delta_freq if freq > 0]  # note: 0-freqs are dominant transitions without evidence -> can't be input complete without these

    score -= 1 / np.min(dominant_delta_freq)
    logger.debug_ext(f"Subtracting inverse of minimum of dominant delta frequencies ({np.min(dominant_delta_freq)}) from {score=}")

    return float(score)  # cast numpy away (nicer output)


def intermediary_heuristic(*args, **kwargs):
    # basically the simple heuristic plus a reward for lower numbers of glitches
    return advanced_heuristic(*args, **kwargs, punish_input_incompleteness=False, punish_unreachable_states=False)


def advanced_heuristic(learned_model: SupportedAutomaton, learning_info: PmSatLearningInfo, traces: list[Trace],
                       *, reward_lower_glitch_mean=True, reward_lower_num_glitches=True, punish_unreachable_states=True,
                       punish_low_dominant_freq=True, punish_input_incompleteness=True):
    score = 0

    if learned_model is None:
        return -100

    glitched_delta_freq = learning_info["glitched_delta_freq"]
    dominant_delta_freq = learning_info["dominant_delta_freq"]

    # reward lower mean of glitched frequencies
    # ADDS between 0 and 1
    if reward_lower_glitch_mean:
        if len(glitched_delta_freq) > 0:
            score += 1 / np.mean(glitched_delta_freq)

    # reward lower numbers of glitches
    # ADDS between 0 and 1
    if reward_lower_num_glitches:
        num_glitches = len(learning_info["glitch_steps"])
        if num_glitches > 0:
            score += 1 / num_glitches
        else:
            score += 1

    # penalize low dominant frequencies
    # SUBTRACTS between 0 and 1
    if punish_low_dominant_freq:
        dominant_delta_freq_without_zero = [freq for freq in dominant_delta_freq if freq > 0]
        score -= 1 / np.min(dominant_delta_freq_without_zero)

    # penalize unreachable states (in absolute numbers!) a LOT
    # SUBTRACTS between 0 and inf
    if punish_unreachable_states:
        learned_model.compute_prefixes()
        num_unreachable = sum(1 for state in learned_model.states if state.prefix is None)
        score -= num_unreachable

    # punish dominant frequencies that are 0 (-> not input complete)
    # SUBTRACTS between 0 and inf
    if punish_input_incompleteness:
        num_dominant_zero_freq = sum(1 for freq in dominant_delta_freq if freq == 0)
        score -= num_dominant_zero_freq

    return float(score)  # cast numpy away (nicer output)
