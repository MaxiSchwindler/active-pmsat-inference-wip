from abc import ABC
from dataclasses import dataclass
from typing import TypeAlias, Any, Callable

from aalpy import MooreMachine, MealyMachine, Oracle
from aalpy.SULs import MooreSUL, MealySUL

from active_pmsatlearn.oracles import RobustEqOracleMixin
from pmsatlearn.learnalgo import Output, Input

Trace: TypeAlias = tuple[Output | tuple[Input, Output]]
SupportedAutomaton: TypeAlias = MooreMachine | MealyMachine
SupportedSUL: TypeAlias = MooreSUL | MealySUL
PossibleHypothesis: TypeAlias = SupportedAutomaton | None
PmSatLearningInfo: TypeAlias = dict[str, Any]
PmSatReturnTuple: TypeAlias = tuple[SupportedAutomaton, SupportedAutomaton, PmSatLearningInfo]
HypothesesWindow: TypeAlias = dict[int, PmSatReturnTuple]
HeuristicFunction: TypeAlias = Callable[[SupportedAutomaton, PmSatLearningInfo, list[Trace]], float]


#####################################
#              ACTIONS              #
#####################################


class Action(ABC):
    pass


@dataclass
class Terminate(Action):
    hyp: PossibleHypothesis
    hyp_stoc: PossibleHypothesis
    pmsat_info: PmSatLearningInfo


@dataclass
class Continue(Action):
    next_min_num_states: int = None
    additional_data: Any = None


#####################################
#        TERMINATION MODES          #
#####################################


class TerminationMode(ABC):
    pass


class RequirementsTermination(TerminationMode, ABC):
    pass


class ImprovementTermination(TerminationMode, ABC):
    pass


@dataclass
class EqOracleTermination(TerminationMode):
    """ Terminates once the eq_oracle does not find a counterexample to the given hypothesis. """
    eq_oracle: RobustEqOracleMixin


@dataclass
class GlitchThresholdTermination(RequirementsTermination):
    """ Terminates once the percentage of glitches of a given hypothesis falls below the set threshold"""
    threshold: float = 1.0


class ScoreImprovementTermination(ImprovementTermination):
    pass


class GlitchImprovementTermination(ImprovementTermination):
    """
    Terminates once the peak hypothesis stays at the same number of states over 2 consecutive rounds
    and the percentage of glitches did not improve in the newer round.
    """
    pass
