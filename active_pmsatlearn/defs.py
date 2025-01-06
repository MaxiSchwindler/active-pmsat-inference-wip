from typing import TypeAlias, Any, Callable

from aalpy import MooreMachine, MealyMachine
from aalpy.SULs import MooreSUL, MealySUL

from pmsatlearn.learnalgo import Output, Input

Trace: TypeAlias = tuple[Output | tuple[Input, Output]]
SupportedAutomaton: TypeAlias = MooreMachine | MealyMachine
SupportedSUL: TypeAlias = MooreSUL | MealySUL
PossibleHypothesis: TypeAlias = SupportedAutomaton | None
PmSatLearningInfo: TypeAlias = dict[str, Any]
HypothesesWindow: TypeAlias = dict[int, tuple[SupportedAutomaton, SupportedAutomaton, PmSatLearningInfo]]
HeuristicFunction: TypeAlias = Callable[[SupportedAutomaton, PmSatLearningInfo, list[Trace]], float]
