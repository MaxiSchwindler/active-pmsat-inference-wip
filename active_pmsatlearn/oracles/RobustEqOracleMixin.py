import math
import time
from functools import wraps

NONE = 0
INFO = 1
DETAIL = 2

DEBUG_LEVEL = NONE


def log(msg, level=INFO):
    if DEBUG_LEVEL >= level:
        print(msg)


class NoMajorityTrace(Exception):
    pass


class RobustEqOracleMixin:
    """
    Mixin-class to make equality oracle robust against glitches in the SUL.
    Once a possible counterexample has been found, the counterexample is performed a number of times on the SUL,
    and only if a certain percentage of calls yields the same trace, the counterexample is accepted as valid (glitchless)
    and returned
    """
    def __init__(self, perform_n_times: int = 20, validity_threshold: float = 0.51):
        self.perform_n_times = perform_n_times
        self.threshold = math.ceil(validity_threshold * perform_n_times)
        self.eq_query_time = 0
        self.num_steps_validation = 0

        self.info = dict(
            counterexamples_investigated=0,
            counterexamples_validated=0,
            counterexamples_invalidated=0,
            no_majority_trace_found=0,
            find_cex_called=0,
            continued_after_no_majority_trace=0,
            aborted_after_no_majority_trace=0,
            detailed_oracle_info=dict(
                params=dict(
                    perform_n_times=self.perform_n_times,
                    validity_threshold=self.threshold,
                ),
            ),
            counterexamples_found=0,
            no_counterexamples_found=0,
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, 'find_cex'):
            original_find_cex = getattr(cls, 'find_cex')

            @wraps(original_find_cex)
            def timed_find_cex(self, *args, **kwargs):
                eq_query_start = time.time()
                result = original_find_cex(self, *args, **kwargs)
                self.eq_query_time += time.time() - eq_query_start

                return result

            # Replace the method in the subclass with the timed version
            setattr(cls, 'find_cex', timed_find_cex)

    def validate_counterexample(self, inputs, outputs_sul, outputs_hyp):
        log(f"{type(self).__name__} found a possible counterexample. "
            f"Performing counterexample {self.perform_n_times} times on SUL to validate.")
        log(f"{inputs=}\n{outputs_sul=}\n{outputs_hyp=}", level=DETAIL)
        outputs_hyp = tuple(outputs_hyp)
        outputs_sul = tuple(outputs_sul)
        assert None not in inputs
        assert len(inputs) == len(outputs_sul) == len(outputs_hyp)
        assert outputs_sul != outputs_hyp
        assert outputs_sul[:-1] == outputs_hyp[:-1], f"Output in SUL and HYP should be identical except for last step! {outputs_sul[:-1]=} != {outputs_hyp[:-1]=}"
        assert outputs_sul[-1] != outputs_hyp[-1], f"Output in SUL and HYP should differ in the last step! {outputs_sul[-1]=} == {outputs_hyp[-1]=}"
        self.info["counterexamples_investigated"] += 1

        def _query_sul(inputs_to_query):
            if hasattr(self, 'num_steps'):
                self.num_steps += len(inputs_to_query)
            if hasattr(self, 'num_queries'):
                self.num_queries += 1

            steps_before = self.sul.num_steps
            queries_before = self.sul.num_queries

            res = self.sul.query(inputs_to_query)

            self.num_steps_validation += len(inputs_to_query)

            # do this so that the num_steps in the SUL don't get influenced by the oracle, as usual
            if self.sul.num_steps != steps_before:
                self.sul.num_steps -= len(inputs_to_query)
            if self.sul.num_queries != queries_before:
                self.sul.num_queries -= 1
            assert self.sul.num_steps == steps_before
            assert self.sul.num_queries == queries_before

            return res

        traces = [tuple(outputs_sul)]
        for _ in range(self.perform_n_times):
            traces.append(tuple(_query_sul(inputs)))
            log(f"Collected trace: {traces[-1]}", level=DETAIL)

        majority_trace = None
        for trace in set(traces):
            if (c := traces.count(trace)) >= self.threshold:
                majority_trace = trace
                log(f"The output sequence {majority_trace} was returned {c} "
                    f"out of {self.perform_n_times} times, which is above the threshold ({self.threshold}).")
                break

        if majority_trace is not None and majority_trace != outputs_hyp:
            log("This trace does not fit with the outputs of our hypothesis -> counterexample!")
            assert len(majority_trace) == len(outputs_sul) == len(outputs_hyp)

            if majority_trace == outputs_sul:
                log("The originally found counterexample was validated.")
            else:
                log("The originally found counterexample was not valid, but another one has been found during validation.")
                for i in range(len(majority_trace)):
                    if majority_trace[i] != outputs_hyp[i]:
                        break
                majority_trace = tuple(majority_trace[:i + 1])
                outputs_hyp = tuple(outputs_hyp[:i + 1])
                inputs = inputs[:i + 1]

            assert majority_trace[:-1] == outputs_hyp[:-1], f"{majority_trace[:-1]=} != {outputs_hyp[:-1]=}"
            assert majority_trace[-1] != outputs_hyp[-1], f"{majority_trace[-1]=} == {outputs_hyp[-1]=}"

            self.info["counterexamples_validated"] += 1
            return True, inputs, majority_trace

        elif majority_trace is None:
            # cannot handle this - in what state should the SUL be? (Maybe need to reset CEX search)? Maybe just discard last input?
            log(f"For input sequence {inputs}, no output sequence occurred more than {self.threshold} times. ")
            self.info["no_majority_trace_found"] += 1
            raise NoMajorityTrace

        elif majority_trace == outputs_hyp:
            log(f"For input sequence {inputs}, the majority output sequence was identical to the hypothesis ({outputs_hyp}).")

        # if we return false, we also want to bring the SUL into the correct state
        log(f"Bringing the SUL into correct state...")
        while tuple(_query_sul(inputs)) != tuple(majority_trace):
            pass

        self.info["counterexamples_invalidated"] += 1
        return False, inputs, majority_trace