import math

class RobustEqOracleMixin:
    """
    Mixin-class to make equality oracle robust against glitches in the SUL.
    Once a possible counterexample has been found, the counterexample is performed a number of times on the SUL,
    and only if a certain percentage of calls yields the same trace, the counterexample is accepted as valid (glitchless)
    and returned
    """
    def __init__(self, perform_n_times: int = 10, validity_threshold: float = 0.7):
        self.perform_n_times = perform_n_times
        self.threshold = math.ceil(validity_threshold * perform_n_times)

    def validate_counterexample(self, inputs, outputs_sul, outputs_hyp):
        print(f"{type(self).__name__} found a possible counterexample {inputs}. "
              f"Performing counterexample {self.perform_n_times} times on SUL to validate.")
        outputs_hyp = tuple(outputs_hyp)
        outputs_sul = tuple(outputs_sul)
        assert len(inputs) == len(outputs_sul) == len(outputs_hyp)
        assert outputs_sul != outputs_hyp

        traces = [tuple(outputs_sul)]
        for _ in range(self.perform_n_times):
            traces.append(tuple(self.sul.query(inputs)))

        majority_trace = None
        for trace in set(traces):
            if (c := traces.count(trace)) >= self.threshold:
                majority_trace = trace
                print(f"The output sequence {majority_trace} was returned {c} "
                      f"out of {self.perform_n_times} times, which is above the threshold ({self.threshold}).")
                break

        if majority_trace is not None and majority_trace != outputs_hyp:
            print("This trace does not fit with the outputs of our hypothesis -> counterexample!")
            assert len(majority_trace) == len(outputs_sul) == len(outputs_hyp)

            if majority_trace == outputs_sul:
                print("The originally found counterexample was validated.")
            else:
                print("The originally found counterexample was not valid, but another one has been found during validation.")
                print(f"{majority_trace=} {outputs_sul=}")
                for i in range(len(majority_trace)):
                    if majority_trace[i] != outputs_hyp[i]:
                        break
                print(f"{i=}")
                majority_trace = tuple(majority_trace[:i + 1])
                outputs_hyp = tuple(outputs_hyp[:i + 1])
                inputs = inputs[:i + 1]
                print(f"{majority_trace=} {outputs_sul=}")

            assert majority_trace[:-1] == outputs_hyp[:-1], f"{majority_trace[:-1]=} != {outputs_hyp[:-1]=}"
            assert majority_trace[-1] != outputs_hyp[-1], f"{majority_trace[-1]=} == {outputs_hyp[-1]=}"

            return True, inputs, majority_trace

        return False, None, None