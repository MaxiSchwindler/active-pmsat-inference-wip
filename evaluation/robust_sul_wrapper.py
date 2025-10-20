from aalpy import SUL

class RobustSUL(SUL):
    def __init__(self, sul: SUL, perform_n_times: int = 20, validity_threshold: float = 0.51):
        super().__init__()
        self.sul = sul

        self.perform_n_times = perform_n_times
        self.validity_threshold = validity_threshold * perform_n_times

        self.num_validation_queries = 0
        self.num_validation_steps = 0

    def step(self, letter):
        return self.sul.step(letter)

    def pre(self):
        return self.sul.pre()

    def post(self):
        return self.sul.post()

    def query(self, word):
        self.num_validation_queries += self.perform_n_times - 1
        self.num_validation_steps += (self.perform_n_times - 1) * len(word)

        traces = []
        for _ in range(self.perform_n_times):
            q = self.sul.query(word)

            traces.append(
                q
            )

        majority_trace = None
        for trace in traces:
            if (c := traces.count(trace)) >= self.validity_threshold:
                majority_trace = trace
                print(f"The output sequence {majority_trace} was returned {c} "
                    f"out of {self.perform_n_times} times, which is above the threshold ({self.validity_threshold}).")
                break

        if majority_trace is not None:
            return majority_trace
        else:
            raise SystemExit("No Majority Trace found!")

    def __getattr__(self, item):
        return getattr(self.sul, item)