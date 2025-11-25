import argparse
import collections
from collections import defaultdict
from pathlib import Path
from typing import Any
import json


def remove_traces(d: dict):
    if "detailed_learning_info" in d and isinstance(d["detailed_learning_info"], dict):
        for round_data in d["detailed_learning_info"]["learning_rounds"].values():
            if isinstance(round_data, dict):
                round_data.pop("traces_used_to_learn", None)
    return d


def read_old_style_result(filename: Path, remove_traces_used_to_learn: bool = True) -> dict[str, Any]:
    with open(filename, "r") as f:
        try:
            if remove_traces_used_to_learn:
                res = json.load(f, object_hook=remove_traces)
            else:
                res = json.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            raise e
        return res


def _old_config_to_new(old_config: dict) -> dict:
    new_config = {}
    for k, v in old_config.items():
        if k not in (
                "sul", "automaton_type", "heuristic_function",
                "input_completeness_preprocessing", "cex_processing", "glitch_processing", "replay_glitches",
                "termination_mode", "pm_strategy", "cost_scheme",
        ):
            try:
                v = eval(v)
            except Exception as e:
                print(f"Failed to evaluate {k} ('{v}'): {e}")
                raise ValueError from e

        if k == "input_completeness_preprocessing":
            # ic was 'random_suffix' by default and if True
            assert v in ("False", "True", "random_suffix"), v
            mode = "random_suffix"
            enabled = v in ("random_suffix", "True")
            new_config[k] = (enabled, mode)

        elif k == "cex_processing":
            assert v in ("True", "False", "random_suffix", "all_suffixes"), v
            if v == "True":
                enabled = True
                mode = "all_suffixes"
            elif v == "False":
                enabled = False
                mode = "all_suffixes"
            elif v == "all_suffixes":
                enabled = True
                mode = "all_suffixes"
            else:
                assert v == "random_suffix"
                enabled = True
                mode = "random_suffix"

            new_config[k] = (enabled, mode)

        elif k == "glitch_processing":
            assert v in ("True", "False", "random_suffix"), v
            enabled = v in ("True", "random_suffix")
            mode = "random_suffix"
            new_config["reproduce_glitches"] = (enabled, mode)

        elif k == "replay_glitches":
            assert v in ("True", "False", "('original_suffix', 'random_suffix')"), v
            enabled = v in ("True", "('original_suffix', 'random_suffix')")
            modes = ["original_suffix", "random_suffix"]
            new_config["replay_glitches"] = (enabled, modes)

        elif k == "state_prefix_coverage_steps_per_round":
            assert isinstance(v, (int, tuple, list)), v
            if isinstance(v, int):
                nsteps = v
                rprob = 0.09
                mode = "balanced_traces"
                on_all_hyps = False
            elif len(v) == 3:
                nsteps, rprob, mode = v
                on_all_hyps = False
            else:
                assert len(v) == 4
                nsteps, rprob, mode, on_all_hyps = v
            assert on_all_hyps is False

            new_config["state_prefix_coverage"] = (nsteps, rprob, mode)

        elif k == "random_steps_per_round_with_reset_prob":
            assert isinstance(v, (int, tuple, list))
            if isinstance(v, int):
                nsteps = v
                rprob = 0.09
            else:
                assert isinstance(v, (tuple, list))
                assert len(v) == 2
                nsteps, rprob = v

            new_config["random_words"] = (nsteps, rprob)

        elif k == "window_cex_processing":
            assert isinstance(v, bool)
            new_config["window_cex_processing"] = v

        elif k == "random_steps_per_round":
            # Not used anymore
            assert v == 0

        elif k == "transition_coverage_steps":
            # Not used anymore
            assert v == 0

        else:
            new_config[k] = v

    return new_config


def _old_generated_traces_to_new(old: dict) -> tuple[dict[str, int], dict[str, int]]:
    new = {}
    rounds = old["detailed_learning_info"]["learning_rounds"]

    counter_traces = defaultdict(int)
    counter_steps = defaultdict(int)

    for round_data in rounds.values():
        for key, value in round_data.items():
            if not key.startswith("additional_traces_"):
                continue
            if "preprocessing" in key:
                pkey = key[len("additional_traces_preprocessing_"):]
            elif "postprocessing" in key:
                pkey = key[len("additional_traces_postprocessing_"):]
            else:
                pkey = key[len("additional_traces_"):]

            counter_traces[pkey] += len(value)
            counter_steps[pkey] += sum(len(t[1:]) for t in value)

            # round_data.pop(key)

    return counter_traces, counter_steps


def _get_new_algname(oldname, new_config) -> str:
    basename = oldname.split("(")[0]
    assert basename == "APMSL"

    param_strings = []
    if new_config["input_completeness_preprocessing"][0]:
        param_strings.append("ic")

    if new_config["cex_processing"][0]:
        mode = new_config["cex_processing"][1]
        if mode == "random_suffix":
            mode_s = "random"
        elif mode == "all_suffixes":
            mode_s = "all"
        else:
            assert False
        param_strings.append(f"cex={mode_s}")

    if (spc_steps := new_config["state_prefix_coverage"][0]) != 0:
        param_strings.append(f"spc={spc_steps}")

    if (rw_steps := new_config["random_words"][0]) != 0:
        param_strings.append(f"rw={rw_steps}")

    if new_config["replay_glitches"][0]:
        param_strings.append(f"replay")

    if new_config["reproduce_glitches"][0]:
        param_strings.append(f"repro")

    if new_config["window_cex_processing"]:
        param_strings.append("wcp")

    return f"{basename}({','.join(param_strings)})"


def old_style_to_new_style_results(old: dict[str, Any]) -> dict[str, Any]:
    new = dict()
    new["configuration"] = _old_config_to_new(old["detailed_learning_info"]["params"])

    num_gen_traces, num_gen_steps = _old_generated_traces_to_new(old)
    new["num_generated_traces"] = num_gen_traces
    new["num_generated_steps"] = num_gen_steps

    old.pop("detailed_learning_info")
    old.pop("sul_transitions")
    old.pop("max_steps_reached")
    old.pop("algorithm_kwargs")

    for k, v in old.items():
        new[k] = v

    new["algorithm_name"] = _get_new_algname(old["algorithm_name"], new["configuration"])

    return new


def _old_filename_to_new(filename: str, alg_name: str) -> str:
    suffix = filename.rsplit("_", 1)[-1]
    prefix = filename.split(".dot_")[0]

    alg_name = alg_name.replace(" ", "_")
    alg_name = alg_name.replace("*", "star")
    return f"{prefix}_{alg_name}_{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_results_dir')
    parser.add_argument('output_results_dir')
    args = parser.parse_args()

    input_dir = Path(args.input_results_dir)
    output_dir = Path(args.output_results_dir)
    assert input_dir.is_dir()
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f'Rewriting results from {input_dir} to {output_dir}')
    for filename in input_dir.glob('*.json'):
        if filename.name == "info.json":
            continue
        print(f"Processing {filename}...")
        results = read_old_style_result(filename)
        new_results = old_style_to_new_style_results(results)
        new_filename = _old_filename_to_new(filename.name, new_results["algorithm_name"])
        new_results["_original_filename"] = filename.name
        with open(output_dir / new_filename, 'w') as f:
            json.dump(new_results, f, indent=4)
        print(f"Written to {output_dir/new_filename}")








if __name__ == '__main__':
    main()