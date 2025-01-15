import argparse
import atexit
import csv
import json
import multiprocessing
import os
import itertools
import uuid

from datetime import datetime

import numpy as np
from aalpy import bisimilar
from aalpy.utils import load_automaton_from_file
from pebble import ProcessPool
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import active_pmsatlearn
from active_pmsatlearn.defs import EqOracleTermination
from active_pmsatlearn.log import set_current_process_name, std_out_err_redirect_tqdm, get_logger
from active_pmsatlearn.utils import *
from evaluation.generate_automata import *
from evaluation.utils import *
from evaluation.defs import *

logger = get_logger(__file__)


def get_all_automata_files_from_dir(directory: str):
    if not os.path.isdir(directory):
        return []

    return [os.path.join(os.path.abspath(directory), filename) for filename in os.listdir(directory) if (os.path.splitext(filename)[-1] == ".dot")]


def get_generated_automata_files_to_learn(num_automata_per_combination: int, automata_type: Literal["mealy", "moore"],
                                          num_states: int | range, num_inputs: int | range, num_outputs: int | range,
                                          generated_automata_dir: str, reuse_existing_automata: bool = True):
        files = []
        for current_num_states in parse_range(num_states):
            for current_num_inputs in parse_range(num_inputs):
                for current_num_outputs in parse_range(num_outputs):
                    if not is_valid_automata_configuration(automata_type, current_num_states, current_num_inputs, current_num_outputs):
                        continue
                    if reuse_existing_automata:
                        matching_automata = matching_automata_files_in_directory(generated_automata_dir,
                                                                                 automata_type,
                                                                                 current_num_states,
                                                                                 current_num_inputs,
                                                                                 current_num_outputs)
                        files.extend(matching_automata[:num_automata_per_combination])
                        missing = num_automata_per_combination - len(matching_automata)
                    else:
                        missing = num_automata_per_combination

                    for _ in range(missing):
                        _automaton, filename = generate_automaton(automata_type,
                                                                  current_num_states,
                                                                  current_num_inputs,
                                                                  current_num_outputs,
                                                                  generated_automata_dir)
                        files.append(filename)

        return files


def setup_sul(automaton: MealyMachine | MooreMachine, max_num_steps: int | None = None, glitch_percent: float = 0.0):
    if isinstance(automaton, MooreMachine):
        sul = TracedMooreSUL(automaton)
    elif isinstance(automaton, MealyMachine):
        sul = MealySUL(automaton)
        if glitch_percent:
            raise NotImplementedError("TracedMealySUL has not yet been implemented - needed to introduce glitches!")
    else:
        raise TypeError("Invalid automaton type")

    if max_num_steps is not None:
        sul = MaxStepsSUL(sul, max_num_steps)

    if glitch_percent:
        sul = GlitchingSUL(sul, glitch_percent)

    return sul


def check_equal(sul: MooreSUL, learned_mm: MooreMachine):
    if learned_mm is None:
        return False
    if isinstance(sul, GlitchingSUL):
        # For the equality check, don't have the glitches
        sul = sul.sul
    perfect_oracle = RobustPerfectMooreOracle(sul)
    cex, cex_outputs = perfect_oracle.find_cex(learned_mm)
    return True if cex is None else False


def stop_remaining_jobs(pool: ProcessPool):
    logger.info("Stopping remaining jobs in process pool...")
    pool.close()
    pool.stop()
    pool.join()
    logger.info("Processes should be stopped now...")


def result_json_files_to_csv(results_dir):
    results = []
    for file in os.listdir(results_dir):
        if file == "info.json":
            continue
        file_path = os.path.join(results_dir, file)
        with open(file_path, 'r') as f:
            try:
                result = json.load(f)
            except Exception as e:
                print(f"Error while loading file {file_path}: {e}")
                raise e
            results.append(result)
    results_file = os.path.join(results_dir, "all_results.csv")
    logger.info(f"Writing all results to {results_file}...")
    write_results_to_csv(results, results_file)
    logger.info(f"Finished writing results to {results_file}.")


def write_results_info(results_dir, automata_type, automata_files, algorithm_names, learn_num_times, max_num_steps,
                       oracle_types, results_file):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "info.json"), "w") as f:
        json.dump(dict(automata_type=automata_type,
                       automata_files=automata_files,
                       algorithm_names=algorithm_names,
                       oracle_types=oracle_types,
                       learn_num_times=learn_num_times,
                       max_num_steps=max_num_steps,
                       results_file=results_file),
                  f,
                  indent=4)

    atexit.register(result_json_files_to_csv, results_dir)


def write_single_json_result(results_dir, automaton_file, info):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    alg_name = info['algorithm_name']
    alg_name = alg_name.replace(" ", "_")
    alg_name = alg_name.replace("*", "star")

    with open(os.path.join(results_dir, f"{now}_learning_results_{os.path.basename(automaton_file)}_{alg_name}_{uuid.uuid4().hex[:8]}.json"), "w") as f:
        json.dump(info, f, indent=4)


def write_results_to_csv(results: list[dict], results_file: str):
    # TODO: overwrite/append?
    headers = set()
    for d in results:
        headers.update(d.keys())
    headers = list(headers)

    if os.path.splitext(results_file)[-1] != ".csv":
        results_file += ".csv"

    logger.info(f"Writing results to {results_file}")

    with open(results_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def print_results_info(results: list[dict]):
    def sep():
        print("-" * 40)

    algorithms = list(set([entry["algorithm_name"] for entry in results]))
    valid_results = [entry for entry in results if "exception" not in entry]
    results_with_model = [entry for entry in results if entry["learned_automaton_size"] is not None ]

    sep()
    print("RESULTS:")
    sep()
    print(f"Algorithms: {algorithms}")
    print(f"Oracles: {list(set(r['oracle'] for r in results))}")
    sep()

    num_results = len(results)
    num_valid = len(valid_results)
    num_learned = sum(r['learned_correctly'] for r in results)
    num_timed_out = sum(r['timed_out'] for r in valid_results)
    num_bisimilar = sum(r['bisimilar'] for r in results)
    num_returned_model = sum(r['learned_automaton_size'] is not None for r in valid_results)

    print(f"Valid results (no exceptions): {num_valid} of {num_results} ({num_valid / num_results * 100:.2f}%)")
    print(f"Returned models (did not return None): {num_returned_model} of {num_valid} ({num_returned_model / num_valid * 100:.2f}%)")
    if num_valid > 0:
        print(f"Learned correctly: {num_learned} of {num_valid} ({num_learned / num_valid * 100:.2f}%)")
        print(f"Bisimilar: {num_bisimilar} of {num_valid} ({num_bisimilar / num_valid * 100:.2f}%)")
    if num_timed_out:
        num_timed_out_but_correct = sum(r['timed_out'] and r['learned_correctly'] for r in valid_results)
        num_timed_out_but_bisimilar = sum(r['timed_out'] and r['bisimilar'] for r in valid_results)
        print(f"Timed out: {num_timed_out} of {num_results} ({num_timed_out / num_results * 100:.2f}%)")
        print(f"  - Nevertheless correct: {num_timed_out_but_correct}")
        print(f"  - Nevertheless bisimilar: {num_timed_out_but_bisimilar}")
    sep()

    print("Statistics:")
    for stat in ["Precision (all steps)", "Precision (traces)",
                 "Precision per trace (mean)", "Precision per trace (median)",
                 "Strong accuracy (mean)","Strong accuracy (median)",
                 "Medium accuracy (mean)","Medium accuracy (median)",
                 "Weak accuracy (mean)","Weak accuracy (median)"]:
        print(f"  {stat}: ")
        for stat_method in (np.mean, np.median, np.min, np.max):
            val_all = stat_method([r[stat] for r in results])
            val_corr = stat_method([r[stat] for r in (res for res in valid_results if res["learned_correctly"])] or [0])
            val_timed = stat_method([r[stat] for r in (res for res in valid_results if res["timed_out"])] or [0])
            val_not_timed = stat_method([r[stat] for r in (res for res in valid_results if not res["timed_out"])] or [0])
            print(f"    {stat_method.__name__:6}: {val_all:.2f} (with TO: {val_timed:.2f} | no TO: {val_not_timed:.2f})")
    sep()

    num_automata = len(set(r['original_automaton'] for r in results))
    min_num_states = min(r['original_automaton_size'] for r in results)
    max_num_states = max(r['original_automaton_size'] for r in results)
    min_num_inputs = min(r['original_automaton_num_inputs'] for r in results)
    max_num_inputs = max(r['original_automaton_num_inputs'] for r in results)
    min_num_outputs = min(r['original_automaton_num_outputs'] for r in results)
    max_num_outputs = max(r['original_automaton_num_outputs'] for r in results)

    print(f"Unique automata: {num_automata}")
    print(f"Number of states per automaton: {min_num_states} to {max_num_states}")
    print(f"Number of inputs per automaton: {min_num_inputs} to {max_num_inputs}")
    print(f"Number of outputs per automaton: {min_num_outputs} to {max_num_outputs}")
    sep()

    glitch_percent = set(r.get('glitch_percent', None) for r in results)
    assert len(glitch_percent) == 1
    glitch_percent = list(glitch_percent)[0]

    if glitch_percent:
        print(f"Glitch percentage: {glitch_percent}%")
        print("Fault type: 'send_random_input'")
        sep()

    exceptions = [r['exception'] for r in results if "exception" in r]
    unique_exceptions = list(set(exceptions))
    num_exceptions = len(exceptions)

    if num_exceptions:
        print(f"Exceptions: {num_exceptions}")
        print(f"Unique exceptions: {len(unique_exceptions)}")

    for e in unique_exceptions:
        print("\t", e)
        num_times = exceptions.count(e)
        perc_of_exc = num_times / num_exceptions * 100
        perc_of_runs = num_times / num_results * 100
        unique_automata = list(set(r['original_automaton'] for r in results if r.get('exception', None) == e))
        unique_algorithms = list(set(r['algorithm_name'] for r in results if r.get('exception', None) == e))

        print("\t\t",
              f"occurred {num_times} times ({perc_of_exc:.2f}% of all exceptions / {perc_of_runs:.2f}% of all runs)")
        print("\t\t", f"occurred in {len(unique_automata)} unique automata (", end="")

        every_time = []
        for a in unique_automata:
            runs_with_a = sum(1 for r in results if r['original_automaton'] == a)
            runs_with_a_and_e = sum(
                1 for r in results if (r['original_automaton'] == a) and (r.get('exception', None) == e))
            print(f"{runs_with_a_and_e}/{runs_with_a}", end="")
            if a != unique_automata[-1]:
                print(", ", end="")
            else:
                print(')')
            if runs_with_a == runs_with_a_and_e:
                every_time.append(a)

        for a in every_time:
            print("\t\t\t", f"occurred every time with {a}")

        print("\t\t", f"occurred in {len(unique_algorithms)} unique algorithms: {unique_algorithms}")


def calculate_accuracy(true_outputs, learned_outputs):
    total_steps = len(true_outputs)

    # strong: Check if the entire trace matches
    strong = 1 if true_outputs == learned_outputs else 0

    # medium: Number of steps that fit from the start until the first divergence
    medium_fit_steps = 0
    for t, l in zip(true_outputs, learned_outputs):
        if t == l:
            medium_fit_steps += 1
        else:
            break
    medium = medium_fit_steps / total_steps

    # weak: Number of matching steps divided by total steps
    weak_fit_steps = sum(1 for t, l in zip(true_outputs, learned_outputs) if t == l)
    weak = weak_fit_steps / total_steps

    return strong, medium, weak


@timeit("Calculating statistics")
def calculate_statistics(original_automaton: MooreMachine, learned_automaton: MooreMachine):
    input_alphabet = original_automaton.get_input_alphabet()
    extension_length = len(original_automaton.states) + 1
    input_combinations = list(itertools.product(input_alphabet, repeat=extension_length))

    num_completely_correct_traces = 0
    num_outputs = 0
    num_correct_outputs = 0
    strong_accs = []
    medium_accs = []
    weak_accs = []
    precision_per_trace = []

    if learned_automaton is not None:
        for input_combination in input_combinations:
            orig_outputs = original_automaton.execute_sequence(original_automaton.initial_state, input_combination)
            learned_outputs = learned_automaton.execute_sequence(learned_automaton.initial_state, input_combination)

            if orig_outputs == learned_outputs:
                num_completely_correct_traces += 1

            num_outputs += len(orig_outputs)
            num_correct_outputs_trace = sum((a == b) for a, b in zip(orig_outputs, learned_outputs))
            num_correct_outputs += num_correct_outputs_trace

            precision_per_trace.append(num_correct_outputs_trace / len(orig_outputs))

            strong_acc, medium_acc, weak_acc = calculate_accuracy(orig_outputs, learned_outputs)
            strong_accs.append(strong_acc)
            medium_accs.append(medium_acc)
            weak_accs.append(weak_acc)

    return {
        "Number of performed traces": len(input_combinations),
        "Number of correct traces": num_completely_correct_traces,
        "Length of each trace": len(input_combinations[0]),
        "Number of outputs": num_outputs,
        "Number of correct outputs": num_correct_outputs,
        "Precision (all steps)": (num_correct_outputs / num_outputs) if num_outputs > 0 else 0,
        "Precision (traces)": num_completely_correct_traces / len(input_combinations),
        "Precision per trace (mean)": np.mean(precision_per_trace or [0]),
        "Precision per trace (median)": np.median(precision_per_trace or [0]),
        # "Recall": num_correct_outputs / num_outputs ??
        "Strong accuracy (mean)": np.mean(strong_accs or [0]),
        "Strong accuracy (median)": np.median(strong_accs or [0]),
        "Medium accuracy (mean)": np.mean(medium_accs or [0]),
        "Medium accuracy (median)": np.median(medium_accs or [0]),
        "Weak accuracy (mean)": np.mean(weak_accs or [0]),
        "Weak accuracy (median)": np.median(weak_accs or [0]),
    }


def learn_automaton(automaton_type: str, automaton_file: str, algorithm_name: str, oracle_type: str, results_dir: str,
                    max_num_steps: int | None = None, glitch_percent: float = 0.0, print_level: int = 0) -> dict[str, Any]:
    automaton = load_automaton_from_file(automaton_file, automaton_type)
    sul = setup_sul(automaton, max_num_steps, glitch_percent)
    oracle = oracles[oracle_type](sul)
    algorithm: AlgorithmWrapper = eval(AlgorithmWrapper.preprocess_algorithm_call(algorithm_name))
    start_time = time.time()
    output_alphabet = set(s.output for s in automaton.states)

    if multiprocessing.parent_process():
        set_current_process_name(f"{algorithm_name}_{multiprocessing.current_process().pid}")

    logger.info(f"Running {algorithm_name} with keywords {algorithm.unique_keywords} and oracle '{oracle_type}' on an "
                f"automaton with {automaton.size} states, {len(automaton.get_input_alphabet())} inputs, {len(output_alphabet)} outputs. "
                f"({automaton_file}). SUL has {glitch_percent}% glitches.")
    alg_kwargs = dict(alphabet=automaton.get_input_alphabet(), sul=sul, print_level=print_level)
    if oracle is not None:
        if algorithm_name.startswith("APMSL"):
            alg_kwargs['termination_mode'] = EqOracleTermination(oracle)
        else:
            alg_kwargs['eq_oracle'] = oracle

    try:
        learned_model, info = algorithm.run(**alg_kwargs)
        info["max_steps_reached"] = False
    except MaxStepsReached:
        learned_model = None
        info = {"max_steps_reached": True}
    except Exception as e:
        learned_model = None
        info = {'exception': repr(e)}
        import traceback
        logger.warning(f"Unknown exception during algorithm call:\n {traceback.format_exc()}")

    learned_correctly = check_equal(sul, learned_model)
    if not learned_correctly:
        if learned_model:
            logger.warning(f"{algorithm_name} did not learn correctly!")
        else:
            logger.warning(f"{algorithm_name} returned no learned model")

    info["start_time"] = start_time
    info["algorithm_name"] = algorithm_name
    info["oracle"] = oracle_type
    info["learned_correctly"] = learned_correctly
    info["bisimilar"] = bisimilar(sul.automaton, learned_model) if learned_model is not None else False
    stats = calculate_statistics(sul.automaton, learned_model)
    for key, val in stats.items():
        info[key] = float(val)
    info["learned_model"] = str(learned_model) if learned_model is not None else ""

    info["max_num_steps"] = max_num_steps
    info["glitch_percent"] = glitch_percent

    info["automaton_type"] = automaton_type
    info["original_automaton"] = automaton_file
    info["original_automaton_size"] = automaton.size
    info["original_automaton_num_inputs"] = len(automaton.get_input_alphabet())
    info["original_automaton_num_outputs"] = len(set(s.output for s in automaton.states))

    info["algorithm_kwargs"] = {k: v if is_builtin_type(v) else str(v) for k, v in (algorithm.kwargs | alg_kwargs).items()}

    # TODO also write learned model etc
    #  probably: create folder for this learning, like in generated_data
    write_single_json_result(results_dir, automaton_file, info)

    logger.info(f"Finished {algorithm_name} with keywords {algorithm.unique_keywords} and oracle '{oracle_type}'")
    return info


def learn_automaton_wrapper(args: tuple):
    return learn_automaton(*args)


@timeit("Learning automata")
def learn_automata(automata_type: str, automata_files: list[str],
                   algorithm_names: Sequence[str], oracle_types: Sequence[str], results_dir: str,
                   learn_num_times: int = 5, max_num_steps: int | None = None,
                   glitch_percents: list[float] = None, print_level: int = 0):
    if not glitch_percents:
        glitch_percents = [0.0]

    all_combinations = [
        (automata_type, automata_file, algorithm_name, oracle_type, results_dir, max_num_steps, glitch_percent, print_level)
        for (automata_file, algorithm_name, oracle_type, glitch_percent)
        in itertools.product(automata_files, algorithm_names, oracle_types, glitch_percents)
    ]

    logger.info(f"Learning automata in {len(all_combinations)} unique constellations "
                f"(algorithms: {algorithm_names} | oracles: {oracle_types} | {len(automata_files)} files | {len(glitch_percents)} unique glitch percents). "
                f"Each constellation is learned {learn_num_times} times.")

    all_combinations_n_times = [c for c in all_combinations for _ in range(learn_num_times)]

    write_results_info(results_dir, automata_type, automata_files, algorithm_names, learn_num_times, max_num_steps,
                       oracle_types, results_dir)

    if True:
        pool = ProcessPool(max_workers=multiprocessing.cpu_count() // 2)
        atexit.register(stop_remaining_jobs, pool)

        futures = pool.map(learn_automaton_wrapper, all_combinations_n_times)

        results = []
        with logging_redirect_tqdm():
            with std_out_err_redirect_tqdm() as orig_stdout:
                with tqdm(total=len(all_combinations_n_times), file=orig_stdout, dynamic_ncols=True) as pbar:
                    for f in futures.result():
                        results.append(f)
                        pbar.update(1)
    else:
        # dont use multiple processes (debugging)
        results = [learn_automaton_wrapper(args) for args in all_combinations_n_times]
    print_results_info(results)


def main():
    parser = argparse.ArgumentParser(description='Learn automata with different algorithms & oracles. '
                                                 'You can also run interactively.')
    parser.add_argument('-a', '--algorithms', type=AlgorithmWrapper.validate_type, nargs="+", required=True,
                        help='Learning algorithms to learn automata with')
    parser.add_argument('-e', '--explain', action=AlgorithmWrapper.ExplainAlgorithmAction,
                        help="Show help message for only specified parameters and exit")
    parser.add_argument('-o', '--oracles', type=str, nargs="+", choices=oracles.keys(), required=True,
                        help='Equality oracles used during learning')
    parser.add_argument('--learn_num_times', type=int, default=5,
                        help='Number of times to learn the same automaton')
    parser.add_argument('--max_num_steps', type=int, default=float('inf'),
                        help='Maximum number of steps taken in the SUL during learning before aborting')
    parser.add_argument('--glitch_percent', type=float, nargs="+",
                        help='How many percent of steps in the SUL glitch (currently, only fault_mode="send_random_input" is supported). '
                             'You can specify multiple glitch percent at once.')
    parser.add_argument('--learn_all_automata_from_dir', type=str,
                        help='Learn all automata from a directory. If this directory is specified, '
                             'all arguments concerning automata generation except -t are ignored. '
                             'You have to ensure yourself that all of these automata are instances of the '
                             'automaton type given with -t.')
    add_automata_generation_arguments(parser)
    parser.add_argument('--reuse', type=bool, default=True,
                        help="Whether to reuse existing automata or to force generation of new ones")
    parser.add_argument('-rd', '--results_dir', type=str, default="learning_results",
                        help='Directory to store results in.')
    parser.add_argument('-pl', '--print_level', type=int, default=0,
                        help="Print level for all algorithms. Usually ranges from 0 (nothing) to 3 (everything).")

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print("No arguments provided. Switching to interactive mode (experimental).")
        args = get_args_from_input(parser)

    if args.learn_all_automata_from_dir:
        files_to_learn = get_all_automata_files_from_dir(args.learn_all_automata_from_dir)
    else:
        files_to_learn = get_generated_automata_files_to_learn(args.num_automata_per_combination, args.type,
                                                               args.num_states, args.num_inputs, args.num_outputs,
                                                               args.generated_automata_dir, args.reuse)

    max_num_steps = args.max_num_steps if args.max_num_steps != float('inf') else None
    results_dir = new_file(args.results_dir)

    set_current_process_name(os.path.basename(__file__))
    learn_automata(args.type, files_to_learn, args.algorithms, args.oracles, results_dir,
                   args.learn_num_times, max_num_steps, args.glitch_percent, args.print_level)

if __name__ == '__main__':
    main()