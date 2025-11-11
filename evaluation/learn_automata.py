import argparse
import atexit
import csv
import json
import multiprocessing
import os
import itertools
import pprint
import statistics
import uuid
import sys

from datetime import datetime
from pathlib import Path

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
from evaluation.utils import print_results_info

logger = get_logger(__file__)


def get_all_automata_files_from_dir(directory: str):
    if not os.path.isdir(directory):
        return []

    return [os.path.join(os.path.abspath(directory), filename)
            for filename in os.listdir(directory)
            if (os.path.splitext(filename)[-1] == ".dot")]


def valid_dot_file_or_directory(path):
    if os.path.isdir(path):
        files = get_all_automata_files_from_dir(path)
        if len(files) == 0:
            raise argparse.ArgumentTypeError(f"Directory {path} does not contain any valid .dot files.")
        return files

    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File or directory '{path}' does not exist.")

    if not path.endswith(".dot"):
        raise argparse.ArgumentTypeError(f"File '{path}' must have a .dot extension.")

    return os.path.abspath(path)


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


def setup_sul(automaton: MealyMachine | MooreMachine, max_num_steps: int | None = None, glitch_percent: float = 0.0, glitch_mode: str = None):
    if isinstance(automaton, MooreMachine):
        sul = TracedMooreSUL(automaton)
    elif isinstance(automaton, MealyMachine):
        sul = MealySUL(automaton)
        if glitch_percent:
            raise NotImplementedError("TracedMealySUL has not yet been implemented - needed to introduce glitches!")
    else:
        raise TypeError("Invalid automaton type")

    if max_num_steps is not None:
        if glitch_percent:
            raise NotImplementedError("Due to the isinstance-check for TracedMooreSUL in GlitchingSUL, max_num_steps and glitch_percent can currently not be specified simultaneously")
        sul = MaxStepsSUL(sul, max_num_steps)

    if glitch_percent:
        assert glitch_mode is not None, f"If glitch_percentage is given, glitch_mode must also be specified!"
        sul = GlitchingSUL(sul, glitch_percentage=glitch_percent, fault_type=glitch_mode)

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
        if file in ("info.json", "call.txt"):
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

    with open(os.path.join(results_dir, "call.txt"), 'w') as f:
        f.write(" ".join(sys.argv))

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


def compute_accuracy(
    automaton_a: MooreMachine | None,
    automaton_b: MooreMachine | None,
    num_samples: int = 100_000,
    reset_prob: float = 0.09,
) -> float:
    if reset_prob <= 0:
        raise ValueError("Reset Probability should be > 0")

    if automaton_b is None or automaton_a is None:
        return 0.

    inputs_a = sorted(set(automaton_a.get_input_alphabet()))
    inputs_b = sorted(set(automaton_b.get_input_alphabet()))

    if inputs_a != inputs_b:
        raise ValueError("Only automaton with the same input alphabet can be compared")

    if bisimilar(automaton_a, automaton_b):
        return 1.

    random_words = []
    for _ in range(num_samples):
        random_words.append([])
        while True:
            inp = random.choice(inputs_a)
            random_words[-1].append(inp)
            if random.random() < reset_prob:
                break

    assert len(random_words) == num_samples

    correct = 0
    for random_word in random_words:
        out_a = automaton_a.execute_sequence(automaton_a.initial_state, random_word)
        out_b = automaton_b.execute_sequence(automaton_b.initial_state, random_word)
        if out_a == out_b:
            correct += 1

    return correct / num_samples


def learn_automaton(automaton_type: str, automaton_file: str, algorithm_name: str, oracle_type: str, results_dir: str,
                    max_num_steps: int | None = None, glitch_percent: float = 0.0, glitch_mode: str = None, print_level: int = 0) -> dict[str, Any]:
    automaton = load_automaton_from_file(automaton_file, automaton_type)
    if not automaton.is_input_complete():
        logger.warning(f"Given automaton {automaton_file} is not input complete, but APSML expects input-completeness. "
                       f"Forcing input-completeness via self-loops.")
        automaton.make_input_complete()

    sul = setup_sul(automaton, max_num_steps, glitch_percent, glitch_mode)
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
    alg_kwargs.update(algorithm.kwargs_from_meta_knowledge(glitch_percent=glitch_percent))

    if oracle is not None:
        if algorithm_name.startswith("APMSL"):
            alg_kwargs['termination_mode'] = EqOracleTermination(oracle)
        else:
            alg_kwargs['eq_oracle'] = oracle

    if callable(tm := algorithm.kwargs.get("termination_mode", None)):
        # if the algorithm has a keyword "termination_mode" defined, which is callable, call that callable with the glitch_percent
        # TODO: maybe do this via kwargs_from_meta_knowledge?
        algorithm.kwargs["termination_mode"] = tm(glitch_percent)

    try:
        learned_model, info = algorithm.run(**alg_kwargs)
        info["max_steps_reached"] = False
    except MaxStepsReached:
        learned_model = None
        info = {"max_steps_reached": True}
    except Exception as e:
        learned_model = None
        import traceback
        info = {'exception': traceback.format_exc()}
        logger.warning(f"Unknown exception during algorithm call:\n {traceback.format_exc()}")

    info["sul_transitions"] = getattr(sul, "taken_transitions", None)  # retrieve before check_equal, since this does steps in the sul
    info["sul_num_glitched_steps"] = getattr(sul, "num_glitched_steps", 0)

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
    info["Accuracy"] = compute_accuracy(sul.automaton, learned_model)
    info["learned_model"] = str(learned_model) if learned_model is not None else ""
    info["learned_model_input_complete"] = learned_model.is_input_complete() if learned_model is not None else False
    info["ground_truth_input_complete"] = sul.automaton.is_input_complete()

    if "learned_automaton_size" not in info:
        info["learned_automaton_size"] = learned_model.size if learned_model is not None else 0

    if "timed_out" not in info:
        info["timed_out"] = False

    info["max_num_steps"] = max_num_steps
    info["glitch_percent"] = glitch_percent
    info["glitch_mode"] = glitch_mode

    info["automaton_type"] = automaton_type
    info["original_automaton"] = automaton_file
    info["original_automaton_size"] = automaton.size
    info["original_automaton_num_inputs"] = len(automaton.get_input_alphabet())
    info["original_automaton_num_outputs"] = len(set(s.output for s in automaton.states))

    info["algorithm_kwargs"] = {k: v if is_builtin_type(v) else str(v) for k, v in (algorithm.kwargs | alg_kwargs).items()}

    if oracle is not None:
        info |= oracle.info

    # TODO also write learned model etc
    #  probably: create folder for this learning, like in generated_data
    write_single_json_result(results_dir, automaton_file, info)

    logger.info(f"Finished {algorithm_name} with keywords {algorithm.unique_keywords} and oracle '{oracle_type}'")
    return info


def learn_automaton_wrapper(kwargs: dict):
    try:
        return learn_automaton(**kwargs)
    except Exception as e:
        import traceback
        logger.warning(f"Exception raised during learn_automaton():\n {traceback.format_exc()}")


@timeit("Learning automata")
def learn_automata(automata_type: str, automata_files: list[str],
                   algorithm_names: Sequence[str], oracle_types: Sequence[str], results_dir: str,
                   learn_num_times: int = 5, max_num_steps: int | None = None,
                   glitch_percents: list[float] = None, glitch_modes: list[str] = None, print_level: int = 0,
                   num_processes: int = 3, dry_run: bool = False):
    if not glitch_percents:
        glitch_percents = [0.0]
    if not glitch_modes:
        glitch_modes = [None]

    all_combinations = [
        dict(automaton_type=automata_type,
             automaton_file=automata_file,
             algorithm_name=algorithm_name,
             oracle_type=oracle_type,
             results_dir=results_dir,
             max_num_steps=max_num_steps,
             glitch_percent=glitch_percent,
             glitch_mode=glitch_mode,
             print_level=print_level)
        for (automata_file, algorithm_name, oracle_type, glitch_percent, glitch_mode)
        in itertools.product(automata_files, algorithm_names, oracle_types, glitch_percents, glitch_modes)
    ]

    logger.info(f"Learning automata in {len(all_combinations)} unique constellations "
                f"(algorithms: {algorithm_names} | oracles: {oracle_types} | {len(automata_files)} files | glitch percents: {glitch_percents} | glitch modes: {glitch_modes}). "
                f"Each constellation is learned {learn_num_times} times ({learn_num_times * len(all_combinations)} total runs).")
    if dry_run:
        logger.info(f"Files to be learned: ")
        pprint(automata_files)
        return

    # all_combinations_n_times = [c for c in all_combinations for _ in range(learn_num_times)]
    all_combinations_n_times = all_combinations * learn_num_times

    write_results_info(results_dir, automata_type, automata_files, algorithm_names, learn_num_times, max_num_steps,
                       oracle_types, results_dir)

    if num_processes > 1:
        pool = ProcessPool(max_workers=num_processes)
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
        results = []
        with logging_redirect_tqdm():
            with std_out_err_redirect_tqdm() as orig_stdout:
                with tqdm(total=len(all_combinations_n_times), file=orig_stdout, dynamic_ncols=True) as pbar:
                    for args in all_combinations_n_times:
                        results.append(learn_automaton_wrapper(args))
                        pbar.update(1)
    print_results_info(results)


def validate_arguments(args):
    """Ensure that automata generation arguments are required if no directory is given."""
    required_automata_gen_args = ['num_automata_per_combination', 'num_states', 'num_inputs', 'num_outputs']

    if not args.files:
        # If files are given, ignore automata generation arguments (they must not be given)
        # Otherwise, ensure all automata generation arguments are provided
        missing_args = [arg for arg in required_automata_gen_args if getattr(args, arg, None) is None]
        if missing_args:
            print(f"Error: The following arguments are required unless --learn_all_automata_from_dir is specified: "
                  f"{', '.join('--' + arg.replace('_', '-') for arg in missing_args)}")
            sys.exit(1)

    if args.glitch_percent and any(g > 0 for g in args.glitch_percent) and not args.glitch_mode:
        print(f"Error: If glitch_percent is given, glitch_mode must also be specified.")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description='Learn automata with different algorithms & oracles. '
                                                 'You can also run interactively.',
                                     fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
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
    parser.add_argument('--glitch_mode', type=str, nargs="+", choices=GlitchingSUL.FAULT_TYPES,
                        help='How the SUL glitches. '
                             'You can specify multiple glitch modes at once.')
    parser.add_argument('-f', '--files', type=valid_dot_file_or_directory, nargs="+",
                        help='Specify one or more automata (as paths to .dot files) to be learned. '
                             'If you specify a path to a directory, all contained .dot files will be learned. '
                             'If this option is specified, all arguments concerning automata generation except '
                             '-t are ignored (and not required). '
                             'You have to ensure yourself that all of these automata are instances of the '
                             'automaton type given with -t.')
    add_automata_generation_arguments(parser, required=False)
    parser.add_argument('--reuse', type=bool, default=True,
                        help="Whether to reuse existing automata or to force generation of new ones")
    parser.add_argument('-rd', '--results_dir', type=str, default="learning_results",
                        help='Directory to store results in.')
    parser.add_argument('-pl', '--print_level', type=int, default=0,
                        help="Print level for all algorithms. Usually ranges from 0 (nothing) to 3 (everything).")
    parser.add_argument('-np', '--num_processes', type=int, default=3,
                        help="Number of processes to run concurrently")
    parser.add_argument('-dr', '--dry-run', action='store_true', help=argparse.SUPPRESS)

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print("No arguments provided. Switching to interactive mode (experimental).")
        args = get_args_from_input(parser)

    validate_arguments(args)

    if args.files:
        files_to_learn = []
        for file_or_files in args.files:
            if isinstance(file_or_files, list):
                files_to_learn.extend(file_or_files)
            else:
                files_to_learn.append(file_or_files)
    else:
        files_to_learn = get_generated_automata_files_to_learn(args.num_automata_per_combination, args.type,
                                                               args.num_states, args.num_inputs, args.num_outputs,
                                                               args.generated_automata_dir, args.reuse)

    max_num_steps = args.max_num_steps if args.max_num_steps != float('inf') else None
    results_dir = new_file(args.results_dir)

    set_current_process_name(os.path.basename(__file__))
    learn_automata(
        automata_type=args.type,
        automata_files=files_to_learn,
        algorithm_names=args.algorithms,
        oracle_types=args.oracles,
        results_dir=results_dir,
        learn_num_times=args.learn_num_times,
        max_num_steps=max_num_steps,
        glitch_percents=args.glitch_percent,
        glitch_modes=args.glitch_mode,
        print_level=args.print_level,
        num_processes=args.num_processes,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()