import argparse
import atexit
import csv
import json
import multiprocessing
import os
import itertools
import uuid

from datetime import datetime

from aalpy import bisimilar
from aalpy.utils import load_automaton_from_file
from pebble import ProcessPool

import active_pmsatlearn
from active_pmsatlearn.log import set_current_process_name
from active_pmsatlearn.utils import *
from evaluation.generate_automata import *
from evaluation.utils import *
from evaluation.defs import algorithms, oracles, RobustPerfectMooreOracle

logger = active_pmsatlearn.log.get_logger(__file__)


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
            result = json.load(f)
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

    sep()
    print("RESULTS:")
    sep()
    print(f"Algorithms: {algorithms}")
    print(f"Oracles: {list(set(r['oracle'] for r in results))}")
    sep()

    num_results = len(results)
    num_valid = len(valid_results)
    num_learned = sum(r['learned_correctly'] for r in results)
    num_aborted = sum(r['timed_out'] for r in valid_results)
    num_bisimilar = sum(r['bisimilar'] for r in results)

    print(f"Valid results (no exceptions): {num_valid} of {num_results} ({num_valid / num_results * 100:.2f}%)")
    if num_valid > 0:
        print(f"Learned correctly: {num_learned} of {num_valid} ({num_learned / num_valid * 100:.2f}%)")
        print(f"Bisimilar: {num_bisimilar} of {num_valid} ({num_bisimilar / num_valid * 100:.2f}%)")
    if num_aborted:
        print(f"Aborted: {num_aborted} of {num_results} ({num_aborted / num_results * 100:.2f}%)")
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


def learn_automaton(automaton_type: str, automaton_file: str, algorithm_name: str, oracle_type: str, results_dir: str,
                    max_num_steps: int | None = None, glitch_percent: float = 0.0, print_level: int = 0) -> dict[str, Any]:
    automaton = load_automaton_from_file(automaton_file, automaton_type)
    sul = setup_sul(automaton, max_num_steps, glitch_percent)
    oracle = oracles[oracle_type](sul)
    algorithm = algorithms[algorithm_name]
    start_time = time.time()
    output_alphabet = set(s.output for s in automaton.states)

    if multiprocessing.parent_process():
        set_current_process_name(f"{algorithm_name}_{multiprocessing.current_process().pid}")

    logger.info(f"Running {algorithm_name} with keywords {algorithm.unique_keywords} and oracle '{oracle_type}' on an "
                f"automaton with {automaton.size} states, {len(automaton.get_input_alphabet())} inputs, {len(output_alphabet)} outputs. "
                f"({automaton_file})")
    alg_kwargs = dict(alphabet=automaton.get_input_alphabet(), sul=sul, print_level=print_level)
    if oracle is not None:
        alg_kwargs['eq_oracle'] = oracle

    try:
        learned_model, info = algorithm(**alg_kwargs)
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

    info["max_num_steps"] = max_num_steps
    info["glitch_percent"] = glitch_percent

    info["automaton_type"] = automaton_type
    info["original_automaton"] = automaton_file
    info["original_automaton_size"] = automaton.size
    info["original_automaton_num_inputs"] = len(automaton.get_input_alphabet())
    info["original_automaton_num_outputs"] = len(set(s.output for s in automaton.states))

    write_single_json_result(results_dir, automaton_file, info)

    logger.info(f"Finished {algorithm_name} with keywords {algorithm.unique_keywords} and oracle '{oracle_type}'")
    return info


def learn_automaton_wrapper(args: tuple):
    return learn_automaton(*args)


@timeit("Learning automata")
def learn_automata(automata_type: str, automata_files: list[str],
                   algorithm_names: Sequence[str], oracle_types: Sequence[str], results_dir: str,
                   learn_num_times: int = 5, max_num_steps: int | None = None,
                   glitch_percent: float = 0.0, print_level: int = 0):
    all_combinations = [
        (automata_type, automata_file, algorithm_name, oracle_type, results_dir, max_num_steps, glitch_percent, print_level)
        for (automata_file, algorithm_name, oracle_type)
        in itertools.product(automata_files, algorithm_names, oracle_types)
    ]

    logger.info(f"Learning automata in {len(all_combinations)} unique constellations "
                f"(algorithms: {algorithm_names} | oracles: {oracle_types} | and {len(automata_files)} files). "
                f"Each constellation is learned {learn_num_times} times.")

    all_combinations_n_times = [c for c in all_combinations for _ in range(learn_num_times)]

    write_results_info(results_dir, automata_type, automata_files, algorithm_names, learn_num_times, max_num_steps,
                       oracle_types, results_dir)

    if True:
        pool = ProcessPool(max_workers=multiprocessing.cpu_count() // 2)
        atexit.register(stop_remaining_jobs, pool)

        futures = pool.map(learn_automaton_wrapper, all_combinations_n_times)
        results = [f for f in futures.result()]
    else:
        # dont use multiple processes (debugging)
        results = [learn_automaton_wrapper(args) for args in all_combinations_n_times]
    print_results_info(results)


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Learn automata with different algorithms & oracles. '
                                                     'You can also run interactively.')
        parser.add_argument('-a', '--algorithms', type=str, nargs="+", choices=algorithms.keys(), required=True,
                            help='Learning algorithms to learn automata with')
        parser.add_argument('-o', '--oracles', type=str, nargs="+", choices=oracles.keys(), required=True,
                            help='Equality oracles used during learning')
        parser.add_argument('--learn_num_times', type=int, default=5,
                            help='Number of times to learn the same automaton')
        parser.add_argument('--max_num_steps', type=int, default=None,
                            help='Maximum number of steps taken in the SUL during learning before aborting')
        parser.add_argument('--glitch_percent', type=float, default=0.0,
                            help='How many percent of steps in the SUL glitch (currently, only fault_mode="discard" is supported)')
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
        args = parser.parse_args()

        selected_algorithms = args.algorithms
        selected_oracles = args.oracles
        learn_num_times = args.learn_num_times
        max_num_steps = args.max_num_steps
        glitch_percent = args.glitch_percent
        learn_from_dir = args.learn_all_automata_from_dir
        num_automata_per_combination = args.num_automata_per_combination
        automata_type = args.type
        num_states = args.num_states
        num_inputs = args.num_inputs
        num_outputs = args.num_outputs
        generated_automata_dir = args.generated_automata_dir
        reuse_existing_automata = args.reuse
        results_dir = new_file(args.results_dir)
        print_level = args.print_level

    else:
        selected_algorithms = get_user_choices("Select algorithms to learn with", algorithms.keys())
        selected_oracles = get_user_choices("Select oracles to learn with", oracles.keys())
        learn_num_times = int(input("Enter number of times to learn the same automaton: "))
        max_num_steps = input("Enter maximum number of steps to take in the SUL before aborting: ")
        max_num_steps = int(max_num_steps) if max_num_steps else None
        glitch_percent = input("Enter glitch percent (default: 0.0)")
        glitch_percent = float(glitch_percent) if glitch_percent else 0.0

        automata_type = input("Enter type of automata to generate: ")
        if automata_type not in SUPPORTED_AUTOMATA_TYPES:
            raise TypeError(f"Unsupported type of automata: '{automata_type}'")

        learn_from_dir = bool(input("Learn all automata from a directory? (y/n) (default: no): ") in ("y", "Y", "yes"))
        if learn_from_dir:
            learn_from_dir = input(f"Enter directory (all contained automata must be {automata_type} automata): ")
        else:
            num_automata_per_combination = int(input("Enter number of automata to generate/learn (default: 5): ") or 5)
            num_states = parse_range(input("Enter number of states per automaton: "))
            num_inputs = parse_range(input("Enter number of inputs per automaton: "))
            num_outputs = parse_range(input("Enter number of outputs per automaton: "))
            generated_automata_dir = input("Enter directory to store generated automata in: ") or "generated_automata"
            reuse_existing_automata = input("Reuse existing automata? (y/n) (default: yes): ") in ("y", "Y", "yes", "")

        results_dir = new_file(input("Enter path to results directory: ") or "learning_results")
        print_level = int(input("Enter the print level for the algorithms (0 - 3) (default: 0): ") or 0)

    if learn_from_dir:
        files_to_learn = get_all_automata_files_from_dir(learn_from_dir)
    else:
        files_to_learn = get_generated_automata_files_to_learn(num_automata_per_combination, automata_type,
                                                               num_states, num_inputs, num_outputs,
                                                               generated_automata_dir, reuse_existing_automata)

    set_current_process_name(os.path.basename(__file__))
    learn_automata(automata_type, files_to_learn, selected_algorithms, selected_oracles, results_dir,
                   learn_num_times, max_num_steps, glitch_percent, print_level)

if __name__ == '__main__':
    main()