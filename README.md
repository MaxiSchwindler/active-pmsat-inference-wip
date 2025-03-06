# Active PMSAT-inference
Work-in-progress of an active version of the [PMSAT-inference](https://gitlab.com/felixwallner/pmsat-inference) algorithm.

## Setup
* Currently, we need both a fork of AALpy and PMSAT-inference. 
  * On the same level as the clone of this repository, clone these forks:
    * https://gitlab.com/MaxiSchwindler/pmsat-inference
    * https://github.com/MaxiSchwindler/AALpy
    * The folder structure should look like this:
      * ```
        <ROOT>/
        ├── AALpy/
        ├── active-pmsat-inference-wip/
        ├── pmsat-inference/
* Install the requirements, preferably in a virtual environment
  * Create new venv: `python -m venv venv`
  * Activate venv: 
    * Linux: `source venv/bin/activate`
    * Git Bash on Windows: `source venv/Scripts/activate`
    * Windows CMD: `venv\Scripts\activate`
  * Install requirements: 
    * If only using the algorithm: `pip install -r active_pmsatlearn/requirements.txt`
    * If also using evaluation code: `pip install -r evaluation/requirements.txt`
    * For active development on this repo: `pip install -r requirements-dev.txt`
    * All of the above: `pip install -r active_pmsatlearn/requirements.txt -r evaluation/requirements.txt -r requirements-dev.txt `
* Make sure that the following paths (all the repos) are on the PYTHONPATH: 
  * `<ROOT>/AALpy`
  * `<ROOT>/active-pmsat-inference-wip`
  * `<ROOT>/pmsat-inference`
    * In PyCharm, you can simply mark these two directories as "sources root", which will add them to the PYTHONPATH when you execute something via PyCharm's run/debug configuration.
    * In a terminal:
      * Automatically:
        * In the directory of this repository (`<ROOT>/active-pmsat-inference-wip`):
          * UNIX/Git Bash on Windows: run `source scripts/set_pythonpath.sh` 
          * Windows CMD: run `scripts/set_pythonpath`
      * Manually:
        * UNIX/Git Bash on Windows, run `export PYTHONPATH="./active-pmsat-inference-wip;./pmsat-inference;./AALpy;$PYTHONPATH"` in the `<ROOT>` directory
        * Windows CMD: run `set PYTHONPATH=./active-pmsat-inference-wip;./pmsat-inference;./AALpy;%PYTHONPATH%;` in the `<ROOT>` directory
        * Alternatively, in a python script which imports one of these packages, add the path to the parent directory of the package to `sys.path`:
          * ```python
            import sys
            sys.path = ['.', '../AALpy', '../pmsat-inference'] + sys.path 

## Structure

### [active_pmsatlearn/](active_pmsatlearn)

Contains the active algorithm (in [active_pmsatlearn/learnalgo.py](active_pmsatlearn/learnalgo_mat.py))

### [evaluation/](evaluation)

Contains programs to evaluate the performance of the algorithm
* [generate_automata.py](evaluation/generate_automata.py): Generate automata via the command line or interactively.
  * You can choose either a range or a fixed number of states/inputs/outputs the generated automata should have, and how many automata *of each combination* should be created. See `python evaluation/generate_automata.py -h` for help
* [learn_automata.py](evaluation/learn_automata.py): Learn automata via the command line or interactively.
  * Specify algorithms to learn with by name
  * Specify oracles to learn with by name
  * ... see `-h` for more

### [scripts/](scripts)
* [run_apmsl_on_file_or_dir.py](scripts/run_apmsl_on_file_or_dir.py): Run the active PMSAT-inference algorithm on a given .dot file or on all .dot files in a given directory