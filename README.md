# Active PMSAT-inference
Work-in-progress of an active version of the [PMSAT-inference](https://gitlab.com/felixwallner/pmsat-inference) algorithm.

## Setup
* Install the requirements, preferably in a virtual environment
  * Create new venv: `python -m venv venv`
  * Activate venv: 
    * Linux: `source venv/bin/activate`
    * Git Bash on Windows: `source venv/Scripts/activate`
    * Windows CMD: `venv\Scripts\activate`
  * Install requirements: `pip install -r requirements.txt`
* Make sure that both `active-pmsat-inference-wip` (i.e. the repository root) and `pmsat-inference` (the root of the embedded pmsat-inference repo) are on the PYTHONPATH.
  * In PyCharm, you can simply mark these two directories as "sources root", which will add them to the PYTHONPATH when you execute something via PyCharm's run/debug configuration.
  * In a terminal:
    * Automatically:
      * In the root directory of the repository (`active-pmsat-inference-wip`):
        * UNIX/Git Bash on Windows: run `source scripts/set_pythonpath.sh` 
        * Windows CMD: run `scripts/set_pythonpath`
    * Manually:
      * UNIX/Git Bash on Windows, run `export PYTHONPATH="./pmsat-inference;./;$PYTHONPATH"` in the root directory of the repository (`active-pmsat-inference-wip`)
      * Windows CMD: run `set PYTHONPATH=%PYTHONPATH%;./;./pmsat-inference`
      * Alternatively, in a python script which imports one of these packages, add the path to the parent directory of the package to `sys.path`

## Structure

### [active_pmsatlearn/](active_pmsatlearn)

Contains the active algorithm (in [active_pmsatlearn/learnalgo.py](active_pmsatlearn/learnalgo_mat.py))

### [evaluation/](evaluation)

Contains programs to evaluate the performance of the algorithm
* [generate_automata.py](evaluation/generate_automata.py): Generate automata via the command line or interactively.
  * You can choose either a range or a fixed number of states/inputs/outputs the generated automata should have, and how many automata *of each combination* should be created. See `python evaluation/generate_automata.py -h` for help
* [learn_automata.py](evaluation/learn_automata.py): Learn automata via the command line or interactively.
  * Specify algorithms to learn with by name
    * Supported: "KV", "KV (RS)", "L*", "L* (RS)", and the following combinations for the active Partial Max-SAT inference algorithm:
      * "ActivePMSL(\<el\>)"
      * "ActivePMSL(\<el\>)\_no\_\<processing_type\>"
        * Can contain multiple disabled processing types, i.e. "_no_proc1_no_proc2" - but must be in the correct order (currently) - see below!
      * "ActivePMSL(\<el\>)\_only\_\<processing_type\>"
      * ...where \<el\> is the extension length (supported currently: 2,3,4) and \<processing_type\> is a certain processing type (one of "input_completeness_processing", "cex_processing", "glitch_processing")
  * Specify oracles to learn with by name
    * Supported: "Perfect", "Random", "Random WMethod"
  * Specify how often to learn the same automaton with the same settings
  * Specify a maximum number of steps allowed in the SUL before aborting as failed learning attempt
  * Either specify a directory in which all .dot files will be learned, OR pass the same arguments as for [generate_automata.py](evaluation/generate_automata.py) to generate automata to fit your needs (and optionally re-use ones that were already generated)
  * Specify a results directory, in which results will be written. Each run of each algorithm currently writes one json file in this directory, immediately after the run finishes (enabling you to check out results while the program is still running).

### [scripts/](scripts)
* [run_apmsl_on_file_or_dir.py](scripts/run_apmsl_on_file_or_dir.py): Run the active PMSAT-inference algorithm on a given .dot file or on all .dot files in a given directory