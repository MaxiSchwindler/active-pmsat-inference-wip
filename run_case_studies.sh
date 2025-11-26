#!/usr/bin/env bash

python3 -m evaluation.learn_automata -f case_studies/APC_pmsatLearned-rc2-N9.dot -a @THESIS_CASESTUDIES_NOMAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/nomat/apc/apmsl_05p;

python3 -m evaluation.learn_automata -f case_studies/CYBLE-416045-02_moore_without_mtu_req.dot -a @THESIS_CASESTUDIES_NOMAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/nomat/cyble_reduced/apmsl_05p;
python3 -m evaluation.learn_automata -f case_studies/CYBLE-416045-02_moore_without_mtu_req_and_feature_rsp.dot -a @THESIS_CASESTUDIES_NOMAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/nomat/cyble_reduced_mode/apmsl_05p;

python3 -m evaluation.learn_automata -f case_studies/SmokeMeter_pmsatLearned-rc2-N11.dot -a @THESIS_CASESTUDIES_NOMAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/nomat/smokemeter/apmsl_05p;


python3 -m evaluation.learn_automata -f case_studies/APC_pmsatLearned-rc2-N9.dot -a @THESIS_CASESTUDIES_MAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/mat/apc/apmsl_05p;

python3 -m evaluation.learn_automata -f case_studies/CYBLE-416045-02_moore_without_mtu_req.dot -a @THESIS_CASESTUDIES_MAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/mat/cyble_reduced/apmsl_05p;
python3 -m evaluation.learn_automata -f case_studies/CYBLE-416045-02_moore_without_mtu_req_and_feature_rsp.dot -a @THESIS_CASESTUDIES_MAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/mat/cyble_reduced_mode/apmsl_05p;

python3 -m evaluation.learn_automata -f case_studies/SmokeMeter_pmsatLearned-rc2-N11.dot -a @THESIS_CASESTUDIES_MAT_apmsl_variants_matrix.txt -o None --learn_num_times 10 --glitch_percent 0.5 --glitch_mode enter_random_state --num_processes 2 --results_dir THESIS_CASE_STUDIES/mat/smokemeter/apmsl_05p;