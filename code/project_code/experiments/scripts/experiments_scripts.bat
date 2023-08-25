@echo off

:: run the script for each DUC dataset
for %%D in ("DUC2001" "DUC2002" "DUC2004") do (
    set dataset=%%~D
    python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results

    :: run the script for different margins in MRL func
    for %%M in (0 0.1 1 5) do (
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin %%M --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin %%M --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    )

    :: run the script for different n_samples
    for %%N in (2 5 10 25 50) do (
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples %%N --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples %%N --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    )

    :: run the script for different simulated user temperatures
    for %%T in (0.3 1 2.5) do (
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp %%T --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IB --temp %%T --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    )

    :: run the script for different dropout rates
    for %%R in (0.1 0.25 0.4) do (
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate %%R --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    )

    :: run the script for different dropout layer configurations
    for %%C in ("both" "first" "second") do (
        python -u stage1_active_pref_learning.py --dataset "%dataset%" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers %%~C --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
    )
)

:: Test on lorem summaries
python -u stage1_active_pref_learning.py --dataset "DUC2001" --use_lorem_summs True --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
python -u stage1_active_pref_learning.py --dataset "DUC2001" --use_lorem_summs True --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results
