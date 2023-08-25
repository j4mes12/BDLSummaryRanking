
#  run the script for each DUC dataset
datasets = ("DUC2001" "DUC2002" "DUC2004")

for dataset in "${datasets[@]}"; do
    python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 

    # run the script for different margins in MRL func
    margins = (0 0.1 1 5)

    for m in "${margins[@]}"; do
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin "$m" --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin "$m" --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    done

    # run the script for different n_samples
    n_samples = (2 5 10 25 50)

    for n in "${n_samples[@]}"; do
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples "$n" --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples "$n" --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    done

    # run the script for different simulated user temperatures
    temps = (0.3 1 2.5)

    for t in "${temps[@]}"; do
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp "$t" --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IB --temp "$t" --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    done

    # InLayer-specific experiments

    # run the script for different dropout rates
    dropout_rates = (0.1 0.25 0.4)

    for rate in "${dropout_rates[@]}"; do
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate "$rate" --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    done

    # run the script for different dropout layer configurations
    layer_configs = ("both", "first", "second")

    for config in "${layer_configs[@]}"; do
        python -u stage1_active_pref_learning.py --dataset "$dataset" --use_lorem_summs False --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers "$config" --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
    done
done

# Test on lorem summaries
python -u stage1_active_pref_learning.py --dataset "DUC2001" --use_lorem_summs True --learner_type_str TBDL_IL --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
python -u stage1_active_pref_learning.py --dataset "DUC2001" --use_lorem_summs True --learner_type_str TBDL_IB --temp 1 --dropout_rate 0.1 --dropout_layers both --n_samples 10 --margin 0.1 --n_inter_rounds 100 --n_debug 0 --n_reps 1 --root_dir ./project_code --res_dir experiments/results 
