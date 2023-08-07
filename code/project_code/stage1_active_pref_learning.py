import json
import argparse
import os
from datetime import datetime
import pandas as pd
from obtain_supert_scores import SupertVectoriser
from summariser.oracle.lno_ref_values import SimulatedUser
from summariser.utils.corpus_reader import CorpusReader
from resources import PROCESSED_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.utils.evaluator import evaluateReward
import numpy as np
import logging
from random import seed
from params import QUERIER_TYPE_DICT, LEARNER_TYPE_DICT

logging.basicConfig(level=logging.DEBUG)


def process_cmd_line_args():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional arguments with default values
    parser.add_argument("--learner_type_str", default="BDL")
    parser.add_argument("--n_debug", type=int, default=0)
    parser.add_argument("--output_folder_name_in", default=-1)
    parser.add_argument(
        "--querier_types",
        type=lambda s: s.strip("[]").split(",") if s else None,
        default=["imp"],
    )
    parser.add_argument("--res_dir", default="results")
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--nthreads", type=int, default=0)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n_inter_rounds", type=int, default=None)
    parser.add_argument("--feature_type", default="april")
    parser.add_argument("--rate", type=float, default=200)
    parser.add_argument("--lspower", type=float, default=1)
    parser.add_argument("--temp", type=float, default=2.5)

    # Parse the arguments
    args = parser.parse_args()

    # Deal with arg inputs
    post_weight = 1
    first_rep, n_reps = 0, 1
    reps = np.arange(first_rep, n_reps)

    seed(28923895)
    np.random.seed(1238549)
    # generate a list of seeds that can be used
    # with all queriers in each repetition
    seeds = np.random.randint(1, 10000, n_reps)

    learner_type = LEARNER_TYPE_DICT.get(args.learner_type_str, None)

    return (
        learner_type,
        args.learner_type_str,
        args.n_inter_rounds,
        args.output_folder_name_in,
        args.querier_types,
        args.root_dir,
        args.res_dir,
        post_weight,
        reps,
        seeds,
        args.n_debug,
        args.nthreads,
        args.dataset,
        args.feature_type,
        args.rate,
        args.lspower,
        args.temp,
    )


def load_json(filename):
    with open(filename, "r") as fh:
        return json.load(fh)


def save_json(filename, data):
    with open(filename, "w") as fh:
        json.dump(data, fh)


def learn_model(
    topic,
    model,
    ref_values_dic,
    querier_type,
    learner_type,
    learner_type_str,
    summary_vectors,
    heuristics_list,
    post_weight,
    n_inter_rounds,
    all_result_dic,
    n_debug,
    output_path,
    n_threads,
    temp=2.5,
):
    """
    This function learns a model based on the provided parameters.

    Parameters:
        topic (str): The topic of the model.
        model (list): The model to be learnt.
        ref_values_dic (dict): A dictionary containing reference values.
        querier_type (str): The type of the querier.
        learner_type (str): The type of the learner.
        learner_type_str (str): The string representation of the learner type.
        summary_vectors (ndarray): The summary vectors.
        heuristics_list (list): List of heuristics.
        post_weight (float): The weight of the post.
        n_inter_rounds (int): The number of rounds.
        all_result_dic (dict): Dictionary to store all results.
        n_debug (int): Debug level.
        output_path (str): The output path for saving results.
        n_threads (int): The number of threads to be used in processing.
        temp (float, optional): The temperature for the SimulatedUser,
         defaults to 2.5.

    Returns:
        learnt_rewards (dict): The dictionary of learnt rewards.
    """

    model_name = model[0].split("/")[-1].strip()
    print(f"\n---ref. summary {model_name}---")

    rouge_values = ref_values_dic[model_name]
    if n_debug:
        rouge_values = rouge_values[:n_debug]

    learner_type_label = (
        learner_type.__name__ if learner_type is not None else "nolearner"
    )

    reward_information = "_".join(
        [topic, model_name, querier_type, learner_type_label]
    )
    reward_file = os.path.join(
        output_path, f"rewards_{reward_information}.json"
    )

    # if this has already been done, skip it!
    if os.path.exists(reward_file):
        print("Reloading previously computed results.")
        # reload the pre-computed rewards
        learnt_rewards = load_json(reward_file)
    else:
        oracle = SimulatedUser(rouge_values, m=temp)

        querier = QUERIER_TYPE_DICT.get(
            querier_type, QUERIER_TYPE_DICT["default"]
        )(
            learner_type,
            summary_vectors,
            heuristics_list,
            post_weight,
            rate,
            lspower,
        )

        log = []
        perform_learning(querier, oracle, log, n_inter_rounds, querier_type)

        print("Active learning complete. Now getting mixed rewards")
        learnt_rewards = querier.getMixReward()

        print("Saving the rewards for this model...")
        save_json(reward_file, learnt_rewards)

    print("Computing metrics...")
    compute_and_store_metrics(learnt_rewards, rouge_values, all_result_dic)

    return learnt_rewards


def perform_learning(querier, oracle, log, n_inter_rounds, querier_type):
    """
    This function performs the learning process for a given querier and oracle.

    Parameters:
        querier (object): The querier object used in the learning process.
        oracle (object): The oracle object simulating user feedback.
        log (list): A list to store query and preference history.
        n_inter_rounds (int): The number of intermediate rounds.
        querier_type (str): The type of the querier, affecting the training
         strategy.
    """
    for round in range(n_inter_rounds):
        sum1, sum2 = querier.getQuery(log)
        pref = oracle.getPref(sum1, sum2)
        log.append([[sum1, sum2], pref])
        if querier_type != "random" or round == n_inter_rounds - 1:
            # with random querier, don't train until the last
            # iteration as the intermediate results are not used
            querier.updateRanker(log)


def compute_and_store_metrics(learnt_rewards, rouge_values, all_result_dic):
    """
    This function computes and stores metrics from the learnt rewards.

    Parameters:
        learnt_rewards (dict): The dictionary of learnt rewards.
        rouge_values (array): An array of ROUGE values used for reward
         evaluation.
        all_result_dic (dict): A dictionary to store all metrics results.

    """
    metrics_dic = evaluateReward(learnt_rewards, rouge_values)

    for metric_name, metric_value in metrics_dic.items():
        print(f"metric {metric_name} : {metric_value}")
        if metric_name in all_result_dic:
            all_result_dic[metric_name].append(metric_value)
        else:
            all_result_dic[metric_name] = [metric_value]


def load_summary_vectors(
    summaries, dataset, topic, root_dir, docs, feature_type
):
    """
    Load the summary vectors based on the given feature type.
    """
    feature_type_dir = "./data/summary_vectors/"
    summary_vecs_cache_file = os.path.join(
        root_dir,
        feature_type_dir,
        feature_type,
        f"summary_vectors_{dataset}_{topic}.csv",
    )

    feature_type_path_check = os.path.join(root_dir, feature_type_dir)
    if not os.path.exists(feature_type_path_check):
        os.mkdir(feature_type_path_check)

    if os.path.exists(summary_vecs_cache_file):
        print("Warning: reloading feature vectors for summaries from cache")
        summary_vectors = np.genfromtxt(summary_vecs_cache_file)
        return summary_vectors

    vectorisers = {
        "april": Vectoriser,
        "supertbigram+": Vectoriser,
        "supert": SupertVectoriser,
    }

    if feature_type not in vectorisers:
        raise ValueError(f"Invalid feature type: {feature_type}")

    vec = vectorisers[feature_type](docs)

    if feature_type == "supert":
        summary_vectors, _ = vec.getSummaryVectors(
            summaries, use_coverage_feats=True
        )
    else:
        summary_vectors = vec.getSummaryVectors(summaries)

    np.savetxt(summary_vecs_cache_file, summary_vectors)
    print(f"Cached summary vectors to {summary_vecs_cache_file}")

    return summary_vectors


def save_result_dic(
    all_result_dic,
    output_path,
    rep,
    topic_cnt,
    querier_type,
    learner_type_str,
    n_inter_rounds,
):
    """
    Compute and save metrics for a given topic.
    """
    print(
        f"=== (rep={rep}) RESULTS UNTIL TOPIC {topic_cnt}, "
        f"QUERIER {querier_type.upper()}, LEARNER {learner_type_str}, "
        f"INTER ROUND {n_inter_rounds} ===\n"
    )
    for metric, values in all_result_dic.items():
        print(f"{metric} : {np.mean(values)}")

    file_name = (
        f"metrics_{querier_type}_{learner_type_str}_{n_inter_rounds}.json"
    )
    with open(os.path.join(output_path, file_name), "w") as fh:
        json.dump(all_result_dic, fh)


def create_dataframe_and_save(
    method_names, data_means, data_vars, metrics, filename
):
    """
    This function creates a pandas DataFrame using provided means and
    variances, and saves the DataFrame into a CSV file.
    """

    # Prepare data for DataFrame
    df_data = np.concatenate(
        (
            np.array(method_names)[:, None],
            data_means,
            data_vars,
        ),
        axis=1,
    )

    # Prepare column names
    column_names = np.concatenate(
        (
            ["Method"],
            metrics,
            [f"{metric} var" for metric in metrics],
        )
    )

    # Create DataFrame
    df = pd.DataFrame(df_data, columns=column_names).set_index("Method")

    print(f"Saving data to {filename}")

    # Save DataFrame to CSV
    df.to_csv(filename)


def save_selected_results(
    output_path,
    all_result_dic,
    selected_means,
    selected_vars,
    chosen_metrics,
    method_names,
    this_method_idx,
):
    """
    This function calculates means and variances for selected metrics and
    saves them into a csv file along with method names.
    """

    # Compute means and variances
    for metric_index, metric in enumerate(chosen_metrics):
        selected_means[this_method_idx, metric_index] = np.mean(
            all_result_dic[metric]
        )
        selected_vars[this_method_idx, metric_index] = np.var(
            all_result_dic[metric]
        )

    filename = os.path.join(output_path, "table.csv")

    # Create DataFrame and save to CSV
    create_dataframe_and_save(
        method_names, selected_means, selected_vars, chosen_metrics, filename
    )


def save_selected_results_allreps(
    output_path,
    selected_means_allreps,
    selected_vars_allreps,
    chosen_metrics,
    method_names,
    nreps,
):
    """
    This function calculates average means and variances over all repetitions
    for selected metrics and saves them into a csv file along with method
    names.
    """

    # Compute average means and variances
    average_means_allreps = selected_means_allreps / float(nreps)
    average_vars_allreps = selected_vars_allreps / float(nreps)

    filename = os.path.join(output_path, "table_all_reps.csv")

    # Create DataFrame and save to CSV
    create_dataframe_and_save(
        method_names,
        average_means_allreps,
        average_vars_allreps,
        chosen_metrics,
        filename,
    )


def make_output_dir(root_dir, res_dir, output_folder_name, rep):
    """
    Create an output directory based on provided root directory,
    results directory, output folder name, and repetition index.
    """
    if output_folder_name == -1:
        output_folder_name = datetime.now().strftime(
            r"started-%Y-%m-%d-%H-%M-%S"
        )
    else:
        output_folder_name = f"{output_folder_name}_rep{rep}"

    output_path = os.path.join(root_dir, res_dir, output_folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path


def generate_summary(learnt_rewards, summaries, docs):
    """
    This function generates the best summary based on the learnt rewards.

    Parameters:
    learnt_rewards (numpy array): An array of rewards for each summary.
    summaries (list): A list containing sentence indices for each summary.
    docs (list): A list of documents where each document is a pair of an ID
     and a list of sentences.

    Returns:
    str: The best summary text.
    """
    # Get the indices of the sentences for the best summary
    best_summary_indices = summaries[np.argmax(learnt_rewards)]

    # Initialize the variables
    summary_sents = {}
    sentcount = 0

    # Collect the sentences for the summary
    for _, doc_sents in docs:
        for sent_text in doc_sents:
            if sentcount in best_summary_indices:
                summary_sents[sentcount] = sent_text
            sentcount += 1

    # Construct the summary text
    summary_text = "".join(
        summary_sents[sent] for sent in best_summary_indices
    )

    return summary_text


if __name__ == "__main__":
    """
    Command line arguments:
    python stage1_active_pref_learning.py reward_learner_type n_debug
    output_folder_name querier_types

    reward_learner_type -- can be:
        BDL - copy of GPPL (placeholder);
        MC_BDL - supert ranker with mc dropout;
        SWAG_BDL - supert ranker using swag;
        nBDL - super ranking without bayesian inference.

    n_debug -- set to 0 if you are not debugging; set to a higher number to
    select a subsample of the data for faster debugging of the main setup.

    output_folder_name -- this name will be used to store your results
    (metrics) and the rewards produced by the learner.
    This will be a subfolder of ./results/ .

    querier_types -- a list of querier types. If the reward learner is LR,
    you can pass any subset of [random, unc]. If the reward learner is any
    of the GPPL variants you can pass [random, pair_unc, pair_unc_SO, tp, imp].
    The best performers are tig and imp.

    """

    (
        learner_type,
        learner_type_str,
        n_inter_rounds,
        output_folder_name,
        querier_types,
        root_dir,
        res_dir,
        post_weight,
        reps,
        seeds,
        n_debug,
        n_threads,
        dataset,
        feature_type,
        rate,
        lspower,
        temp,
    ) = process_cmd_line_args()

    # parameters
    if dataset is None:
        dataset = "DUC2001"  # 'DUC2001'  # DUC2001, DUC2002, 'DUC2004'#

    print(
        f"Running stage1 summary preference learning with {dataset}, "
        + f"writing to {root_dir}/{res_dir}/{output_folder_name}"
    )

    # set to greater than zero to use a subset of topics for debugging
    max_topics = -1
    folders = []

    nqueriers = len(querier_types)

    chosen_metrics = [
        "ndcg_at_1%",
        "pcc",
        "tau",
        "ndcg_at_5%",
        "ndcg_at_10%",
        "rho",
    ]

    selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
    selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

    for rep in reps:
        selected_means = np.zeros((nqueriers, len(chosen_metrics)))
        selected_vars = np.zeros((nqueriers, len(chosen_metrics)))

        output_path = make_output_dir(
            root_dir, res_dir, output_folder_name, rep
        )

        # saves a list of result folders containing repeats from the same run
        folders.append(output_path)
        with open(output_path + "/folders.txt", "w") as fh:
            for folder_name in folders:
                fh.write(folder_name + "\n")

        figs = avg_figs = []

        for qidx, querier_type in enumerate(querier_types):
            seed(seeds[rep])
            np.random.seed(seeds[rep])

            # read documents and ref. summaries
            reader = CorpusReader(PROCESSED_PATH)
            data = reader.get_data(dataset)

            # store all results
            all_result_dic = {}
            topic_cnt = 0

            for topic, docs, models in data:
                print(
                    f"\n=====(repeat {rep}) TOPIC {topic}, "
                    + f"QUERIER {querier_type.upper()}, "
                    + f"INTER ROUND {n_inter_rounds}====="
                )

                topic_cnt += 1
                if 0 < max_topics < topic_cnt or (n_debug and topic_cnt > 1):
                    continue

                (
                    summaries,
                    ref_values_dic,
                    heuristic_list,
                ) = readSampleSummaries(dataset, topic, feature_type)
                print(f"num of summaries read: {summaries}")

                summary_vectors = load_summary_vectors(
                    summaries, dataset, topic, root_dir, docs, feature_type
                )

                if n_debug:
                    heuristic_list = heuristic_list[:n_debug]
                    summary_vectors = summary_vectors[:n_debug]

                for model in models:
                    learnt_rewards = learn_model(
                        topic,
                        model,
                        ref_values_dic,
                        querier_type,
                        learner_type,
                        learner_type_str,
                        summary_vectors,
                        heuristic_list,
                        post_weight,
                        n_inter_rounds,
                        all_result_dic,
                        n_debug,
                        output_path,
                        n_threads,
                        temp,
                    )

                    # The following selects the highest rewarded set of
                    # sentences from a list of documents and prints them
                    # out as a summary.
                    print("SUMMARY: ")
                    highest_rewarded_summary_text = generate_summary(
                        learnt_rewards=learnt_rewards,
                        summaries=summaries,
                        docs=docs,
                    )
                    print(highest_rewarded_summary_text)

                if n_debug:
                    heuristic_list = heuristic_list[:n_debug]
                    summary_vectors = summary_vectors[:n_debug]

                for model in models:
                    learn_model(
                        topic,
                        model,
                        ref_values_dic,
                        querier_type,
                        learner_type,
                        learner_type_str,
                        summary_vectors,
                        heuristic_list,
                        post_weight,
                        n_inter_rounds,
                        all_result_dic,
                        n_debug,
                        output_path,
                        n_threads,
                        temp=temp,
                    )

                save_result_dic(
                    all_result_dic,
                    output_path,
                    rep,
                    topic_cnt,
                    querier_type,
                    learner_type_str,
                    n_inter_rounds,
                )

            save_selected_results(
                output_path,
                all_result_dic,
                selected_means,
                selected_vars,
                selected_means_allreps,
                selected_vars_allreps,
                chosen_metrics,
                querier_types,
                qidx,
            )

    save_selected_results_allreps(
        output_path,
        selected_means_allreps,
        selected_vars_allreps,
        chosen_metrics,
        querier_types,
        len(reps),
    )
