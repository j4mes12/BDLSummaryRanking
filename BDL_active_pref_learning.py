# The structure and underlying framework of this code is strongly inspired by
# Simpson et al. [1], but much of this code is new.

import json
import argparse
import os
import warnings
import pandas as pd
import torch
from obtain_summary_text import SummaryTextGenerator

from summariser.oracle.lno_ref_values import SimulatedUser
from summariser.utils.corpus_reader import CorpusReader

from summariser.querier.expected_improvement_querier import (
    ExpImpQuerierForDeepLearner,
)
from summariser.querier.reward_learner import (
    TinyBertDeepLearnerWithMCDropoutInBert,
    TinyBertDeepLearnerWithMCDropoutInLayer,
)

from resources import PROCESSED_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.utils.evaluator import evaluateReward
import numpy as np
import logging

# from params import LEARNER_TYPE_DICT

logging.basicConfig(level=logging.DEBUG)


def process_cmd_line_args():
    print("processing args...")
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional arguments with default values
    parser.add_argument(
        "--dataset",
        default="DUC2001",
        type=str,
        choices=["DUC2001", "DUC2002", "DUC2004"],
    )
    parser.add_argument(
        "--learner_type_str",
        default="TBDL_IB",
        type=str,
        choices=["TBDL_IB", "TBDL_IL"],
    )
    parser.add_argument("--temp", default=1, type=float)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--n_samples", default=10, type=int)
    parser.add_argument(
        "--dropout_layers",
        default="both",
        type=str,
        choices=["both", "first", "second"],
    )
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--n_inter_rounds", type=int, default=10)
    parser.add_argument("--n_debug", type=int, default=0)
    parser.add_argument("--n_reps", type=int, default=1)
    parser.add_argument("--root_dir", type=str, default="./project_code")
    parser.add_argument("--res_dir", type=str, default="experiments/results")

    # Parse the arguments
    args = parser.parse_args()

    return (
        # Data variables
        args.dataset,
        # Model params
        args.learner_type_str,
        args.temp,
        args.dropout_rate,
        args.n_samples,
        args.dropout_layers,
        args.margin,
        # Experiment params
        args.n_inter_rounds,
        args.n_debug,
        args.n_reps,
        # Directories
        args.root_dir,
        args.res_dir,
    )


def load_json(filename):
    with open(filename, "r") as fh:
        return json.load(fh)


def save_json(filename, data):
    with open(filename, "w") as fh:
        json.dump(data, fh)


def get_torch_training_device():
    # Get the device for running the training and prediction
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Selecting device -- using cuda")
        print("Current cuda device: ", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    print("Selected device:", device)

    return device


def learn_dl_model(
    topic,  # d04a
    model,  # D04.M.100.B
    original_document,  # original document
    summaries,  # summary texts
    heuristic_list,  # list of initial values for summaries
    ref_values_dic,  # dict of rouge scores keys are models
    all_result_dic,
    output_path,
    n_iter_rounds,  # num loops
    n_debug,
    learner_type_str: str,
    n_samples: int,  # number of dropout samples
    temp=2.5,
    dropout_layers: str = "both",
    dropout_rate=0.1,
    margin=0.1,
):
    """
    This function learns a model based on the provided parameters.

    Parameters:
        topic (str): The topic of the model.
        model (list): The model to be learnt.
        ref_values_dic (dict): A dictionary containing reference values.
        heuristic_list (list): List of heuristics.
        n_inter_rounds (int): The number of rounds.
        all_result_dic (dict): Dictionary to store all results.
        n_debug (int): Debug level.
        output_path (str): The output path for saving results.
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

    reward_information = "_".join(
        [
            topic,
            model_name,
            "ExpImpForDL",
            learner_type_str,
            f"ns-{n_samples}",
            "margin-{}".format(margin),
            "temp-{}".format(temp),
        ]
    )

    # Add record of additional params for in-layer model
    if learner_type_str == "TBDL_IL":
        reward_information += "_" + "_".join(
            [f"dr-{dropout_rate}", f"dl-{dropout_layers}"]
        )

    reward_file = os.path.join(
        output_path, f"rewards_{reward_information}.json"
    )

    loss_file = os.path.join(
        output_path, f"model_losses_{reward_information}.json"
    )

    # if this has already been done, skip it!
    if os.path.exists(reward_file) and os.path.exists(loss_file):
        print("Reloading previously computed results.")
        # reload the pre-computed rewards
        learnt_rewards = load_json(reward_file)
        model_losses = load_json(loss_file)
    else:
        oracle = SimulatedUser(rouge_values, m=temp)

        querier = ExpImpQuerierForDeepLearner(
            heuristic_values=heuristic_list, full_cov=True, learnt_weight=0.5
        )

        if learner_type_str == "TBDL_IB":
            reward_learner = TinyBertDeepLearnerWithMCDropoutInBert(
                original_document=original_document,
                model_name="huawei-noah/TinyBERT_General_4L_312D",
                n_samples=n_samples,
            )
        elif learner_type_str == "TBDL_IL":
            reward_learner = TinyBertDeepLearnerWithMCDropoutInLayer(
                original_document=original_document,
                model_name="huawei-noah/TinyBERT_General_4L_312D",
                n_samples=n_samples,
                dropout_layers=dropout_layers,
                dropout_rate=dropout_rate,
            )

        log = []
        model_losses = []
        device = get_torch_training_device()
        candidate_summaries = summaries.copy()

        for round in range(n_iter_rounds):
            print(f"Starting round {round} of {n_iter_rounds}.")
            print("Getting DeepLearner similarity distributions.")
            reward_learner.get_scores_with_dropout(candidate_summaries)

            print("Identifying summaries to query user.")
            # Calculate distribution parameters
            f = reward_learner.get_similarity_rewards(return_tensor=False)
            candidate_idxs = querier._get_candidates(f)
            Cov = reward_learner.predictive_cov(
                candidate_idxs, full_cov=reward_learner.full_cov
            )
            # Limit scoring pool for later iterations to top 400
            if round == 0:
                candidate_summaries = np.array(candidate_summaries)[
                    candidate_idxs
                ].tolist()

            summ_idx1, summ_idx2 = querier.getQuery(
                f[candidate_idxs], Cov, candidate_idxs, log
            )
            summaries_to_log = (summ_idx1, summ_idx2)

            print("Simulating user preference...", end=" ")
            pref = oracle.getPref(summ_idx1, summ_idx2)
            print(f"summary {summaries_to_log[pref]} preferred")

            # log iteration
            log.append((summaries_to_log, pref))

            # train deep ranker
            print("Perform incremental train on DeepLearner.")
            iteration_loss = reward_learner.train_incremental(
                device=device,
                training_summaries=[
                    summaries[summ_idx1],
                    summaries[summ_idx2],
                ],
                preference_score=pref,
                loss_margin=margin,
            )

            model_losses.append(iteration_loss)

        print("Active learning complete. Now getting mixed rewards")
        # Score all summaries with most up-to-date weights
        reward_learner.get_scores_with_dropout(candidate_summaries)
        learnt_rewards = querier.getMixReward(
            learnt_values=reward_learner.get_model_rewards()
        )

        print("Saving the rewards for this model...")
        save_json(reward_file, learnt_rewards)
        print("Saving the losses for this model...")
        save_json(loss_file, model_losses)

    print("Computing metrics...")
    if n_debug:
        learnt_rewards = learnt_rewards[:n_debug]
    compute_and_store_metrics(learnt_rewards, rouge_values, all_result_dic)

    return learnt_rewards


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


def load_candidate_summaries(summaries, dataset, topic, root_dir, docs):
    feature_type_dir = "./data/summary_candidates/"
    summary_text_cache_file = os.path.join(
        root_dir,
        feature_type_dir,
        f"summary_texts_{dataset}_{topic}.csv",
    )

    if os.path.exists(summary_text_cache_file):
        warnings.warn(
            "reloading text for summaries from cache", ResourceWarning
        )
        summary_text = np.genfromtxt(
            summary_text_cache_file, delimiter="#####", dtype=str
        )
        return summary_text

    text_generator = SummaryTextGenerator(docs)

    summary_text = text_generator.getSummaryText(summaries)

    np.savetxt(
        summary_text_cache_file, summary_text, fmt="%s", delimiter="#####"
    )
    print(f"Cached summary vectors to {summary_text_cache_file}")

    return summary_text


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
    selected_means_allreps,
    selected_vars_allreps,
    chosen_metrics,
    method_names,
    this_method_idx,
    table_savename,
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
        selected_means_allreps[
            this_method_idx, metric_index
        ] += selected_means[this_method_idx, metric_index]
        selected_vars_allreps[this_method_idx, metric_index] += selected_vars[
            this_method_idx, metric_index
        ]

    filename = os.path.join(output_path, "{}.csv".format(table_savename))

    # Create DataFrame and save to CSV
    create_dataframe_and_save(
        method_names, selected_means, selected_vars, chosen_metrics, filename
    )


def make_output_dir(output_folder_name, rep):
    """
    Create an output directory based on provided root directory,
    results directory, output folder name, and repetition index.
    """

    output_path = os.path.join(
        root_dir, res_dir, output_folder_name, f"rep{rep}"
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path


def generate_summary(summary_index, summaries, docs):
    """
    This function generates the summary based on the learnt rewards.

    Parameters:
    index (numpy array): An array of actions for each summary line.
    summaries (list): A list containing sentence indices for each summary.
    docs (list): A list of documents where each document is a pair of an ID
     and a list of sentences.

    Returns:
    str: The summary text.
    """
    # Get the indices of the sentences for the summary
    summary_indices = summaries[summary_index]

    # Initialize the variables
    summary_sents = {}
    sentcount = 0

    # Collect the sentences for the summary
    for _, doc_sents in docs:
        for sent_text in doc_sents:
            if sentcount in summary_indices:
                summary_sents[sentcount] = sent_text
            sentcount += 1

    # Construct the summary text
    summary_text = " ".join(summary_sents[sent] for sent in summary_indices)

    return summary_text


def load_topic_articles(docs):
    topic_sentences = [
        sentence
        for file_path, article_sentences in docs
        for sentence in article_sentences
    ]

    return " ".join(topic_sentences)


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
        # Data variables
        dataset,
        # Model params
        learner_type_str,
        temp,
        dropout_rate,
        n_samples,
        dropout_layers,
        margin,
        # Experiment params
        n_inter_rounds,
        n_debug,
        n_reps,
        # Directories
        root_dir,
        res_dir,
    ) = process_cmd_line_args()

    output_folder_name = "_".join(
        [dataset.lower(), "ExpImpForDL", learner_type_str]
    )

    output_folder_path = os.path.join(root_dir, res_dir, output_folder_name)

    print(
        f"Running stage1 summary preference learning with {dataset}, "
        + f"writing to {output_folder_path}"
    )

    # set to greater than zero to use a subset of topics for debugging
    max_topics = -1
    folders = []
    rep = 0

    chosen_metrics = [
        "ndcg_at_1%",
        "pcc",
        "tau",
        "ndcg_at_5%",
        "ndcg_at_10%",
        "rho",
    ]
    selected_means_allreps = np.zeros((1, len(chosen_metrics)))
    selected_vars_allreps = np.zeros((1, len(chosen_metrics)))

    selected_means = np.zeros((1, len(chosen_metrics)))
    selected_vars = np.zeros((1, len(chosen_metrics)))

    output_path = make_output_dir(output_folder_path, rep)

    # saves a list of result folders containing repeats from the same run
    folders.append(output_path)
    with open(output_path + "/folders.txt", "w") as fh:
        for folder_name in folders:
            fh.write(folder_name + "\n")

    figs = avg_figs = []

    np.random.seed(1238549)

    print("loading dataset...")

    # read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset)

    # store all results
    all_result_dic = {}
    topic_cnt = 0

    querier_types = ["ExpImpForDL"]
    querier_type = querier_types[0]

    for topic, docs, models in data:
        print(
            f"\n=====(repeat {rep}) TOPIC {topic}, "
            + "QUERIER ExpImpForDL, "
            + f"INTER ROUND {n_inter_rounds}====="
        )

        topic_cnt += 1
        if 0 < max_topics < topic_cnt or (n_debug and topic_cnt > 1):
            continue

        (
            summaries,
            ref_values_dic,
            heuristic_list,
        ) = readSampleSummaries(dataset, topic, "supert")
        print(f"num of summaries read: {len(summaries)}")

        print("loading summary texts...")
        summary_text = load_candidate_summaries(
            summaries, dataset, topic, root_dir, docs
        )

        print("loading topic articles..")
        topic_documents = load_topic_articles(docs)

        if n_debug:
            heuristic_list = heuristic_list[:n_debug]
            summary_text = summary_text[:n_debug]

        print("framework started.")
        for model in models:
            learnt_rewards = learn_dl_model(
                # Data params
                topic=topic,
                model=model,
                original_document=topic_documents,
                summaries=summary_text.tolist(),
                heuristic_list=heuristic_list,
                ref_values_dic=ref_values_dic,
                all_result_dic=all_result_dic,
                # Path params
                output_path=output_path,
                # Experimental params
                n_iter_rounds=n_inter_rounds,
                n_debug=n_debug,
                # Model params
                learner_type_str=learner_type_str,
                n_samples=n_samples,
                temp=temp,
                dropout_layers=dropout_layers,
                dropout_rate=dropout_rate,
                margin=margin,
            )

            # The following selects the highest rewarded set of
            # sentences from a list of documents and prints them
            # out as a summary.
            print("SUMMARY: ")
            highest_rewarded_summary_text = generate_summary(
                summary_index=np.argmax(learnt_rewards),
                summaries=summaries,
                docs=docs,
            )
            print(highest_rewarded_summary_text)

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
            0,
            table_savename="table_of_metrics_"
            + "topic-{}_nsamples-{}_margin-{}_temp-{}_dr-{}_dl-{}".format(
                topic, n_samples, margin, temp, dropout_rate, dropout_layers
            ),
        )
