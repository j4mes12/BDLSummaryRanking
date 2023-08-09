import numpy as np

# Custom version of package
from custom_sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import copy
import sys

sys.path.append("../")

from resources import LANGUAGE  # noqa
from ref_free_metrics.similarity_scorer import parse_documents  # noqa


def kill_stopwords(sent_idx, all_token_vecs, all_tokens, language="english"):
    # Initialize the full vectors and tokens with the first sentence's
    # vectors and tokens
    full_vec = copy.deepcopy(all_token_vecs[sent_idx[0]])
    full_token = copy.deepcopy(all_tokens[sent_idx[0]])

    # Extend the full vectors and tokens with those of the rest sentences
    for si in sent_idx[1:]:
        full_vec = np.row_stack((full_vec, all_token_vecs[si]))
        full_token.extend(all_tokens[si])

    # Get the list of stop words in the given language and add additional ones
    mystopwords = set(stopwords.words(language))
    mystopwords.update(["[cls]", "[sep]"])

    # Filter out the stop words from the full tokens and their vectors
    wanted_idx = [
        j for j, tk in enumerate(full_token) if tk.lower() not in mystopwords
    ]
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]


def encode_token_vecs(model, sents):
    token_embeddings, tokens = model.encode(sents, token_vecs=True)

    assert len(token_embeddings) == len(tokens), (
        "Overall length assertion - "
        + f"{len(token_embeddings)} != {len(tokens)}"
    )

    for idx in range(len(token_embeddings)):
        assert len(token_embeddings[idx]) == len(tokens[idx]), (
            "Index length assertion - "
            + f"{len(token_embeddings[idx])} != {len(tokens[idx])}"
        )

    return token_embeddings, tokens


def get_all_token_vecs(model, sent_info_dict):
    sents = [sent_info_dict[i]["text"] for i in sent_info_dict]
    token_embeddings, tokens = encode_token_vecs(model, sents)

    assert len(token_embeddings) == len(
        tokens
    ), f"{len(token_embeddings)} != {len(tokens)}"
    for i in range(len(token_embeddings)):
        assert len(token_embeddings[i]) == len(
            tokens[i]
        ), f"{len(token_embeddings[i])} != {len(tokens[i])}"
    return token_embeddings, tokens


def compute_similarity_metrics(rvecs, svecs):
    """
    Function to compute recall, precision, and f1 score based on
    cosine similarity between reference vectors (rvecs) and summary
    vectors (svecs).
    """
    # Compute cosine similarity between reference and summary vectors
    sim_matrix = cosine_similarity(rvecs, svecs)
    # Calculate recall, precision, and F1 score
    recall = np.mean(np.max(sim_matrix, axis=1))
    precision = np.mean(np.max(sim_matrix, axis=0))
    f1 = 2.0 * recall * precision / (recall + precision)
    return recall, precision, f1


def get_scores(empty_summs_ids, metric_list, summ_token_vecs):
    """
    Function to compute the mean scores for each summary vector,
     based on the chosen sim_metric
    """
    scores = [
        None if i in empty_summs_ids else np.mean(metric_list[:, i])
        for i in range(len(summ_token_vecs))
    ]
    return scores


def get_sbert_score(ref_token_vecs, summ_token_vecs, sim_metric):
    # Initialisation of necessary lists
    recall_list, precision_list, f1_list, empty_summs_ids = [], [], [], []

    # Iterating over all reference vectors
    for rvecs in ref_token_vecs:
        r_recall_list, r_precision_list, r_f1_list = [], [], []

        # Iterating over all summary vectors
        for idx, svecs in enumerate(summ_token_vecs):
            # If svecs is None, we add None to metric lists and continue
            if svecs is None:
                empty_summs_ids.append(idx)
                r_recall_list.append(None)
                r_precision_list.append(None)
                r_f1_list.append(None)
                continue

            recall, precision, f1 = compute_similarity_metrics(rvecs, svecs)
            r_recall_list.append(recall)
            r_precision_list.append(precision)
            r_f1_list.append(f1)

        # Append the respective scores for each reference vector
        recall_list.append(r_recall_list)
        precision_list.append(r_precision_list)
        f1_list.append(r_f1_list)

    # Remove duplicates
    empty_summs_ids = list(set(empty_summs_ids))

    # Convert lists to numpy arrays
    recall_list, precision_list, f1_list = (
        np.array(recall_list),
        np.array(precision_list),
        np.array(f1_list),
    )

    # Based on the requested metric, calculate and return the scores
    if "recall" in sim_metric:
        return get_scores(empty_summs_ids, recall_list, summ_token_vecs)
    elif "precision" in sim_metric:
        return get_scores(empty_summs_ids, precision_list, summ_token_vecs)
    else:
        assert "f1" in sim_metric
        return get_scores(empty_summs_ids, f1_list, summ_token_vecs)


def get_reference_indices(ref_dic, ref_sources):
    """
    Function to create reference indices based on certain conditions.
    """
    ref_idxs = []
    if len(ref_dic) >= 15:
        for rs in ref_sources:
            ref_idxs.append([k for k in ref_dic if ref_dic[k]["doc"] == rs])
    else:
        ref_idxs.append([k for k in ref_dic])
    return ref_idxs


def get_vecs_tokens(index_list, all_token_vecs, all_tokens):
    """
    Function to get vectors and tokens based on an index list.
    """
    ksw_data = [
        kill_stopwords(ref, all_token_vecs, all_tokens) for ref in index_list
    ]
    vecs, tokens = zip(*ksw_data)
    return vecs, tokens


def get_scoring_artifacts(docs, metric):
    # Get sentence transformer model
    model = SentenceTransformer("bert-large-nli-stsb-mean-tokens")

    # Word and sentence tokenization
    sent_info_dic, _, sents_weights = parse_documents(docs, None, metric)
    all_token_vecs, all_tokens = get_all_token_vecs(model, sent_info_dic)

    # Build pseudo-reference
    ref_dic = {
        k: sent_info_dic[k] for k in sent_info_dic if sents_weights[k] >= 0.1
    }

    # Get sentences in the pseudo reference
    ref_sources = set(ref_dic[k]["doc"] for k in ref_dic)
    ref_idxs = get_reference_indices(ref_dic, ref_sources)

    return (model, ref_idxs, all_token_vecs, all_tokens)


def get_rewards(docs, summaries, ref_metric, sim_metric="f1"):
    model, ref_idxs, all_token_vecs, all_tokens = get_scoring_artifacts(
        docs, ref_metric
    )

    # Get vectors and tokens of the pseudo reference and the summaries
    ref_vecs, ref_tokens = get_vecs_tokens(
        ref_idxs, all_token_vecs, all_tokens
    )
    summ_vecs, summ_tokens = get_vecs_tokens(
        summaries, all_token_vecs, all_tokens
    )

    # Measure similarity between system summaries and the pseudo-reference
    scores = get_sbert_score(ref_vecs, summ_vecs, sim_metric)

    return scores


def get_token_vecs(model, sents, remove_stopwords=True, language="english"):
    if not sents:
        return None, None

    vecs, tokens = encode_token_vecs(model, sents)

    # Concatenate the vectors and tokens
    full_vec = np.concatenate(vecs, axis=0)
    full_token = [token for sublist in tokens for token in sublist]

    # Define wanted indices based on whether to remove stopwords or not
    if remove_stopwords:
        mystopwords = set(stopwords.words(language))
        mystopwords.update(["[cls]", "[sep]"])
        wanted_idx = [
            j
            for j, tk in enumerate(full_token)
            if tk.lower() not in mystopwords
        ]
    else:
        wanted_idx = list(range(len(full_token)))

    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]


def get_sbert_score_metrics(
    docs, summaries, ref_metric, sim_metric="f1", return_summary_vectors=False
):
    model, ref_idxs, all_token_vecs, all_tokens = get_scoring_artifacts(
        docs, ref_metric
    )

    # Get vecs and tokens of the pseudo reference
    ref_vecs, ref_tokens = get_vecs_tokens(
        ref_idxs, all_token_vecs, all_tokens
    )

    # Get vecs and tokens of the summaries
    summ_data = [
        get_token_vecs(model, sent_tokenize(summ)) for summ in summaries
    ]
    summ_vecs, summ_tokens = zip(*summ_data)

    # Measure similarity between system summaries and the pseudo-ref
    scores = get_sbert_score(ref_vecs, summ_vecs, sim_metric)

    return (scores, summ_vecs) if return_summary_vectors else scores
