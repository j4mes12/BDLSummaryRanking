"""
Wrapper for Gaussian process preference learning (GPPL) to learn the latent
reward function from pairwise preferencelabels expressed by a noisy labeler.
"""
import logging
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset
from torch import nn

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# legacy imports
from sklearn.decomposition import PCA

from gppl.gp_classifier_vb import compute_median_lengthscales
from gppl.gp_pref_learning import GPPrefLearning


def do_PPA(new_items_feat, ndims):
    # PPA - subtract mean
    new_items_feat = new_items_feat - np.mean(new_items_feat)
    # PPA - compute PCA components
    pca = PCA(ndims)
    pca.fit_transform(new_items_feat)
    U1 = pca.components_

    # Remove top-d components
    for row, v in enumerate(new_items_feat):
        for u in U1[0:7]:
            new_items_feat[row] -= u.T.dot(v[:, None]) * u

    return new_items_feat


def reduce_dimensionality(new_items_feat):
    # reduce a large feature vector using
    #  the method of https://www.aclweb.org/anthology/W19-4328.pdf
    # because this worked well for reaper... could be optimised more
    ndims = 150

    if new_items_feat.shape[0] < ndims:
        ndims = int(new_items_feat.shape[0] / 2)

    new_items_feat = do_PPA(new_items_feat, ndims * 2)

    new_items_feat = PCA(ndims).fit_transform(new_items_feat)

    new_items_feat = do_PPA(new_items_feat, ndims)

    return new_items_feat


class BERTRewardLearner:
    def __init__(
        self,
        steep=1.0,
        full_cov=False,
        n_threads=0,
        rate=200,
        lspower=1,
        do_dim_reduction=False,
    ):
        self.learner = None
        self.steep = steep

        self.n_labels_seen = 0

        self.n_iterations = -1

        # whether to compute the full posterior covariance
        self.full_cov = full_cov

        self.tune = False
        self.fixed_s = True

        self.mu0 = None

        self.n_threads = n_threads

        self.default_rate = rate
        self.lspower = lspower

        self.items_feat = None
        self.do_dim_reduction = do_dim_reduction

    def train(self, pref_history, vector_list, true_scores=None, tr_items=None):
        """
        :param pref_history: a list of objects of the form
         [[item_id0, item_id1], preference_label ], where preference_label is
         0 to indicate that item 0 is preferred and 1 to indicate item 1.
        :param vector_list: list of feature vectors for the items
        :return: nowt.
        """

        # self.n_iterations += 1
        # # only update the model once every 5 iterations
        # if np.mod(self.n_iterations, 5) != 0:
        #     return

        if self.tune:
            rates = [800, 1600, 3200, 6400, 12800]
        else:
            rates = [self.default_rate]
            # rates = [200]  # used in initial submission
            # rates = [100]
            # rates = [10] # lstest4

        best_train_score = -np.inf
        best_rate = 200

        for rate in rates:
            if self.learner is None or self.tune:  # needs the vectors to init
                new_items_feat = np.array(vector_list)
                self.items_feat = new_items_feat

                logging.debug(
                    "Estimating lengthscales for %i features from %i items"
                    % (new_items_feat.shape[1], new_items_feat.shape[0])
                )

                if self.do_dim_reduction and new_items_feat.shape[1] < 300:
                    self.do_dim_reduction = False
                    logging.info(
                        "Switching off dimensionality reduction as "
                        + "we already have fewer than 300 dimensions."
                    )

                if self.do_dim_reduction:
                    new_items_feat = reduce_dimensionality(new_items_feat)

                ls_initial = compute_median_lengthscales(
                    new_items_feat, multiply_heuristic_power=self.lspower
                )
                # Tested with random selection, a value of
                #  multiply_heuristic_power=1 is better than 0.5 by far on the
                #  April/Reaper and COALA setups.
                # Original submission (REAPER) uses 1.0
                # Earlier results_noisy (supert) uses 1.0
                # results_lstest use 0.5
                # results_lstest2 uses 0.75 -- this was bad
                # consider increasing noise (decreasing rate_s0 to reduce the
                # scale of the function)
                # lstest3 uses 0.5 with s0_rate 100
                # lstest 4 uses 0.25 with s0_Rate 10
                # lstest 5 uses 2 with rate 200
                # lstest 6 uses 2 with rate 20

                logging.debug("Estimated length scales.")

                self.learner = GPPrefLearning(
                    ninput_features=len(vector_list[0]),
                    shape_s0=1.0,
                    rate_s0=rate,
                    use_svi=True,
                    ninducing=500,
                    max_update_size=1000,
                    kernel_combination="*",
                    forgetting_rate=0.7,
                    delay=1,
                    fixed_s=self.fixed_s,
                    verbose=True,
                    ls_initial=ls_initial,
                )

                # self.learner.set_max_threads(self.n_threads)

                logging.debug("Initialised GPPL.")

                # put these settings in to reduce the number of iterations
                # required before declaring convergence
                self.learner.min_iter_VB = 1
                self.learner.conv_check_freq = 1
                self.learner.n_converged = 1
            else:
                # only pass in the feature vectors in the first iteration
                new_items_feat = None

            # use the heuristic mean only for debugging
            new_item_ids0 = [data_point[0][0] for data_point in pref_history]
            new_item_ids1 = [data_point[0][1] for data_point in pref_history]
            new_labels = np.array(
                [1 - data_point[1] for data_point in pref_history]
            )  # for GPPL, item 2 is preferred if label == 0

            # new_item_ids0 = []
            # new_item_ids1 = []
            # new_labels = []

            logging.debug("GPPL fitting with %i pairwise labels" % len(new_labels))

            self.learner.fit(
                new_item_ids0,
                new_item_ids1,
                new_items_feat,
                new_labels,
                optimize=False,
                input_type="binary",
                use_median_ls=False,
                mu0=self.mu0[:, None] if self.mu0 is not None else None,
            )

            if self.tune:
                # Can't really use Pearson in a realistic setting because
                # we don't have the oracle
                # train_score = pearsonr(
                #     self.learner.f[tr_items].flatten(), true_scores
                # )[0]
                # print("Training Pearson r = %f" % train_score)

                train_score = self.learner.lowerbound()
                print("ELBO = %.5f" % train_score)

                if train_score > best_train_score:
                    best_train_score = train_score
                    best_model = self.learner
                    best_rate = rate
                    print(
                        "New best train score %f with rate_s0=%f" % (train_score, rate)
                    )

        print("GPPL fitting complete in %i iterations." % self.learner.vb_iter)

        if self.tune:
            self.learner = best_model
            print("Best tuned model has rate_s=%f" % best_rate)
            with open("./results/tuning_results.csv", "a") as fh:
                fh.writelines(["%i" % best_rate])
        if not self.fixed_s:
            print("Learned model has s=%f" % self.learner.s)
            with open("./results/learning_s_results.csv", "a") as fh:
                fh.writelines(["%f" % self.learner.s])

        self.n_labels_seen = len(pref_history)

        self.rewards, self.reward_var = self.learner.predict_f(
            full_cov=False,
            reuse_output_kernel=True,
            mu0_output=self.mu0[:, None] if self.mu0 is not None else None,
        )
        logging.debug("...rewards obtained.")

    def get_rewards(self):
        return self.rewards.flatten()

    def predictive_var(self):
        return self.reward_var.flatten()

    def predictive_cov(self, idxs):
        if self.full_cov:
            mu0_output = (
                self.mu0[idxs, None]
                if self.mu0 is not None and not np.isscalar(self.mu0)
                else self.mu0
            )
            return self.learner.predict_f(
                out_idxs=idxs,
                full_cov=True,
                mu0_output=mu0_output,
            )[1]
        else:
            if self.reward_var.shape[1] == 1:
                return self.reward_var[idxs]
            else:
                return np.diag(self.reward_var[idxs])


class BERTHRewardLearner(BERTRewardLearner):
    def __init__(
        self,
        steep=1.0,
        full_cov=False,
        heuristics=None,
        n_threads=0,
        heuristic_offset=0.0,
        heuristic_scale=1.0,
        rate=200,
        lspower=1,
    ):
        super(BERTHRewardLearner, self).__init__(
            steep, full_cov, n_threads=n_threads, rate=200, lspower=lspower
        )

        minh = np.min(heuristics)
        maxh = np.max(heuristics)

        self.mu0 = (heuristics - minh) / (maxh - minh) - 0.5  # * 2 * np.sqrt(200)
        self.mu0 = self.mu0 * heuristic_scale + heuristic_offset


class BaseBERTDeepRanker(nn.Module):
    def __init__(
        self, sbert_model_name: str = "bert-large-nli-stsb-mean-tokens"
    ) -> None:
        super().__init__()
        self.sbert_model = SentenceTransformer(model_name_or_path=sbert_model_name)
        cs = nn.CosineSimilarity(dim=1)
        self.cs = nn.DataParallel(cs)

    def forward(self, doc, summ1, summ2):
        self.doc_embedding = self.sbert_model.encode(
            doc, convert_to_tensor=True
        ).unsqueeze(0)
        self.summary1_embedding = self.sbert_model.encode(
            summ1, convert_to_tensor=True
        ).unsqueeze(0)
        self.summary2_embedding = self.sbert_model.encode(
            summ2, convert_to_tensor=True
        ).unsqueeze(0)

        score1 = self.cs(self.doc_embedding, self.summary1_embedding)
        score2 = self.cs(self.doc_embedding, self.summary2_embedding)

        return score1, score2


class MCDBERTDeepRanker(BaseBERTDeepRanker):
    def __init__(
        self,
        sbert_model_name: str = "bert-large-nli-stsb-mean-tokens",
        n_samples: int = 100,
    ) -> None:
        self.n_samples = n_samples
        super().__init__(sbert_model_name)
        self.sbert_model.train()
        # for name, module in self.sbert_model.named_modules():
        #     if isinstance(module, nn.Dropout):
        #         module.train()

    def predict_with_dropout(self, doc, summ1, summ2):
        scores1 = []
        scores2 = []

        for _ in range(self.n_samples):
            score1, score2 = self.forward(doc=doc, summ1=summ1, summ2=summ2)

            scores1.append(score1), scores2.append(score2)

        return torch.stack(scores1), torch.stack(scores2)

    def mean_var_from_scores(self, scores: torch.tensor):
        mean_scores = scores.mean().numpy()

        var_scores = scores.var().numpy()

        return mean_scores, var_scores

    def predict_mean_var(self, doc, summ1, summ2):
        scores1, scores2 = self.predict_with_dropout(doc, summ1, summ2)

        return self.mean_var_from_scores(scores1), self.mean_var_from_scores(scores2)


class TinyBertDeepLearner(nn.Module):
    def __init__(
        self,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
    ) -> None:
        super().__init__()
        self.base_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.base_model.train()

        cs = nn.CosineSimilarity(dim=1)
        self.cs = nn.DataParallel(cs)

    def _get_embedding(self, text):
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        embeddings = self.base_model(**encodings)[0]
        # use cls token as it is more robust to padding
        return embeddings[:, 0, :]

    def forward(self, doc, summ1, summ2):
        self.doc_embedding = self._get_embedding(doc)
        self.summary1_embedding = self._get_embedding(summ1)
        self.summary2_embedding = self._get_embedding(summ2)

        score1 = self.cs(self.summary2_embedding, self.summary1_embedding)
        score2 = self.cs(self.doc_embedding, self.summary2_embedding)

        return score1, score2


class TinyBertDeepLearnerWithMCDropout(TinyBertDeepLearner):
    def __init__(
        self,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        n_samples: int = 10,
    ) -> None:
        super().__init__(model_name=model_name)
        self.n_samples = n_samples

    def predict_with_dropout(self, doc, summ1, summ2):
        scores1 = []
        scores2 = []

        for _ in range(self.n_samples):
            score1, score2 = self.forward(doc=doc, summ1=summ1, summ2=summ2)

            scores1.append(score1)
            scores2.append(score2)

        return torch.stack(scores1), torch.stack(scores2)

    def mean_var_from_scores(self, scores: torch.tensor):
        mean_scores = scores.mean().numpy()
        var_scores = scores.var().numpy()

        return mean_scores, var_scores

    def predict_mean_var(self, doc, summ1, summ2):
        scores1, scores2 = self.predict_with_dropout(doc, summ1, summ2)

        return self.mean_var_from_scores(scores1), self.mean_var_from_scores(scores2)


class SummDataset(Dataset):
    def __init__(self, summ_pairs: list) -> None:
        super().__init__()
        self.summ_pairs = summ_pairs

    def __len__(self):
        return len(self.summ_pairs)

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
