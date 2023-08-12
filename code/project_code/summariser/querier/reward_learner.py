"""
Wrapper for Gaussian process preference learning (GPPL) to learn the latent
reward function from pairwise preferencelabels expressed by a noisy labeler.
"""
import logging
from typing import List, Optional

import numpy as np

import torch

# from torch.utils.data import Dataset
from torch import nn
from torch.nn.modules.activation import ReLU

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# legacy imports
from sklearn.decomposition import PCA

from gppl.gp_classifier_vb import compute_median_lengthscales
from gppl.gp_pref_learning import GPPrefLearning
from summariser.utils.misc import normaliseList


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

    def train(
        self, pref_history, vector_list, true_scores=None, tr_items=None
    ):
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

            logging.debug(
                "GPPL fitting with %i pairwise labels" % len(new_labels)
            )

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
                        "New best train score %f with rate_s0=%f"
                        % (train_score, rate)
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

        self.mu0 = (heuristics - minh) / (
            maxh - minh
        ) - 0.5  # * 2 * np.sqrt(200)
        self.mu0 = self.mu0 * heuristic_scale + heuristic_offset


class MCDTinyBertDeepLearner(nn.Module):
    def __init__(
        self,
        original_document,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        n_samples: int = 10,
    ):
        super().__init__()
        self.base_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Record of summary processing
        self.n_labels_seen = 0
        # Define og
        self.original_document = original_document
        # Set MCD params
        self.n_samples = n_samples

    def set_layers_to_training_mode(self):
        for module in self.training_mode_layers:
            module.train()

    def set_layers_to_eval_mode(self):
        for module in self.training_mode_layers:
            module.eval()

    def _get_sliding_window_embedding(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256,
    ):
        """Use a specific approach for calculating original document embedding
        since these documents are very long. This model applies a sliding
        window approach to embed the long text.

        The text is divided into overlapping windows of a fixed size, and each
        window is embedded separately. The resulting embeddings are then
        aggregated to form a representation of the entire text.

        Having a representation of the entire text is important since a
        summary's extracted sentences might come from a truncated part of the
        text.
        """

        input_ids = self.tokenizer.encode(text)
        embeddings = []

        # Create sliding windows
        for i in range(0, len(input_ids) - window_size + 1, stride):
            window_ids = input_ids[i : i + window_size]

            # Convert to tensor and add batch dimension
            window_ids_tensor = torch.tensor([window_ids])

            # Embed the window
            with torch.no_grad():
                output = self.base_model(window_ids_tensor)
                # switch approach since we want to take into account entire emb
                window_embedding = output.last_hidden_state.mean(dim=1)
                embeddings.append(window_embedding)

        return torch.stack(embeddings).mean(axis=0)

    def get_similarity_rewards(self, return_tensor: bool = True):
        # return mean scores
        if return_tensor:
            sim_scores = self.similarity_scores

        else:
            sim_scores = self.similarity_scores.detach().numpy()

        return sim_scores.mean(axis=0)

    def get_model_rewards(self):
        values = self.get_similarity_rewards(return_tensor=False)

        return normaliseList(values)

    def predictive_var(self, return_tensor: bool = False):
        # return variance of scores
        if return_tensor:
            sim_scores = self.similarity_scores

        else:
            sim_scores = self.similarity_scores.detach().numpy()

        return sim_scores.var(axis=0)

    def predictive_cov(
        self, idxs: Optional[List[int]], full_cov: bool = False
    ):
        if full_cov:
            # rowvar = False ensures columns are treated as variables
            return np.cov(
                self.similarity_scores.detach().numpy(), rowvar=False
            )
        else:
            reward_variance = self.predictive_var(return_tensor=False)

            if reward_variance.shape[0] == 1:
                return reward_variance[idxs]
            else:
                return np.diag(reward_variance[idxs])


class TinyBertDeepLearnerWithMCDropoutInLayer(MCDTinyBertDeepLearner):
    def __init__(
        self,
        original_document: str,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        n_samples: int = 10,
        dropout_rate: float = 0.1,
        dropout_layers: str = "both",
    ) -> None:
        assert dropout_layers in [
            "both",
            "first",
            "second",
        ], f"{dropout_layers} is not a dropout_layers option."

        super().__init__(
            original_document=original_document,
            model_name=model_name,
            n_samples=n_samples,
        )

        linear1 = nn.Linear(self.base_model.config.hidden_size, 100)
        self.linear1 = nn.DataParallel(linear1)
        self.dropout1 = nn.Dropout(dropout_rate).eval()

        linear2 = nn.Linear(100, 10)
        self.linear2 = nn.DataParallel(linear2)
        self.dropout2 = nn.Dropout(dropout_rate).eval()

        self.out = nn.Linear(10, 1)

        self.relu = ReLU()

        self.train_mode_layers = {
            "both": [self.dropout1, self.dropout2],
            "first": [self.dropout1],
            "second": [self.dropout2],
        }[dropout_layers]

    def forward(self, candidate_summaries: List[str]):
        scores = []
        for summary in candidate_summaries:
            document_summary_pair = (
                self.original_document + " [SEP] " + summary
            )
            ds_embedding = self._get_sliding_window_embedding(
                document_summary_pair
            )

            h1_1 = self.relu(self.linear1(ds_embedding))
            h1_2 = self.dropout1(h1_1)

            h2_1 = self.relu(self.linear2(h1_2))
            h2_2 = self.dropout2(h2_1)

            score = self.out(h2_2)

            scores.append(score.squeeze())

        return torch.stack(scores)

    def get_scores_with_dropout(self, summaries) -> None:
        self.set_layers_to_training_mode()

        all_scores = [
            self.forward(candidate_summaries=summaries)
            for _ in range(self.n_samples)
        ]

        self.set_layers_to_eval_mode()

        self.similarity_scores = torch.stack(all_scores)

    def train_incremental(
        self,
        device: torch.device,
        training_summaries: list,
        preference_score: int,
        loss_margin: float,
    ):
        # Move the model to the specified device
        self = self.to(device)

        # Set the model to training mode
        self.train()

        # Freeze the pretrained BERT model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Define an optimizer that will update only the linear layers
        linear_layers_params = (
            list(self.linear1.parameters())
            + list(self.linear2.parameters())
            + list(self.out.parameters())
        )
        optimizer = torch.optim.Adam(linear_layers_params, lr=5e-5)

        # Define the loss function
        criterion = nn.MarginRankingLoss(margin=loss_margin).to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Call the existing logic for generating scores with dropout
        self.get_scores_with_dropout(training_summaries)

        # Get the mean scores using Monte Carlo Dropout
        mean_scores = self.get_similarity_rewards(return_tensor=True)

        # Map preference score into MRL target
        mrl_target = torch.tensor(1 - 2 * preference_score)

        # Compute the loss
        loss = criterion(mean_scores[0], mean_scores[1], mrl_target)

        # Backpropagate the loss
        loss.backward()

        # Update the weights of the linear layers
        optimizer.step()

        print(f"Loss: {loss.item()}")


class TinyBertDeepLearnerWithMCDropoutInBert(MCDTinyBertDeepLearner):
    def __init__(
        self,
        original_document,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        n_samples: int = 10,
    ) -> None:
        super().__init__(
            original_document=original_document,
            model_name=model_name,
            n_samples=n_samples,
        )

        cs = nn.CosineSimilarity(dim=1)
        self.cs = nn.DataParallel(cs)

        # Text variables
        self.doc_embedding = self._get_sliding_window_embedding(
            original_document
        )

        self.training_mode_layers = [self.base_model]

    def _get_embedding(self, text):
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        embeddings = self.base_model(**encodings).last_hidden_state
        # use cls token as it is more robust to padding
        return embeddings[:, 0, :]

    def forward(self, candidate_summaries):
        # Batched embeddings
        self.summaries_embeddings = self._get_embedding(candidate_summaries)

        scores = self.cs(
            self.doc_embedding.repeat(len(candidate_summaries), 1),
            self.summaries_embeddings,
        )

        return scores

    def get_scores_with_dropout(self, summaries) -> None:
        self.set_layers_to_training_mode()

        all_scores = [
            self.forward(candidate_summaries=summaries)
            for _ in range(self.n_samples)
        ]

        self.set_layers_to_eval_mode()

        self.similarity_scores = torch.stack(all_scores)

    def train_incremental(
        self,
        device: torch.device,
        training_summaries: list,
        preference_score: int,
        loss_margin: float,
    ):
        self = self.to(device)
        # Set the model to training mode
        self.train()

        # Define optimiser
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        # Define the loss function
        criterion = nn.MarginRankingLoss(margin=loss_margin).to(device)

        # Clear the gradients
        optimizer.zero_grad()

        self.get_scores_with_dropout(training_summaries)

        # Get the mean scores using Monte Carlo Dropout
        mean_scores = self.get_similarity_rewards(return_tensor=True)

        # map preference score into MRL target
        # If preference_score is 0 -> 1 (first input is prefered)
        # If preference_score is 1 -> -1 (second input is prefered)
        mrl_target = torch.tensor(1 - 2 * preference_score)

        # Compute the loss
        loss = criterion(mean_scores[0], mean_scores[1], mrl_target)

        # # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()

        print(f"Loss: {loss.item()}")


class MCDSBertDeepRanker(nn.Module):
    def __init__(
        self,
        original_document: str,
        model_name: str = "bert-large-nli-stsb-mean-tokens",
        n_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Architecture
        self.base_model = SentenceTransformer(model_name_or_path=model_name)

        linear1 = nn.Linear(self.base_model.config.hidden_size, 100)
        self.linear1 = nn.DataParallel(linear1)
        self.dropout1 = nn.Dropout(dropout_rate).eval()

        linear2 = nn.Linear(100, 10)
        self.linear2 = nn.DataParallel(linear2)
        self.dropout2 = nn.Dropout(dropout_rate).eval()

        self.out = nn.Linear(10, 1)

        self.relu = ReLU()

        # Define variables
        self.original_document = original_document
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        self.doc_embedding = self.base_model.encode(
            original_document, convert_to_tensor=True
        ).unsqueeze(0)

    def forward(self, summary: str):
        pass
