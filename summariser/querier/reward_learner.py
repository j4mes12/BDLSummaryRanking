from typing import List, Optional

import numpy as np

import torch

# from torch.utils.data import Dataset
from torch import nn
from torch.nn.modules.activation import ReLU

# from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

from summariser.utils.misc import normaliseList


class MCDTinyBertDeepLearner(nn.Module):
    def __init__(
        self,
        original_document,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        n_samples: int = 10,
        full_cov: bool = True,
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
        self.training_mode_layers = []
        self.full_cov = full_cov

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
                window_embedding = output[0].mean(dim=1)
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
                self.similarity_scores[:, idxs].detach().numpy(), rowvar=False
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

        self.training_mode_layers = {
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

        return loss.item()


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
        embeddings = self.base_model(**encodings)[0]
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

        return loss.item()


# class MCDSBertDeepRanker(nn.Module):
#     def __init__(
#         self,
#         original_document: str,
#         model_name: str = "bert-large-nli-stsb-mean-tokens",
#         n_samples: int = 10,
#         dropout_rate: float = 0.1,
#     ):
#         super().__init__()

#         # Architecture
#         self.base_model = SentenceTransformer(model_name_or_path=model_name)

#         linear1 = nn.Linear(self.base_model.config.hidden_size, 100)
#         self.linear1 = nn.DataParallel(linear1)
#         self.dropout1 = nn.Dropout(dropout_rate).eval()

#         linear2 = nn.Linear(100, 10)
#         self.linear2 = nn.DataParallel(linear2)
#         self.dropout2 = nn.Dropout(dropout_rate).eval()

#         self.out = nn.Linear(10, 1)

#         self.relu = ReLU()

#         # Define variables
#         self.original_document = original_document
#         self.n_samples = n_samples
#         self.dropout_rate = dropout_rate

#         self.doc_embedding = self.base_model.encode(
#             original_document, convert_to_tensor=True
#         ).unsqueeze(0)

#     def forward(self, summary: str):
#         pass
