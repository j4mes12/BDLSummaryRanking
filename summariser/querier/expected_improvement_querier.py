# This pulls strong inspiration from Simpson et al. [1], but this function has
# undergone a structureal refresh to perform differently.

import numpy as np
from scipy.stats import norm
from summariser.utils.misc import normaliseList


class ExpImpQuerierForDeepLearner:
    """this is the same as the above class, but has been adapted to work
    with a deep learning-type reward_learner
    """

    def __init__(
        self,
        heuristic_values: np.array,
        full_cov: bool = True,
        learnt_weight: float = 0.5,
    ):
        self.heuristics = heuristic_values
        self.learnt_values = [0.0] * len(heuristic_values)
        self.full_cov = full_cov
        self.learnt_weight = learnt_weight

    def _get_candidates(self, f):
        # consider only the top ranked items.
        num = 20**2
        candidate_idxs = np.argsort(f)[-num:]

        return candidate_idxs

    def _compute_pairwise_scores(self, f: np.array, Cov: np.array) -> np.array:
        best_idx = np.argmax(f)
        f_best = f[best_idx]

        sigma = (
            Cov[best_idx, best_idx]
            + np.diag(Cov)
            - Cov[best_idx, :]
            - Cov[:, best_idx]
        )
        sigma[
            best_idx
        ] = 1  # avoid the invalid value errors -- the value of u should be 0

        # for all candidates, compute u = (mu - f_best) / sigma
        # mean improvement. Similar to preference likelihood but that adds in 2
        u = (f - f_best) / np.sqrt(sigma)
        # due to labelling noise
        cdf_u = norm.cdf(u)  # probability of improvement
        pdf_u = norm.pdf(u)

        E_improvement = np.sqrt(sigma) * (
            u * cdf_u + pdf_u
        )  # the sqrt was added back in in October 2020 after it was
        # accidentally removed in???
        E_improvement[best_idx] = -np.inf

        # make it back into a matrix
        E_imp_mat = np.zeros((f.size, f.size))
        E_imp_mat[best_idx, :] = E_improvement

        return E_imp_mat

    def inLog(self, sum1: int, sum2: int, log: tuple):
        for entry in log:
            if [sum1, sum2] in entry:
                return True
            elif [sum2, sum1] in entry:
                return True

        return False

    def getMixReward(self, learnt_values, learnt_weight=-1):
        if learnt_weight == -1:
            learnt_weight = self.learnt_weight

        mix_values = np.array(learnt_values) * learnt_weight + np.array(
            self.heuristics
        ) * (1 - learnt_weight)
        return normaliseList(mix_values)

    def getQuery(
        self,
        f,
        Cov,
        candidate_idxs,
        log: tuple,
    ):
        # get the current best estimate

        pairwise_scores = self._compute_pairwise_scores(f, Cov)

        # Find out which of our candidates have been compared already
        for data_point in log:
            if (
                data_point[0][0] not in candidate_idxs
                or data_point[0][1] not in candidate_idxs
            ):
                continue
            dp0 = np.argwhere(candidate_idxs == data_point[0][0]).flatten()[0]
            dp1 = np.argwhere(candidate_idxs == data_point[0][1]).flatten()[0]
            pairwise_scores[dp0, dp1] = -np.inf
            pairwise_scores[dp1, dp0] = -np.inf

        selected = np.unravel_index(
            np.argmax(pairwise_scores), pairwise_scores.shape
        )
        pe_selected = pairwise_scores[selected[0], selected[1]]
        selected = (candidate_idxs[selected[0]], candidate_idxs[selected[1]])

        print(
            f"Chosen candidate: {selected[0]}, vs. best: {selected[1]}, "
            + f"with score = {pe_selected}"
        )

        return selected[0], selected[1]
