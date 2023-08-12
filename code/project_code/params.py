# Reward Learners
from summariser.querier.reward_learner import (
    BERTHRewardLearner,
    BERTRewardLearner,
    TinyBertDeepLearnerWithMCDropout,
)
from summariser.querier.logistic_reward_learner import LogisticRewardLearner

# Queriers
from summariser.querier.expected_improvement_querier import (
    ExpectedImprovementQuerier,
    ExpImpQuerierForDeepLearner,
)
from summariser.querier.expected_information_querier import (
    InformationGainQuerier,
)
from summariser.querier.gibbs_querier import GibbsQuerier
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier
from summariser.querier.pairwise_uncertainty_secondorder_querier import (
    PairUncSOQuerier,
)
from summariser.querier.thompson_querier import (
    ThompsonTopTwoQuerier,
    ThompsonInformationGainQuerier,
)
from summariser.querier.uncertainty_querier import UncQuerier
from summariser.querier.random_querier import RandomQuerier

QUERIER_TYPE_DICT = {
    "gibbs": GibbsQuerier,
    "unc": UncQuerier,
    "pair_unc": PairUncQuerier,
    "pair_unc_SO": PairUncSOQuerier,
    "imp": ExpectedImprovementQuerier,
    "dl_imp": ExpImpQuerierForDeepLearner,
    "eig": InformationGainQuerier,
    "ttt": ThompsonTopTwoQuerier,
    "tig": ThompsonInformationGainQuerier,
    "tp": ThompsonInformationGainQuerier,
    "default": RandomQuerier,
}

LEARNER_TYPE_DICT = {
    "LR": LogisticRewardLearner,
    # GPPL with heuristics as the prior mean
    "BDLH": BERTHRewardLearner,
    "BDL": BERTRewardLearner,
    "TBDL_MCD": TinyBertDeepLearnerWithMCDropout,
}
