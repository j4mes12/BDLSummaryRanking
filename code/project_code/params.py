# Reward Learners
from summariser.querier.reward_learner import (
    TinyBertDeepLearnerWithMCDropoutInBert,
    TinyBertDeepLearnerWithMCDropoutInLayer,
)


LEARNER_TYPE_DICT = {
    "TBDL_IB": TinyBertDeepLearnerWithMCDropoutInBert,
    "TBDL_IL": TinyBertDeepLearnerWithMCDropoutInLayer,
}
