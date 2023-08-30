# This code was inspired by Simpson et al. when extracting summary vectors [1].
# However, this implementation is new and is to save summary text within the HPC.


from summariser.utils.reader import readSampleSummaries
from summariser.utils.corpus_reader import CorpusReader
from summariser.vector.vector_generator import Vectoriser
from resources import PROCESSED_PATH, BASE_DIR

import numpy as np
import os


class SummaryTextGenerator(Vectoriser):
    def getSummaryText(self, summary_actions_list, stemmed=True):
        summary_texts = []

        for actions in summary_actions_list:
            summary = ". ".join(
                [
                    (
                        self.stemmed_sentences_list[act]
                        if stemmed
                        else self.sentences.untokenized_form
                    )
                    for act in actions
                ]
            )

            summary_texts.append(summary)
        return np.array(summary_texts)


reader = CorpusReader(PROCESSED_PATH)
base_savepath = os.path.join(BASE_DIR, "data", "summary_candidates")

for dataset in ["DUC2001", "DUC2002", "DUC2004"]:
    print("---------", dataset, "---------")

    data = reader.get_data(dataset)

    for topic, docs, models in data:
        print("---------", topic, "---------")
        summary_text_cache_file = os.path.join(
            base_savepath, "summary_texts_{}_{}.csv".format(dataset, topic)
        )

        (
            summaries,
            ref_values_dic,
            heuristic_list,
        ) = readSampleSummaries(dataset, topic, "supert")
        print(f"num of summaries read: {len(summaries)}")

        text_generator = SummaryTextGenerator(docs)
        summary_text = text_generator.getSummaryText(summaries)

        np.savetxt(
            summary_text_cache_file, summary_text, fmt="%s", delimiter="#####"
        )
        print(f"Cached summary vectors to {summary_text_cache_file}")
