import os

import resources as res
from summariser.utils.corpus_reader import CorpusReader
from summariser.utils.reader import readSummaries
from summariser.vector.vector_generator import Vectoriser
import numpy as np


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


if __name__ == "__main__":
    for dataset in ["DUC2001", "DUC2002", "DUC2004"]:
        # read documents and ref. summaries

        summary_vecs_cache_dir = "./data/summary_vectors/supert/"
        reader = CorpusReader(res.PROCESSED_PATH)
        data = reader.get_data(dataset)

        for topic, docs, models in data:
            summaries_actions_list, _ = readSummaries(dataset, topic, "heuristic")
            print(f"num of summaries read: {len(summaries_actions_list)}")

            # Use the vectoriser to obtain summary embedding vectors
            text_generator = SummaryTextGenerator(docs)
            summaries_action_text = text_generator.getSummaryText(
                summaries_actions_list
            )

            # Write to the output file
            output_file = os.path.join(res.SUMMARY_DB_DIR, dataset, topic, "text")
            with open(output_file, "w") as ofh:
                for actions_list, sum_text in zip(
                    summaries_actions_list, summaries_action_text
                ):
                    act_str = np.array(actions_list).astype(str)
                    actions_line = "actions:" + ",".join(act_str) + "\n"
                    ofh.write(actions_line)

                    text_line = "text:" + ". ".join(sum_text) + "\n"
                    ofh.write(text_line)
