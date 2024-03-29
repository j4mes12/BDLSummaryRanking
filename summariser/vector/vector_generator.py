from nltk.tokenize import word_tokenize
from summariser.vector.base import Sentence
import summariser.utils.data_helpers as dh

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from summariser.vector.state_type import State, StateLengthComputer
import random


class Vectoriser:
    def __init__(
        self,
        docs,
        sum_len=100,
        no_stop_words=True,
        stem=True,
        block=1,
        base=200,
        lang="english",
    ):
        self.docs = docs
        self.without_stopwords = no_stop_words
        self.stem = stem
        self.block_num = block
        self.base_length = base
        self.language = lang
        self.sum_token_length = sum_len

        self.stemmer = PorterStemmer()
        self.stoplist = set(stopwords.words(self.language))
        self.sim_scores = {}
        self.stemmed_sentences_list = []

        self.load_data()

    def sampleRandomReviews(
        self, num, heuristic_reward=True, rouge_reward=True, models=None
    ):
        heuristic_list = []
        rouge_list = []
        action_list = []

        for _ in range(num):
            state = State(
                self.sum_token_length,
                self.base_length,
                len(self.sentences),
                self.block_num,
                self.language,
            )
            while state.available_sents != [0]:
                new_id = random.choice(state.available_sents)
                if (new_id == 0) or (
                    new_id > 0
                    and len(self.sentences[new_id - 1].untokenized_form.split(" "))
                    > self.sum_token_length
                ):
                    continue
                state.updateState(new_id - 1, self.sentences)
            actions = state.historical_actions
            action_list.append(actions)

            if heuristic_reward:
                rew = state.getTerminalReward(
                    self.sentences,
                    self.stemmed_sentences_list,
                    self.sent2tokens,
                    self.sim_scores,
                )
                heuristic_list.append(rew)

            if rouge_reward:
                assert models is not None
                r_dic = {}
                for model in models:
                    model_name = model[0].split("/")[-1].strip()
                    rew = state.getOptimalTerminalRougeScores(model)
                    r_dic[model_name] = rew
                rouge_list.append(r_dic)

        return action_list, heuristic_list, rouge_list

    def getSummaryVectors(self, summary_actions_list):
        vector_list = []

        for action_list in summary_actions_list:
            # For each summary, we create a state object
            state = State(
                self.sum_token_length,
                self.base_length,
                len(self.sentences),
                self.block_num,
                self.language,
            )

            # Now, we construct the text of the summary from the list of
            # actions, which are IDs of sentences to add.
            for _, action in enumerate(action_list):
                state.updateState(action, self.sentences, read=True)

            # state.getSelfVector will return the vector representation of the
            # summary. This calls self.getStateVector, which we need to
            # overwrite to define a new type of vector representation. This
            # makes use of tate.draft_summary_list, which is a list of the
            # sentence strings in the summary.
            vector = state.getSelfVector(self.top_ngrams_list, self.sentences)
            vector_list.append(vector)

        return vector_list

    def sent2tokens(self, sent_str):
        if self.without_stopwords and self.stem:
            return dh.sent2stokens_wostop(
                sent_str, self.stemmer, self.stoplist, self.language
            )
        elif (not self.without_stopwords) and self.stem:
            return dh.sent2stokens(sent_str, self.stemmer, self.language)
        elif self.without_stopwords and (not self.stem):
            return dh.sent2tokens_wostop(sent_str, self.stoplist, self.language)
        else:  # both false
            return dh.sent2tokens(sent_str, self.language)

    def load_data(self):
        self.sentences = []
        for doc_id, doc in enumerate(self.docs):
            doc_name, doc_sents = doc
            doc_tokens_list = []
            for sent_id, sent_text in enumerate(doc_sents):
                token_sent = word_tokenize(sent_text, self.language)
                current_sent = Sentence(token_sent, doc_id, sent_id + 1)
                untokenized_form = dh.untokenize(token_sent)
                current_sent.untokenized_form = untokenized_form
                current_sent.length = len(untokenized_form.split(" "))
                self.sentences.append(current_sent)
                sent_tokens = self.sent2tokens(untokenized_form)
                doc_tokens_list.extend(sent_tokens)
                stemmed_form = " ".join(sent_tokens)
                self.stemmed_sentences_list.append(stemmed_form)
        # print('total sentence num: ' + str(len(self.sentences)))

        self.state_length_computer = StateLengthComputer(
            self.block_num, self.base_length, len(self.sentences)
        )
        self.top_ngrams_num = self.state_length_computer.getStatesLength(self.block_num)
        self.vec_length = self.state_length_computer.getTotalLength()

        sent_list = []
        for sent in self.sentences:
            sent_list.append(sent.untokenized_form)
        self.top_ngrams_list = dh.getTopNgrams(
            sent_list,
            self.stemmer,
            self.language,
            self.stoplist,
            2,
            self.top_ngrams_num,
        )
