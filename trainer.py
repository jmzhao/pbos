import collections

import numpy as np

from dag import DAG


class Trainer:
    def __init__(self, subword_prob, embedding_dim):
        self.subword_prob = subword_prob
        self.subword_emb = collections.defaultdict(
            lambda: np.random.rand(embedding_dim)
        )

    def train(self, word, embedding):
        dag = DAG(self.subword_prob, word)

        graph = dag.graph
        suffix_score = dag.suffix_score
        prefix_score = dag.prefix_score

        # TODO: feed forward


