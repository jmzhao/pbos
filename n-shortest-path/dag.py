import heapq
import math
import operator
from itertools import combinations


def get_substrings(s):
    return [s[x:y] for x, y in combinations(range(len(s) + 1), r=2)]


class DAG:
    def __init__(self, n_largest):
        self.word_prob = {}
        self.n_largest = n_largest

    def load_data(self, filename):
        count = 0
        with open(filename, "r") as f:
            for word in f:
                parts = get_substrings(word.strip())
                for part in parts:
                    self.word_prob[part] = self.word_prob.get(part, 0) + 1
                    count += 1

        self.word_prob = {
            k: math.pow(v / count, 1 / (len(k)))
            for k, v in self.word_prob.items()
        }

    def build_dag(self, word):
        dag = {}
        for start in range(len(word)):
            tmp = {}
            for stop in range(start + 1, len(word) + 1):
                fragment = word[start:stop]
                num = self.word_prob.get(fragment, 0)
                if num > 0:
                    tmp[stop] = num
            dag[start] = tmp
        return dag

    # def predict(self, word):
    #     results = []
    #     dag = self.build_dag(word)  # {i: (stop, prob)}
    #     starts = [0]
    #     ends = [0]
    #     probs = [1] * self.n_largest
    #     while all(s < len(word) for s in starts):
    #         adj = [
    #             (stop, p * prob)
    #             for s, p in zip(ends, probs)
    #             for (stop, prob) in dag[s].items()
    #         ]
    #         res = heapq.nlargest(self.n_largest, adj, key=operator.itemgetter(1))
    #         starts, ps = list(zip(*res))
    #         probs = [p1*p2 for p1, p2 in zip(probs, ps)]
    #         starts = ends
    #
    #     result.append(word[start:end])
    #     return result, prob

    def predict(self, word):
        result = []
        dag = self.build_dag(word)  # {i: (stop, prob)}
        start = 0
        prob = 1
        while start < len(word):
            end, p = max(dag[start].items(), key=operator.itemgetter(1))
            prob *= p
            result.append(word[start:end])
            start = end
        return result, prob

    def test(self, cases):
        for case in cases:
            result, prob = self.predict(case)
            print(f"{prob:.5g}: {' '.join(result)}")
