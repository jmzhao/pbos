import heapq
import operator


class DAG:
    def __init__(self, part_prob, n_largest):
        self.part_prob = part_prob
        self.n_largest = n_largest

    def build_dag(self, word):
        dag = {}
        for start in range(len(word)):
            tmp = {}
            for stop in range(start + 1, len(word) + 1):
                fragment = word[start:stop]
                num = self.part_prob.get(fragment, 0)
                if num > 0:
                    tmp[stop] = num
            dag[start] = tmp
        return dag

    def predict(self, word):
        dag = self.build_dag(word)  # {i: (stop, prob)}
        starts = [0]
        word_probs = [1] * self.n_largest
        parts_arr = [[] for _ in range(self.n_largest)]
        while any(s < len(word) for s in starts):
            candidates = []
            for start, word_prob, parts in zip(starts, word_probs, parts_arr):
                if start not in dag:
                    candidates.append((start, word_prob, parts))
                else:
                    for (stop, part_prob) in dag[start].items():
                        prob = word_prob * part_prob
                        candidates.append((stop, prob, parts + [word[start:stop]]))

            res = heapq.nlargest(self.n_largest, candidates, key=operator.itemgetter(1))
            starts, word_probs, parts_arr = zip(*res)

        return zip(parts_arr, word_probs)

    def test(self, words):
        for word in words:
            print(word)
            for parts, prob in self.predict(word):
                print(f"{prob:.5E}: {'-'.join(parts)}")
            print()
