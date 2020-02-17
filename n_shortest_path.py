import argparse
import heapq
import logging
import operator

from utils.load import load_vocab
from utils.preprocess import normalize_prob

logging.basicConfig(level=logging.INFO)


class DAG:
    def __init__(self, subword_score, n_largest):
        self.subword_score = subword_score
        self.n_largest = n_largest

    def build_graph(self, word):
        graph = {}
        for start in range(len(word)):
            out_edges = {}
            for stop in range(start + 1, len(word) + 1):
                subword = word[start:stop]
                if subword in self.subword_score:
                    out_edges[stop] = self.subword_score.get(subword)
            graph[start] = out_edges
        return graph

    def predict(self, word):
        dag = self.build_graph(word)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_list', '-f', default="./datasets/unigram_freq.csv",
                        help="list of words to create subword vocab")
    parser.add_argument('--n_largest', '-n', type=int, default=5,
                        help="the number of segmentations to show")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="interactive mode")
    parser.add_argument('--test_file', '-t',
                        help="list of words to be segmented")
    args = parser.parse_args()

    logging.info(f"building subword vocab from `{args.word_list}`...")
    part_count = load_vocab(args.word_list)
    part_prob = normalize_prob(part_count)
    logging.info(f"subword vocab size: {len(part_prob)}")
    dag = DAG(subword_score=part_prob, n_largest=args.n_largest)
    logging.info(f"Ready.")
    if args.interactive:
        while True:
            word = input().strip()
            if not word:
                break
            dag.test([word])
    elif args.test_file:
        dag.test((line.strip() for line in open(args.test_file)))
    else:
        dag.test(
            [
                "ilikeeatingapples",
                "pineappleanapplepie",
                "bioinfomatics",
                "technical",
                "electrical",
                "electronic",
            ]
        )
