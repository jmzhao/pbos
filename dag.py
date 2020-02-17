import heapq


class DAG:
    def __init__(self, subword_score, word):
        self.subword_score = subword_score
        self.word = word

    @property
    def graph(self):
        n = len(self.word) + 1
        graph = [[0 for _ in range(n)] for _ in range(n)]
        for start in range(len(self.word)):
            for stop in range(start + 1, n):
                subword = self.word[start:stop]
                graph[start][stop] = self.subword_score.get(subword, 0)
        return graph

    @property
    def prefix_score(self):
        prefix_score = [0] * (len(self.word) + 2)
        prefix_score[0] = 1

        pq = [0]
        while pq:
            src = heapq.heappop(pq)
            for dst, score in enumerate(self.graph[src]):
                if not self.graph[src][dst]:
                    continue
                prefix_score[dst] += prefix_score[src] * score
                if dst not in pq:
                    heapq.heappush(pq, dst)

        return prefix_score

    @property
    def suffix_score(self):
        suffix_score = [0] * (len(self.word) + 2)
        suffix_score[len(self.word)] = 1

        pq = [len(self.word)]
        while pq:
            dst = heapq.heappop(pq)
            # TODO: improve the performance of this loop
            for src, score in enumerate(
                [self.graph[e][dst] for e in range(len(self.word) + 1)]
            ):
                if not self.graph[src][dst]:
                    continue
                suffix_score[src] += suffix_score[dst] * score
                if src not in pq:
                    heapq.heappush(pq, src)

        return suffix_score

