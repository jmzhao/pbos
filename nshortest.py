import heapq
import operator

def nshortest(adjmat, n):
    """N shortest paths given a DAG as adjacency matrix, assuming (i, j) for all i < j.

    Returns:
        A list of (score, path), sorted by score. Path is a tuple of node ids.

    Examples:
    >>> adj = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    >>> nshortest(adj, 2)
    [(1, (0, 2)), (2, (0, 1, 2))]
    """
    candss = [[(0, (0, ))]]
    for j in range(1, len(adjmat)):
        cands = []
        for i in range(j):
            for icand in candss[i]:
                iscore, ipath = icand
                score = iscore + adjmat[i][j]
                path = ipath + (j, )
                cand = score, path
                cands.append(cand)
        candss.append(heapq.nsmallest(n, cands, key=operator.itemgetter(0)))
    return candss[-1]
