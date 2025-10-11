# If you need to import additional packages or classes, please import here.
from typing import List, Tuple
import math
from collections import Counter

from PyQt5.QtGui.QRawFont import weight


def history_search(
        corpus: List[str],
        K: int,
        queries: List[Tuple[int, str]]
) -> str:
    results = []
    N = len(corpus)
    tokenized_docs = [doc.split() for doc in corpus]

    for t,query in queries:
        start = max(0, t - K + 1)
        window_docs = tokenized_docs[start:t+1]
        window_ids = list(range(start,t+1))
        window_size = len(window_docs)

        if window_size == 0:
            results.append("-1")
            continue

        vocab = set()
        for doc in window_docs:
            vocab.update(doc)
        vocab.update(query.split())


        idf = {}
        for w in vocab:
            df = sum(1 for doc in window_docs if w in doc)
            idf[w] = math.log((window_size + 1) / (df + 1)) + 1

        q_tokens = query.split()
        q_tf = Counter(q_tokens)
        q_vec = {w: q_tf[w] * idf[w] for w in q_tf if w in idf}

        q_norm = math.sqrt(sum(v*v for v in q_vec.values()))

        best_doc = -1
        best_sim = -1.0

        for idx, doc in enumerate(window_docs):
            doc_id = window_ids[idx]
            tf = Counter(doc)
            weight = (K - (window_size - 1 - idx)) / K

            d_vec = {w: tf[w] * idf[w] * weight for w in tf if w in idf}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))

            if d_norm == 0 or q_norm ==0 :
                sim = 0.0
            else:
                dot = sum(q_vec.get(w, 0) * d_vec.get(w, 0) for w in vocab)
                sim = dot / (q_norm * d_norm)

            if sim >= 0.6:
                if sim > best_sim or (abs(sim - best_sim) < 1e-9 and doc_id<best_doc):
                    best_sim = sim
                    best_doc = doc_id
        results.append(str(best_doc if best_doc != -1 else -1))
    return " ".join(results)






if __name__ == "__main__":
    # please define the python3 input here. For example: line = input()
    N = int(input().strip())

    corpus =  [input().strip() for _ in range(N)]
    K = int(input().strip())
    P = int(input().strip())
    queries = []
    for _ in range(P):
        line = input().strip().split()
        t = int(line[0])
        q = " ".join(line[1:])
        queries.append((t, q))
    print(history_search(corpus, K, queries))
# please define the python3 output here. For example: print().

