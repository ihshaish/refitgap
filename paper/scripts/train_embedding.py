"""Train the ASRS domain embedding: word2vec CBOW, 200 dimensions, on narratives up to December 2009 only.

Reads ../data_asrs_v2/*.csv; writes ../embeddings/asrs_w2v_pre2010_200d.kv, the domain vectors of
conditions C1 and C3 in train_sequential.py and train_methods.py.
Needs the ASRS export, which the repository does not ship, and gensim."""
import glob

from build_streams import read_records, normalise
from asrs_profile import YEAR_RE
from gensim.models import Word2Vec

sents = []
for p in sorted(glob.glob("../data_asrs_v2/*.csv")):
    m = YEAR_RE.search(p)
    if m and int(m.group(1)) <= 2009:
        sents += [normalise(n).split() for _, ym, _, n in read_records(p) if ym <= 200912 and n]
print("sentences:", len(sents))
w2v = Word2Vec(sentences=sents, vector_size=200, sg=0, window=5, min_count=5,
               negative=5, epochs=30, workers=8, seed=20260802)
w2v.wv.save("../embeddings/asrs_w2v_pre2010_200d.kv")
print("vocab:", len(w2v.wv))
