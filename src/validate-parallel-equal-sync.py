import numpy as np
import pandas as pd

from tf_idf.services.parallel_tf_idf import ParallelTfIdfVectorizer
from tf_idf.services.tf_idf import TfIdfVectorizer


df = pd.read_csv("/home/kinfi4/python/TF-IDF/src/datasets/train.csv")
df = df[:5_000]
df = df.dropna()

parallel_tf_idf = ParallelTfIdfVectorizer()
parallel_tf_idf.fit(df["text"])
parallel_vectors = parallel_tf_idf.transform(df["text"]).to_numpy()

sync_tf_idf = TfIdfVectorizer()
sync_tf_idf.fit(df["text"])
sync_vectors = sync_tf_idf.transform(df["text"]).to_numpy()

print(f"ARRAYS ARE EQUAL: {np.array_equal(parallel_vectors, sync_vectors)}")
