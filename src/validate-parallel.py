import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from tf_idf.services.parallel_tf_idf import ParallelTfIdfVectorizer


df = pd.read_csv("./datasets/train.csv")
df = df[:4_000]
df = df.dropna()

vectorizer = ParallelTfIdfVectorizer(10)

print("VECTORIZER INITIALIZED")

vectorizer.fit(df["text"])

print("VECTORIZER IS FITTED")

my_tf_idf_vectors = vectorizer.transform(df["text"])

print("TEXTS WERE TRANSFORMED")

labels, label_to_str = df["sentiment"].factorize()

accuracies = []
for _ in range(5):
    x_train__my, x_test__my, y_train__my, y_test__my = train_test_split(my_tf_idf_vectors.to_numpy(), labels, test_size=0.2, shuffle=True)

    my_tf_idf_svc = SVC()
    _ = my_tf_idf_svc.fit(x_train__my, y_train__my)

    predicted_results = my_tf_idf_svc.predict(x_test__my)
    acc = accuracy_score(predicted_results, y_test__my)
    accuracies.append(acc)
    print(f"Accuracy is: {acc}")

print(f"AVERAGE ACCURACY FOR PARALLEL TF-IDF: {sum(accuracies) / len(accuracies)}")
