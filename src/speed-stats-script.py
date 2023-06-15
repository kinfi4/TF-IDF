from time import perf_counter
from typing import Type

import pandas as pd

from tf_idf.services.parallel_tf_idf import ParallelTfIdfVectorizer
from tf_idf.services.tf_idf import TfIdfVectorizer

BEST_WORKERS_NUMBER = 5


def single_transform(vectorizer: TfIdfVectorizer, data) -> float:
    start_time = perf_counter()
    vectorizer.transform(data)
    end_time = perf_counter()

    return end_time - start_time


def run_testing(vectorizer_type: Type[TfIdfVectorizer]) -> dict[int, float]:
    average_time_needed: dict[int, float] = {}

    for texts_number in [1000, 5000, 10_000, 25_000, 50_000, 100_000, 250_000]:
        time_needed_list: list[float] = []

        data = pd.read_csv("./datasets/test-speed-data.csv")
        data = data[:texts_number]

        vectorizer = TfIdfVectorizer()

        vectorizer.fit(data["text"])

        for _ in range(10):
            t = single_transform(vectorizer, data["text"])
            time_needed_list.append(t)
            print(f"TIME NEEDED FOR {texts_number} IS: {t}")

        # average_time_needed[texts_number] = single_transform(texts_number)

        average_time_needed[texts_number] = sum(time_needed_list) / len(time_needed_list)
        print(f"TESTS FOR TEXTS NUMBER: {texts_number} FOR AVG TIME: {average_time_needed[texts_number]} DONE...\n\n\n")

    return average_time_needed


if __name__ == "__main__":
    res = run_testing(TfIdfVectorizer)

    print("\nTESTS FOR SYNC VECTORIZER DONE.\n")

    res = run_testing(ParallelTfIdfVectorizer)

    print(res)


# TIME NEEDED FOR 250000 IS: 1237.2508207259816
# TIME NEEDED FOR 250000 IS: 1236.3954397650086
# TIME NEEDED FOR 250000 IS: 1236.7988030969864
# TESTS FOR TEXTS NUMBER: 250000 FOR AVG TIME: 1236.8150211959921 DONE...
#
#
#
# {250000: 1236.8150211959921}
#
# TESTS FOR SYNC VECTORIZER DONE.


# TIME NEEDED FOR 100000 IS: 267.2508580540016
# TIME NEEDED FOR 100000 IS: 266.59274962201016
# TIME NEEDED FOR 100000 IS: 266.57580062001944
# TIME NEEDED FOR 100000 IS: 268.216293578007
# TESTS FOR TEXTS NUMBER: 100000 FOR AVG TIME: 267.55109216340935 DONE...
