from time import perf_counter

import click
import pandas as pd

from tf_idf.services.parallel_tf_idf import ParallelTfIdfVectorizer
from tf_idf.services.tf_idf import TfIdfVectorizer


@click.group()
def cli():
    pass


@cli.command()
@click.option("-t", "--type", "vectorizer_type")
@click.option("-w", "--workers", type=int)
def transform_text(vectorizer_type: str, workers: int = None):
    if vectorizer_type not in ("sync", "parallel"):
        raise ValueError("Vectorizer type must be equal `sync` or `parallel`")

    while True:
        items_number = int(input("Enter the number of item to transform: "))
        if items_number == 0:
            return

        if items_number > 1_000_000_000:
            print("Item number must be less than 1000000")
            continue

        data = pd.read_csv("./datasets/test-speed-data.csv")
        data = data[:items_number]

        if vectorizer_type == "sync":
            vectorizer = TfIdfVectorizer()
        else:
            vectorizer = ParallelTfIdfVectorizer(workers_number=workers)

        vectorizer.fit(data["text"])

        start_time = perf_counter()
        vectorizer.transform(data["text"])
        end_time = perf_counter()

        print(f"Time took for {vectorizer_type} algorithm to transform {items_number} texts is: {end_time - start_time} seconds")


if __name__ == "__main__":
    cli()
