from time import perf_counter

import pandas as pd

from tf_idf.services.parallel_tf_idf import ParallelTfIdfVectorizer


def single_transform(runs_count: int, texts_count: int, workers_number: int) -> float:
    # print(f"STARTED EXPERIMENT FOR {texts_count} and {workers_number} workers...")
    data = pd.read_csv("datasets/test-speed-data.csv")
    data = data[:texts_count]

    vectorizer = ParallelTfIdfVectorizer(workers_number=workers_number)
    vectorizer.fit(data["text"])

    single_times = []
    for _ in range(runs_count):
        start_time = perf_counter()
        vectorizer.transform(data["text"])
        end_time = perf_counter()

        # print(f"TIME TOOK FOR {workers_number} FOR TEXTS COUNT {texts_count}: {end_time - start_time}")
        single_times.append(end_time - start_time)

    # print(f"TIMES FOR {workers_number}: {single_times}")
    return sum(single_times) / len(single_times)


if __name__ == "__main__":
    for text_number in [1000, 5000, 10_000, 25_000, 50_000, 100_000, 250_000]:
        times: list[float] = []
        for workers in [2, 4, 8, 16, 32, 64, 128]:
            avg_time = single_transform(5, text_number, workers)
            times.append(avg_time)

            print(f"Workers: {workers}: {avg_time}")

        print(f"TIMES for {text_number}: {times}")


# Times for 1000: [0.2466331600153353, 0.18212963399128057, 0.22375469299731776, 0.2943365300016012, 0.26754108298337087, 0.3116321059933398, 0.6107194630021695]
# Times for 5000: [1.0894949270004872, 0.9048059649940114, 0.8778350939974189, 0.8798444519925397, 0.9800689509720542, 1.1359656080021523, 1.8827132279984653]
# Times for 10000: [2.0588048929930665, 2.0669940570078325, 2.102817816019524, 2.210202066024067, 2.468990106979618, 2.8672760800109245, 3.530411201005336]
# Times for 25000: [8.411148109007627, 8.526359917013906, 9.117280388978543, 9.96596145699732, 10.087681023025652, 11.586738220998086, 12.888127825019183]
# Times for 50000: [22.238508876005653, 24.964640168007463, 26.090662139991764, 27.41300608200254, 28.59995188898756, 30.152511741005583, 33.33415536500979]
# Times for 100000: [73.58503234197269, 81.44119417801267, 79.5503812720126, 79.21179219000624, 80.46030622100807, 85.05359541298822, 87.87752939999336]

# TIME TOOK FOR 2 IS: 321.95959113002755
# TIME TOOK FOR 4 IS: 263.06680592597695
# TIME TOOK FOR 8 IS: 300.5082066920004
# TIME TOOK FOR 20 IS: 308.5521712420159
# TIME TOOK FOR 50 IS: 314.55421536200447
# TIME TOOK FOR 100 IS: 323.9319790889858
# TIME TOOK FOR 250 IS: 327.9676687849569


# TIMES for 1000: [0.3701256630069111, 0.2833263868116774, 0.2752255014027469, 0.2653761540015694, 0.313458329194691, 0.3105935849947855, 0.36372531059896573]
# TIMES for 5000: [2.20069878140348, 1.5884559017955326, 0.8294484490063041, 0.8301763828087132, 0.8589505893993191, 1.0829428451950662, 1.179801742202835]
# TIMES for 10000: [3.701473764801631, 2.5096961583883965, 2.3816063634003513, 2.416482576995622, 2.4353030962054616, 2.5469476486032363, 2.6647588910069318]
# TIMES for 25000: [11.74206210699631, 8.075677523005288, 8.077293976192596, 8.614702030201443, 9.126031560200499, 9.452302918198985, 9.906961893185507]
# TIMES for 50000: [28.533160459599458, 20.684943473991005, 22.591311279014917, 24.156246587401256, 25.526019394997274, 27.090638578002107, 28.109246084390907]
# TIMES for 100000: [76.33711202499398, 60.681182322019595, 69.6455923635076, 73.0563720545033, 73.90422296400357, 77.00334838849085, 79.88226587798272]
# TIMES for 250000: [323.21258699850296, 270.6221910339955, 289.7796310749982, 303.20127662000596, 309.0816360754834, 311.30240988949663, 315.95001212050556]
