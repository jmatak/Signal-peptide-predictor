import numpy as np

from src.dataset_manager import fetch_data


def get_mean(l):
    return np.mean(l)


def get_st_dev(l):
    return np.std(l)


def get_label_frequencies(input_data, labels):
    frequencies = {}
    for l in labels:
        label_data = []
        for data in input_data:
            frequency = 0
            for s in data.output:
                if s.lower() == l.lower(): frequency += 1
            label_data.append(frequency)
        frequencies[l] = label_data

    return frequencies


if __name__ == '__main__':
    data = fetch_data("../data/training_data/positive_examples", 1)
    frequencies = get_label_frequencies(data, "nhc")
    for key in frequencies.keys():
        print(key)
        print(get_mean(frequencies[key]))
        print(get_st_dev(frequencies[key]))
        print(frequencies[key])
