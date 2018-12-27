import os
import feature_extractor as feature_extractor
from models import Sequence
import numpy as np
import random
from Bio import SeqIO


def fetch_protein_data(training_data):
    data = fetch_data(training_data + "/positive_examples", 1)
    data.extend(fetch_data(training_data + "/negative_examples", -1))
    return data


def fetch_data(directory_name, label):
    data = []
    for root, dir_names, file_names in os.walk(directory_name):
        for f in file_names:
            if f.startswith("."): continue
            examples = get_file_io(os.path.join(root, f), label)
            data.extend(examples)

    return data


def prepare_dataset(data, window_size, callback):
    input_data, signal_data, cleavage_data = [], [], []
    length = len(data)
    percent = int(length / 100)
    for i, sequence in enumerate(data):
        id, sd, cd = feature_extractor.code_sequence(sequence, window_size, data=True)

        if length % (i+1) == percent:
            callback(i+1, length)

        for d, s, c in zip(id, sd, cd):
            input_data.append(d)
            signal_data.append(s)
            cleavage_data.append(c)

    c = list(zip(input_data, signal_data, cleavage_data))
    random.shuffle(c)
    input_data, signal_data, cleavage_data = zip(*c)
    return np.array(input_data), \
           np.hstack((np.array(signal_data).reshape(-1, 1), np.array(cleavage_data).reshape(-1, 1)))


def train_test_split(input_data, output_data, ratio=0.8):
    split_number = int(len(input_data) * ratio)

    return (input_data[:split_number], output_data[:split_number]), \
           (input_data[split_number:], output_data[split_number:])


def prepare_example(sequence, window_size, size=50):
    if len(sequence.input) > size:
        sequence.input = sequence.input[:size]
        sequence.output = sequence.output[:size] if sequence.output else None
    return np.array(feature_extractor.code_sequence(sequence, window_size))


def get_file_io(filename, label):
    sequences = []

    name = None
    input_sequence = None
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith(">"):
                name = line[1:]
                continue
            if line.startswith("#"):
                output_sequence = line[1:].strip()
                sequence = Sequence(input_sequence, output_sequence, label=label, name=name)
                sequences.append(sequence)
                input_sequence = None
                name = None
                continue
            if input_sequence is None:
                input_sequence = line
                continue

    return sequences


def get_file_sequences(filename):
    sequences = []
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequences.append(Sequence(sequence, name=name))

    return sequences
