import numpy as np

AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
ACIDS_LENGTH = len(AMINO_ACIDS)


def code_sequence(sequence, window_size, data=False):
    if not window_size % 2:
        raise Exception("Window size must be odd number!")

    half = int(window_size / 2)
    padded_sequence = "-" * half
    coded_sequence = padded_sequence + str(sequence.input) + padded_sequence
    output_sequence = padded_sequence + str(sequence.output) + padded_sequence

    codes = []
    [codes.extend(one_hot_encoding(a)) for a in coded_sequence]
    codes = np.array(codes)

    signal_sites = [is_signal(o) for o in output_sequence]
    cleavage_sites = [is_cleavage(o) for o in output_sequence]

    input_data, signal_data, cleavage_data = [], [], []
    for i in range(half, len(coded_sequence) - half):
        input_data.append(codes[(i - half) * (ACIDS_LENGTH + 1): (i + half + 1) * (ACIDS_LENGTH + 1)])
        if data:
            signal_data.append(signal_sites[i])
            cleavage_data.append(cleavage_sites[i])

    if data:
        return input_data, signal_data, cleavage_data
    else:
        return input_data


def one_hot_encoding(amino_acid):
    code = np.zeros(ACIDS_LENGTH + 1)
    if amino_acid in AMINO_ACIDS:
        code[AMINO_ACIDS.index(amino_acid)] = 1
    else:
        code[ACIDS_LENGTH] = 1

    return code


def is_cleavage(output):
    return 0 if output != "C" else 1


def is_signal(output):
    return 0 if output not in "nhcC" else 1
