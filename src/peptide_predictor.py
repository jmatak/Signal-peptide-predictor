#!/usr/bin/env python3
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

import datetime
import argparse
import os
import sys
import time

sys.path.insert(0, '../src')
import dataset_manager as data_manager
import neural_network as nn


TEST_PATH = "../data/"
MODEL_PATH = '../models/'
PLOT_PATH = 'plots/'
MODEL_NAME = 'model.hdf5'
LOG_NAME = "signal_process.log"
HISTORY_NAME = "history.png"
THRESHOLD = 0.2


def get_directory_name():
    return 'model_{0:%Y-%m-%d_%H:%M:%S}/' \
        .format(datetime.datetime.now())


def save_history(history, filename):
    plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model accuracy')
    plt.ylabel('Mean square error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename)


def save_log(window_size, architecture, test_eval, precision, exec_time, filename):
    with open(filename, "w") as f:
        f.write("WINDOW_SIZE:{}\n".format(window_size))
        f.write("ARCHITECTURE:{}\n".format(architecture))
        f.write("EXEC_TIME:{}\n".format(exec_time))
        f.write("TEST_EVAL:{}\n".format(test_eval))
        f.write("PRECISION:{}\n".format(precision))


def save_results(results, plot, filename):
    if plot:
        os.makedirs(PLOT_PATH, exist_ok=True)

    with open(filename, "w") as f:
        for sequence, prediction, is_signal in results:
            if plot:
                plt.figure()
                plt.plot(range(len(prediction)), prediction)
                plt.savefig(PLOT_PATH + sequence.name + "-fig.png")

            f.write('{}:{}\n'.format(sequence.name, is_signal))


def loading_screen(i, length):
    print("Processed: {}/{}".format(i, length), file=sys.stderr)


def train_action(window_size, hidden_layers):
    print("Fetching data...", file=sys.stderr)
    data = data_manager.fetch_protein_data("../data/training_data")
    print("Done! Preparing for training...", file=sys.stderr)
    input_data, signal_data = data_manager.prepare_dataset(data, window_size, loading_screen)
    print("Done! Train and test splitting...", file=sys.stderr)
    train_input, test_input = data_manager.train_test_split(input_data, signal_data)
    print("Done", file=sys.stderr)

    print("Building signal model", file=sys.stderr)

    # Input layer
    architecture = [input_data[0].shape[0]]
    # Hidden layer
    architecture.extend(hidden_layers)
    # Output layer
    architecture.append(2)

    model = nn.build_model(architecture)

    directory = MODEL_PATH + get_directory_name()
    try:
        os.makedirs(directory)
    except:
        print("There was an error while creating model's sub-directory.", file=sys.stderr)
        exit(1)

    start_time = time.time()
    history = nn.train_model(model, train_input[0], train_input[1],
                             filename=directory + MODEL_NAME)
    exec_time = time.time() - start_time

    test_eval = model.evaluate(test_input[0], test_input[1])

    results, precision = test_sequences(data, model, window_size, eval=True)
    save_history(history, directory + HISTORY_NAME)
    save_log(window_size, architecture, test_eval, precision, exec_time, directory + LOG_NAME)
    print("Done.", file=sys.stderr)


def predict_signal(result):
    index = None
    signals = result[:, 0]
    cleavage = result[:, 1]
    for i, c in enumerate(cleavage):
        if c > THRESHOLD:
            index = i
            break
    if not index: return False

    signals = signals[:index]
    maximum = len(signals)
    count = 0
    for s in signals:
        if s > THRESHOLD:
            count += 1

    return count / maximum > 0.9


def test_sequences(sequences, model, window_size, eval=False):
    results = []
    count = 0
    count_signal = 0
    for s in sequences:
        result = model.predict(data_manager.prepare_example(s, window_size))
        is_signal = predict_signal(result)
        results.append((s, result, is_signal))
        if eval:
            if (s.label == 1 and is_signal == True) or (s.label == -1 and is_signal == False): count += 1

        if is_signal: count_signal += 1
        print("{}:{}".format(s.name, is_signal), file=sys.stderr)

    print("{}:{}/{}".format(len(results), count_signal, len(results) - count_signal), file=sys.stderr)
    if not eval: return results
    else: return results, count/len(results)


def test_action(window_size, model_dir, test_file, output_file, plot):
    model = None
    try:
        model = load_model(MODEL_PATH + model_dir + '/' + MODEL_NAME)
    except:
        print("No such model! Please specify model with -m flag.", file=sys.stderr)
        exit(1)

    sequences = []
    try:
        sequences = data_manager.get_file_sequences(TEST_PATH + test_file)
    except:
        print("No testing data, please put it in /data folder.", file=sys.stderr)
        exit(1)

    try:
        results = test_sequences(sequences, model, window_size)


    except:
        print("Model was built with different window size.", file=sys.stderr)
        exit(1)

    save_results(results, plot, output_file)


def evaluate_action(window_size, model_dir):
    data = data_manager.fetch_protein_data("../data/training_data")
    results, precision = test_sequences(data, load_model(MODEL_PATH + model_dir + '/' + MODEL_NAME), window_size,
                                        eval=True)
    print("Precision:{}".format(precision))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Peptide classifier program. Program requires to have /data and /models folders.")
    parser.add_argument("--train", action="store_true",
                        help="Use this to train model.")
    parser.add_argument("--test", action="store_true",
                        help="Use this to do tests with specific model. [-m, -w]")
    parser.add_argument("--evaluate", action="store_true",
                        help="Use this to do evaluate model with set it was trained on [-m, -w]")
    parser.add_argument("-w", "--window_size", type=int, default=21,
                        help="Define used window size, default is 21.")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Use this to plot each sequence while testing.")
    parser.add_argument("-m", "--model",
                        help="Define model's directory, used for testing.")
    parser.add_argument("-o", "--output", default="output",
                        help="Define output file name.")
    parser.add_argument("-f", "--test_file", default="test.fa",
                        help="Define test file name, stored in /data.")
    parser.add_argument("-a", "--architecture", nargs='*', default=[128, 64, 16],
                        help="Define architecture of model. Provide only hidden layers, for example: -a 8 4 will yield"
                             "architecture INPUTx8x4xOUTPUT.")

    args = parser.parse_args()

    if args.train:
        train_action(args.window_size, args.architecture)

    if args.test:
        test_action(args.window_size, args.model, args.test_file, args.output, args.plot)

    if args.evaluate:
        evaluate_action(args.window_size, args.model)
