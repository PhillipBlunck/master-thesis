#!/usr/bin/python3

#############################################################################
##                                                                         ##
## trainmlpbasic.py: Train an MLP-Network and display the loss curve       ##
##                   and other metrics.                                    ##
##                                                                         ##
## Phillip Blunck, 2021-10-02                                              ##
##                                                                         ##
#############################################################################

NAME = "training-mlpbasic"

interactive = False # Flag for interactive mode

DEVELOP = True # Flag for develop-mode. If active second test set is used

EXPORT_LATEX = True

SAVE_MODEL = False

#############################################################################

import datetime
import matplotlib
if not interactive: matplotlib.use("agg")
import matplotlib.pyplot
import numpy
import pandas
import subprocess
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from time import process_time
from joblib import dump, load

#############################################################################

COMMA = ","
SPACE = " "
EMPTY = ""
DELIMITER = COMMA + SPACE

WIDTH = 1920
HEIGHT = 1080
RES = 100
SIZE = (WIDTH/RES, HEIGHT/RES)

MIN_CURRENT = -200
MAX_CURRENT = 500

MIN_SOC = 0
MAX_SOC = 100

MIN_TEMP = -20
MAX_TEMP = 90

STEP = 10 # step size for data preprocessing

#############################################################################

class version:
   
    hashtext = "unknown"

    def set(filename):
        with open(filename) as hashfile:
            version.hashtext = hashfile.read()

    def get(string=EMPTY):
        # get date
        date = datetime.datetime.now()
        result = f"{date:%Y-%m-%d}"
        result += COMMA + SPACE
        # get git commit hash value
        head = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"]
        )
        head = head.decode("ascii").strip()
        result += "Program" + SPACE + head
        if subprocess.check_output(["git", "status", "--porcelain"]):
            result += SPACE + "modified"
        result += COMMA + SPACE
        # get data hash value
        result += "Data" + SPACE + version.hashtext[:7]
        if string:
            result += COMMA + SPACE
            result += string
        # return complete version string
        return result

#############################################################################

def scale_minmax(input_arr, min, max):
    """ Transform array by scaling each value between zero and one.
    """
    scaled = numpy.array(
        [(x - min) / (max - min) for x in input_arr]
    )
    return scaled

#############################################################################

def plottraintestsplitLatex(logger, X_train, y_train,
                            X_test, y_test, signals
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    #matplotlib.pyplot.title("Train dataset")
    # plot current
    matplotlib.pyplot.subplot(3, 1, 1)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(X_train[:,0])
    matplotlib.pyplot.grid(True)
    # plot SoC
    matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(X_train[:,1])
    matplotlib.pyplot.grid(True)
    # plot voltage
    matplotlib.pyplot.subplot(3, 1, 3)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.plot(y_train[:])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-train-test-split.pdf"
    )
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    #matplotlib.pyplot.title("Test dataset")
    # plot current
    matplotlib.pyplot.subplot(3, 1, 1)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(X_test[:,0])
    matplotlib.pyplot.grid(True)
    # plot SoC
    matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(X_test[:,1])
    matplotlib.pyplot.grid(True)
    # plot volage
    matplotlib.pyplot.subplot(3, 1, 3)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.plot(y_test[:])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-train-test-split.pdf"
    ) 

#############################################################################

def plotaltertestsetLatex(logger, X_test, y_test, signals):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    #matplotlib.pyplot.title("Test dataset")
    # plot current
    matplotlib.pyplot.subplot(3, 1, 1)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(X_test[:,0])
    matplotlib.pyplot.grid(True)
    # plot SoC
    matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(X_test[:,1])
    matplotlib.pyplot.grid(True)
    # plot volage
    matplotlib.pyplot.subplot(3, 1, 3)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.plot(y_test[:])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-alter-test-set.pdf"
    ) 

#############################################################################

def plottrainingresult(model, logger, signals, prediction, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.title("Basic model evaluation with test set")
    matplotlib.pyplot.xlabel("ith epoch")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.plot(model.loss_curve_)
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.grid(True)
    # show comparison between prediction and test set
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(
        test_y[:], '-', color='m', label="Test set"
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.text(
        0.02, 0.02, version.get(f"Logger {logger}"),
        transform=matplotlib.pyplot.gcf().transFigure
    )
    # plot score
    matplotlib.pyplot.text(
        0.65, 0.02, f"Score: {score} MAE: {mae}",
        transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}.png"
    )

#############################################################################

def plottrainingresultLatex(model, logger, signals, prediction, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.xlabel("ith epoch")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.plot(model.loss_curve_)
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.grid(True)
    # show comparison between prediction and test set
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(
        test_y[:], '-', color='m', label="Test set"
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-training-result.pdf"
    )
    #############################################################################
    
def plotvalidationsetresult(logger, signals, prediction, test_x, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.subplot(3, 1, 1)
    matplotlib.pyplot.title(
        "Basic model evaluation with validation set"
    )
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(test_x[:, 0])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(test_x[:, 1])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.subplot(3, 1, 3)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(
        test_y[:], '-', color='m', label="Validation set"
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.text(
        0.02, 0.02, version.get(f"Logger {logger}"),
        transform=matplotlib.pyplot.gcf().transFigure
    )
    # plot score
    matplotlib.pyplot.text(
        0.65, 0.02, f"Score: {score} MAE: {mae}",
        transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}.png"
    )
    
#############################################################################
    
def plotvalidationsetresultLatex(logger, signals, prediction, test_x, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    matplotlib.pyplot.subplot(3, 1, 1)
    #matplotlib.pyplot.title(
    #    "Basic model evaluation with second test set"
    #)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(test_x[:, 0])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlim([1400, 1600])
    matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(test_x[:, 1])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlim([1400, 1600])
    matplotlib.pyplot.subplot(3, 1, 3)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(
        test_y[:], '-', color='m', label="Validation set"
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlim([1400, 1600])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 7 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-validation-result.pdf"
    )
    
#############################################################################

def training(stdin):
    try:
        # read csv from stdin as pandas array
        data = pandas.read_csv(stdin, delimiter=DELIMITER, engine="python")
        # get signal names
        signals = data.columns[2:]
        # convert data into numpy array
        data = data.to_numpy()
        # get logger numbers
        loggers = {* data[:, 0]}
        
        for logger in sorted(loggers):
            # extract data and preprocess
            logdata = numpy.array([d for d in data if d[0] == logger])
            # combine into array of features current, SoC
            if DEVELOP:
                X, y = logdata[:500000:STEP, [3,5]], \
                    logdata[:500000:STEP, 2] #500000
            else:
                X, y = logdata[::STEP, [3,5]], logdata[::STEP, 2] #500000
            print("Inputs: ", X.shape)
            print("Output: ", y.shape)
            # generate train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=1, test_size=0.2, shuffle=False
            )
            # generate alternative test/validation set
            if DEVELOP:
        	    Xt, yt = logdata[625000::STEP, [3,5]], \
        	        logdata[625000::STEP, 2] #625000
            # plot train test split for latex documents
            if EXPORT_LATEX:
                plottraintestsplitLatex(int(logger), X_train, y_train,
                    X_test, y_test, signals
                )
            # plot alternative test split for latex documents
            if DEVELOP and EXPORT_LATEX:
                plotaltertestsetLatex(int(logger), Xt, yt, signals)
            if interactive: matplotlib.pyplot.show(block = True)
            # normalize data (Min-max feature scaling)
            # train set
            X_train_current = scale_minmax(
                X_train[:, 0], MIN_CURRENT, MAX_CURRENT
            )
            X_train_soc = scale_minmax(X_train[:, 1], MIN_SOC, MAX_SOC)
            X_train_norm = numpy.vstack(
                (X_train_current, X_train_soc)
            ).T
            # test set
            X_test_current = scale_minmax(
                X_test[:, 0], MIN_CURRENT, MAX_CURRENT
            )
            X_test_soc = scale_minmax(X_test[:, 1], MIN_SOC, MAX_SOC)
            X_test_norm = numpy.vstack(
                (X_test_current, X_test_soc)
            ).T
            # alterantive test set
            if DEVELOP:
                Xt_current = scale_minmax(Xt[:, 0], MIN_CURRENT, MAX_CURRENT)
                Xt_soc = scale_minmax(Xt[:, 1], MIN_SOC, MAX_SOC)
                X_test_sor_norm = numpy.vstack((Xt_current, Xt_soc)).T
            # Set up the model 12, 10 (2 features, one output)
            regr = MLPRegressor(
                hidden_layer_sizes=(5, 5), activation='relu',
                solver='sgd', alpha=0.0001, learning_rate_init=0.000001,
                momentum=0.9, random_state=1, max_iter=5000, verbose=True,
                batch_size=128, early_stopping=True
            )
            # Start stopwatch
            tim_start = process_time()
            # fit the model to training set
            regr.fit(X_train_norm, y_train) 
            # Stop stopwatch
            tim_stop = process_time()
            print("Elapsed time in seconds:", (tim_stop - tim_start))
            # make predictions on test set
            pred = regr.predict(X_test_norm)
            # make predictions on alternative test set
            if DEVELOP:
                pred_s = regr.predict(X_test_sor_norm)
            
            # evaluate predictions
            print(f"Number of layers: {regr.n_layers_}")
            print(f"Number of features: {regr.n_features_in_}")
            score = regr.score(X_test_norm, y_test)
            print(f"Score: {score}")
            mae = mean_absolute_error(y_test, pred)
            print('MAE: %.3f' % mae)
            
            # evaluate predictions alternative test set
            if DEVELOP:
                score2 = regr.score(X_test_sor_norm, yt)
                print(f"Test2 Score: {score2}")
                mae2 = mean_absolute_error(yt, pred_s)
                print('Test2 MAE: %.3f' % mae2)
        
            # show loss curve
            plottrainingresult(regr, int(logger), signals, pred, y_test,
                score, mae
            )
            if EXPORT_LATEX:
                plottrainingresultLatex(regr, int(logger), signals, pred,
                    y_test, score, mae
                )
            # show test set sorted
            if DEVELOP:
                plotvalidationsetresult(int(logger), signals, pred_s,
                    X_test_sor_norm, yt, score2, mae2
                )
                if EXPORT_LATEX:
                    plotvalidationsetresultLatex(int(logger), signals,
                        pred_s, X_test_sor_norm, yt, score2, mae2
                    )
        if interactive: matplotlib.pyplot.show(block = True)
        # persist machine learning model into file
        if SAVE_MODEL:
            datahash = version.hashtext[:7]
            dump(regr, f'models/mlpbasic-{datahash}.joblib')
            
        result = 0
    except pandas.errors.EmptyDataError:
        print("# No data found.", file=sys.stderr)
        result = 2
    return result

#############################################################################

def main(argv):
    if len(argv) == 2:
        version.set(argv[1])
        result = training(sys.stdin)
    else:
        program = argv[0] if argv else __file__
        print(f"Usage: <pipeline> | {program} <hashfile> <output_signals>",
            file=sys.stderr
        )
        result = 1
    return result

#############################################################################

if __name__ == "__main__":
    STATUS = main(sys.argv)
    sys.exit(STATUS)

#############################################################################

