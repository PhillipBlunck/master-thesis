#!/usr/bin/python3

#############################################################################
##                                                                         ##
## trainsvm.py: Train a SVM and display the metrics.                       ##
##                                                                         ##
## Phillip Blunck, 2021-10-15                                              ##
##                                                                         ##
#############################################################################

NAME = "training-svm"

interactive = False # Flag for interactive mode

DEVELOP = False # Flag for develop-mode. If active second test set is used

DEBUG_EXTR = False # Debug flag for extraction loop

EXPORT_LATEX = True # Flag for latetx pdf export

SAVE_MODEL = True

#############################################################################

import datetime
import matplotlib
if not interactive: matplotlib.use("agg")
import matplotlib.pyplot
import numpy
import pandas
import subprocess
import sys

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

HIST_LEN = 19 # historical current values (in seconds)

if DEVELOP:                    # Note that sampling rate is 10 Hz
    TRAIN_TEST_IDX = 50000     # End index for training/test set (raw)
    TEST_TWO_IDX = [62500, -1] # -1 equals all items until end of list

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

def relative_error_percent(exact_arr, approx_arr):
    """Calculate the relative error between the approximation
       and the exact values.
    """
    rel_err = numpy.array(
        [abs((exact - approx) / exact) * 100 \
         for approx, exact in zip(approx_arr, exact_arr)]
    )        
    return rel_err
    
#############################################################################

def scale_minmax(input_arr, min, max):
    """ Transform array by scaling each value between zero and one.
    """
    scaled = numpy.array(
        [(x - min) / (max - min) for x in input_arr]
    )
    return scaled

#############################################################################

def prepare_data(data, window):
    """ Transforms data array to feature array for svm.
        The generated array contains samples of the following features:
                current history, temperature, SoC, voltage.
        Amount:  (HIST_LEN + 1),           1,   1,       1.
    """
    # create new feature array with current+history, temperature, SoC
    arr_size = int(((window / 10) + 1) + 3)
    data_new = numpy.empty((0, arr_size))
    for index in range(window, len(data[:]), 10):          
        # add currenthistory
        data_store = (data[(index - window):(index + 1):10, 3])
        if DEBUG_EXTR:
            date_dump = data[(index - window):(index + 1):10, 1]
            print("Date dump:", date_dump)
            print("Stored current values:", data_store)
        # add temperature
        data_store = numpy.append(data_store,
            data[index, 4]
        )
        # add soc
        data_store = numpy.append(data_store,
            data[index, 5]
        )
        # add voltage
        data_store = numpy.append(data_store,
            data[index, 2]
        )
        # combine stored data
        data_new = numpy.vstack((data_new, data_store))
    # return preprocessed data array
    return data_new
        
#############################################################################

def plottrainingresult(model, logger, signals, prediction, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.title(
        "Current history SVM model evaluation with test set"
    )
    # show comparison between prediction and test set
    matplotlib.pyplot.plot(prediction[:], '--', color='b',
        label="Prediction"
    )
    matplotlib.pyplot.plot(test_y[:], '-', color='m', label="Test set")
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
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")
    
#############################################################################

def plottrainingresultLatex(model, logger, signals, prediction, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    # show comparison between prediction and test set
    matplotlib.pyplot.plot(prediction[:], '--', color='b',
        label="Prediction"
    )
    matplotlib.pyplot.plot(test_y[:], '-', color='m', label="Test set")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-training-result.pdf"
    )
    
#############################################################################

def plotvalidationsetresult(logger, signals, prediction, test_x, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.subplot(4, 1, 1)
    matplotlib.pyplot.title(
        "Current history SVM model evaluation with validation set"
    )
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(test_x[:, HIST_LEN])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.subplot(4, 1, 2)
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[2]}")
    matplotlib.pyplot.plot(test_x[:, (HIST_LEN + 1)])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.subplot(4, 1, 3)
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(test_x[:, (HIST_LEN + 2)])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.subplot(4, 1, 4)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(test_y[:], '-', color='m', label="Validation set")
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
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")
    #############################################################################

def plot_rel_err_percent(logger, rel_err):
    """Display relative error of validation set and save figure as png.
    """
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.title(
        "Current history SVM model evaluation with validation set"
    )
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"Relative error / %")
    matplotlib.pyplot.plot(rel_err[:], '-', color='r')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.text(
        0.02, 0.02, version.get(f"Logger {logger}"),
        transform=matplotlib.pyplot.gcf().transFigure
    )
    # plot score
    matplotlib.pyplot.text(
        0.65, 0.02, f"Maximum relative error: {max(rel_err)}",
        transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")
    
#############################################################################

def plot_rel_err_percentLatex(logger, rel_err):
    """Display relative error of validation set and save figure as png.
    """
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"Relative error / %")
    matplotlib.pyplot.plot(rel_err[:], '-', color='r')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{logger}-{number:02d}-validation-result.pdf"
    )
    
#############################################################################

def plotvalidationsetresultLatex(logger, signals, prediction, test_x, test_y,
        score, mae
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    matplotlib.pyplot.subplot(4, 1, 1)
    #matplotlib.pyplot.title(
    #    "Current history model evaluation with second test set"
    #)
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[1]}")
    matplotlib.pyplot.plot(test_x[:, HIST_LEN])
    matplotlib.pyplot.grid(True)
    #matplotlib.pyplot.xlim([1381, 1581])
    matplotlib.pyplot.subplot(4, 1, 2)
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[2]}")
    matplotlib.pyplot.plot(test_x[:, (HIST_LEN + 1)])
    matplotlib.pyplot.grid(True)
    #matplotlib.pyplot.xlim([1381, 1581])
    matplotlib.pyplot.subplot(4, 1, 3)
    #matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[3]}")
    matplotlib.pyplot.plot(test_x[:, (HIST_LEN + 2)])
    matplotlib.pyplot.grid(True)
    #matplotlib.pyplot.xlim([1381, 1581])
    matplotlib.pyplot.subplot(4, 1, 4)
    matplotlib.pyplot.plot(
        prediction[:], '--', color='b', label="Prediction"
    )
    matplotlib.pyplot.plot(test_y[:], '-', color='m', label="Validation set")
    matplotlib.pyplot.grid(True)
    #matplotlib.pyplot.xlim([1381, 1581])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("ith sample")
    matplotlib.pyplot.ylabel(f"{signals[0]}")
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 6 + 1
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
        print("Data:", data.shape)
        # get logger numbers
        loggers = {* data[:, 0]}
        
        for logger in sorted(loggers):
            # extract data and preprocess
            logdata = numpy.array([d for d in data if d[0] == logger])
            # Start stopwatch
            tim_start = process_time()
            # feature extraction for current history
            data_new = prepare_data(logdata, (HIST_LEN * 10))
            # Stop stopwatch
            tim_stop = process_time()
            print("Elapsed time in seconds:", (tim_stop - tim_start))
            # print shape of new data array
            print("Data_new:", data_new.shape)
            if DEBUG_EXTR: print("Data_new[0]:", data_new[0])
        
            # split data into training+test set and second test set
            if DEVELOP:
                X, y = data_new[:TRAIN_TEST_IDX, :(HIST_LEN + 3)],\
                       data_new[:TRAIN_TEST_IDX, (HIST_LEN + 3)]
                if TEST_TWO_IDX[1] == -1:
                    Xt, yt = data_new[TEST_TWO_IDX[0]:, :(HIST_LEN + 3)],\
                             data_new[TEST_TWO_IDX[0]:, (HIST_LEN + 3)]
                else:
                    Xt, yt = data_new[TEST_TWO_IDX[0]:TEST_TWO_IDX[1],
                                    :(HIST_LEN + 3)],\
                             data_new[TEST_TWO_IDX[0]:TEST_TWO_IDX[1],
                                     (HIST_LEN + 3)]
            else:
                X, y = data_new[:, :(HIST_LEN + 3)],\
                       data_new[:, (HIST_LEN + 3)]
            print("Inputs: ", X.shape)
            print("Output: ", y.shape)
        
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=1, test_size=0.2, shuffle=False
            )
        
            # normalize data (Min-max feature scaling)
            # train set
            X_train_current = scale_minmax(X_train[:, :(HIST_LEN + 1)],
                MIN_CURRENT, MAX_CURRENT
            )
            print("X_train", X_train.shape)
            print("X_train_current:", X_train_current.shape)
            print("X_train_current[0]:", X_train_current[0, :])
            X_train_temp = scale_minmax(X_train[:, (HIST_LEN + 1)],
                MIN_TEMP, MAX_TEMP
            )
            X_train_soc = scale_minmax(X_train[:, (HIST_LEN + 2)],
                MIN_SOC, MAX_SOC
            )
            print("X_train_soc shape:", X_train_soc.shape)
            print("X_train_soc[0]:", X_train_soc[0])
            X_train_norm = numpy.column_stack(
                (X_train_current, X_train_temp, X_train_soc)
            )
            print("Train set norm:", X_train_norm.shape)
            print("Train set norm:", X_train_norm[0, :])
            # test set
            X_test_current = scale_minmax(X_test[:, :(HIST_LEN + 1)],
                MIN_CURRENT, MAX_CURRENT
            )
            X_test_temp = scale_minmax(X_test[:, (HIST_LEN + 1)],
                MIN_TEMP, MAX_TEMP
            )
            X_test_soc = scale_minmax(X_test[:, (HIST_LEN + 2)],
                MIN_SOC, MAX_SOC
            )
            X_test_norm = numpy.column_stack(
                (X_test_current, X_test_temp, X_test_soc)
            )
            if DEVELOP:
                # second test set
                Xt_current = scale_minmax(Xt[:, :(HIST_LEN + 1)],
                    MIN_CURRENT, MAX_CURRENT
                )
                Xt_temp = scale_minmax(Xt[:, (HIST_LEN + 1)],
                    MIN_TEMP, MAX_TEMP
                )
                Xt_soc = scale_minmax(Xt[:, (HIST_LEN + 2)],
                    MIN_SOC, MAX_SOC
                )
                X_test_sor_norm = numpy.column_stack(
                    (Xt_current, Xt_temp, Xt_soc)
                )
        
            # Set up the model and grid search parameters
            parameters = {'kernel':['rbf'],
                          'C':[1, 2, 4, 6, 8, 10],
                          'gamma':['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
                          'epsilon':[0.1]
            }
            parameters2 = {'kernel':['rbf'],
                          'C':[1, 2, 4, 6, 8, 10],
                          'gamma':['scale', 'auto',\
                                   0.01, 0.1, 1, 2, 4, 6, 8, 10, 12, 14],
                          'epsilon':[0.1, 0.01, 0.001]
            }
            regr = SVR(kernel='rbf', C=8, gamma=0.01, coef0=0.0,
                       tol=0.001, epsilon=0.1, shrinking=True,
                       cache_size=200, verbose=True, max_iter=-1
            )
            # Klass et. al.: the best SVM performance was obtained for a
            # RBF kernel with γ=14, C = 1.0134, and ε = 0.01.
            # use grid search cv
            clf = GridSearchCV(estimator=regr, param_grid=parameters2,
                cv=5, n_jobs=-1, verbose=2
            )
            #fit the model to training set
            regr.fit(X_train_norm, y_train)
            #clf.fit(X_train_norm, y_train)
            #print(sorted(clf.cv_results_.keys()))
            #print(clf.best_params_)
            
            # make predictions
            pred = regr.predict(X_test_norm)
            if DEVELOP:
                pred_s = regr.predict(X_test_sor_norm)
            # evaluate predictions
            print(f"Number of sv: {regr.n_support_}")
            print(f"Features: {regr.n_features_in_}")
            score = regr.score(X_test_norm, y_test)
            print(f"Score: {score}")
            mae = mean_absolute_error(y_test, pred)
            print('MAE: %.3f' % mae)
            mse = mean_squared_error(y_test, pred)
            print('MSE: %.3f' % mse)
            if DEVELOP:
                score2 = regr.score(X_test_sor_norm, yt)
                print(f"Validation score: {score2}")
                mae2 = mean_absolute_error(yt, pred_s)
                print('Validation MAE: %.3f' % mae2)
                mse2 = mean_squared_error(yt, pred_s)
                print('Validation MSE: %.3f' % mse2)
                rel_err = relative_error_percent(yt, pred_s)
        
            # training results
            plottrainingresult(regr, int(logger), signals,
                pred, y_test, score, mae
            )
            if EXPORT_LATEX:
                plottrainingresultLatex(regr, int(logger), signals,
                    pred, y_test, score, mae
                )
            # show test set sorted
            if DEVELOP:
                plotvalidationsetresult(int(logger), signals, pred_s,
                    X_test_sor_norm, yt, score2, mae2
                )
                plot_rel_err_percent(int(logger), rel_err)
                if EXPORT_LATEX:
                    plotvalidationsetresultLatex(int(logger), signals,
                        pred_s, X_test_sor_norm, yt, score2, mae2
                    )
                    plot_rel_err_percentLatex(int(logger), rel_err)
        if interactive: matplotlib.pyplot.show(block = True)
        # persist machine learning model into file
        if SAVE_MODEL:
            datahash = version.hashtext[:7]
            dump(regr, f'models/svm-{datahash}.joblib')
        
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

