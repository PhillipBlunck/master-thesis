#!/usr/bin/python3

#############################################################################
##                                                                         ##
## simulation.py: Simulation of an HPPC-Test to calculate the discharge    ##
##                resistance of a lithium ion battery.                     ##
##                                                                         ##
##                The discharge pulse starts at 10 seconds with the length ##
##                of 10 seconds.                                           ##
##                                                                         ##
## Phillip Blunck, 2021-10-20                                              ##
##                                                                         ##
#############################################################################

NAME = "simulation"

interactive = False # Flag for interactive mode

DEBUG = False # Debug flag

EXPORT_LATEX = True

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
from sklearn.svm import SVR

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

def plot_virtual_hppc_test(prediction, data_x, resistance, string=EMPTY):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.subplot(4, 1, 1)
    matplotlib.pyplot.title(
        "Virtual HPPC test to calculate 10s discharge resistance" \
        + SPACE + "(" + string + ")"
    )
    # current
    matplotlib.pyplot.ylabel("STROM / A")
    matplotlib.pyplot.plot(data_x[:, HIST_LEN], '-')
    matplotlib.pyplot.grid(True)
    # temperature
    matplotlib.pyplot.subplot(4, 1, 2)
    matplotlib.pyplot.ylabel("TCAVG / °C")
    matplotlib.pyplot.plot(data_x[:, (HIST_LEN + 1)], '-')
    matplotlib.pyplot.grid(True)
    # state of charge
    matplotlib.pyplot.subplot(4, 1, 3)
    matplotlib.pyplot.ylabel(f"STTOFCHRG / %")
    matplotlib.pyplot.plot(data_x[:, (HIST_LEN + 2)], '-')
    matplotlib.pyplot.ticklabel_format(useOffset=False)
    matplotlib.pyplot.grid(True)
    # predicted voltage
    matplotlib.pyplot.subplot(4, 1, 4)
    matplotlib.pyplot.plot(
        prediction[:], '-'
    )
    # plot voltage points
    matplotlib.pyplot.plot(9,
        prediction[9], 'o', color='red',
        label=f"U_0: {prediction[9]:.3f} V",
        markerfacecolor='none'
    )
    matplotlib.pyplot.plot(20,
        prediction[20], 'o', color='orange',
        label=f"U_10: {prediction[20]:.3f} V",
        markerfacecolor='none'
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.ylabel("UBAT / V")
    matplotlib.pyplot.text(
        0.02, 0.02, version.get(EMPTY),
        transform=matplotlib.pyplot.gcf().transFigure
    )
    # plot discharge resistance
    matplotlib.pyplot.text(
        0.65, 0.02, f"10s discharge resistance: {resistance:.6f} Ohm",
        transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{number:02d}-virtual-hppc-test-{string}.png"
    )

#############################################################################

def plot_virtual_hppc_test_Latex(prediction, data_x, resistance,
        string=EMPTY
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    matplotlib.pyplot.subplot(4, 1, 1)
    # current
    matplotlib.pyplot.ylabel("STROM / A")
    matplotlib.pyplot.plot(data_x[:, HIST_LEN], '-')
    matplotlib.pyplot.grid(True)
    # temperature
    matplotlib.pyplot.subplot(4, 1, 2)
    matplotlib.pyplot.ylabel("TCAVG / °C")
    matplotlib.pyplot.plot(data_x[:, (HIST_LEN + 1)], '-')
    matplotlib.pyplot.grid(True)
    # state of charge
    matplotlib.pyplot.subplot(4, 1, 3)
    matplotlib.pyplot.ylabel(f"STTOFCHRG / %")
    matplotlib.pyplot.plot(data_x[:, (HIST_LEN + 2)], '-')
    matplotlib.pyplot.ticklabel_format(useOffset=False)
    matplotlib.pyplot.grid(True)
    # predicted voltage
    matplotlib.pyplot.subplot(4, 1, 4)
    matplotlib.pyplot.plot(
        prediction[:], '-'
    )
    # plot voltage points
    matplotlib.pyplot.plot(9,
        prediction[9], 'o', color='red',
        label=f"U_0: {prediction[9]:.3f} V",
        markerfacecolor='none'
    )
    matplotlib.pyplot.plot(20,
        prediction[20], 'o', color='orange',
        label=f"U_10: {prediction[20]:.3f} V",
        markerfacecolor='none'
    )
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.ylabel("UBAT / V")
    matplotlib.pyplot.tight_layout()
    # plot discharge resistance
    #matplotlib.pyplot.text(
    #    0.65, 0.02, f"Discharge resistance: {resistance:.6f} Ohm",
    #    transform=matplotlib.pyplot.gcf().transFigure
    #)
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{number:02d}-virtual-hppc-test-{string}.pdf"
    )
    
#############################################################################

def plot_voltage_comparison(prediction_A, prediction_B,
        modelNameA=EMPTY, modelNameB=EMPTY
    ):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    matplotlib.pyplot.title(
        "HPPC test voltage comparison between two machine learning models"
    )
    # model A
    matplotlib.pyplot.plot(prediction_A[:], '-', color='b', label=modelNameA)
    # model B
    matplotlib.pyplot.plot(prediction_B[:],'-', color='r',label=modelNameB)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.ylabel("UBAT / V")
    matplotlib.pyplot.text(
        0.02, 0.02, version.get(EMPTY),
        transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{number:02d}-voltage-comparison.png"
    )
    
#############################################################################

def plot_voltage_comparison_Latex(prediction_A, prediction_B,
        modelNameA=EMPTY, modelNameB=EMPTY
    ):
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    # model A
    matplotlib.pyplot.plot(prediction_A[:], '-', color='b', label=modelNameA)
    # model B
    matplotlib.pyplot.plot(prediction_B[:],'-', color='r',label=modelNameB)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.ylabel("UBAT / V")
    matplotlib.pyplot.tight_layout()
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as pdf
    number = (figure.number - 1) % 6 + 1
    matplotlib.pyplot.savefig(
        f"{NAME}-{number:02d}-voltage-comparison.pdf"
    )
#############################################################################

def generate_data(hist_len, duration, max_current,
                  temp, soc_start, capacity
    ):
    """Generate a dataset for virtual HPPC test.
       The resulting feature array consists of a current curve
       with a pulse length of 10 seconds.
       The state of charge is calculated on the fly with the specific
       capacity of the battery and the current. The starting point of the SoC 
       can also be set to a specific value.
       The temperature is set to a constant value.
    """
    # generate empty feature array
    arr_size = int((hist_len + 1) + 2)
    data_new = numpy.empty((duration, arr_size))
    # set integral of current to zero (coulomb counting is used)
    integ_curr = 0
    # depth of discharge
    DOD = 80 
    # set values with a set length greater than 20 seconds
    for index in range(duration):
        # current   
        if (index >= 10) and (index <= 20): # discharge pulse
            data_new[index, hist_len] = max_current
        else:
            data_new[index, hist_len] = 0 # fresh value
        # current history
        if index == 0:
            data_new[index, 0:hist_len] = 0
        else:
            for idx_past in range(hist_len):
                data_new[index, idx_past] = \
                data_new[(index - 1), (idx_past + 1)]
        # temperature
        data_new[index, (hist_len + 1)] = temp
        # soc
        integ_curr += data_new[index, hist_len]
        soc = soc_start - (integ_curr / (DOD * capacity))
        #soc = (soc_start / 100) - (( 1 / (0.8 * capacity)) * integ_curr)
        data_new[index, (hist_len + 2)] = soc
    return data_new

#############################################################################

def calc_disc_resistance(U_0, U_10, I_max):
    """Discharge resistance is calculated by two voltage values at time 0s
    and time 10s. They represent values at the beginning of the current
    discharge pulse and 1 s after the pulse.
    """
    disc_resistance = (U_0 - U_10) / I_max
    return disc_resistance

#############################################################################

def virtual_hppc_test():
    """Use a trained ML-model for virtual HPPC test and calculate the
       discharge resistance to determine the SoH of the battery.
    """
    # load trained machine learning models
    datahash = version.hashtext[:7]
    regrMLP = load(f'models/mlphist-{datahash}.joblib')
    regrSVM = load(f'models/svm-{datahash}.joblib')
    # set battery and virtual test parameter
    DURATION = 61 # hppc test duration in seconds [0 - (DURATION-1)]
    MAX_DISC_CURRENT = 135.3 # maximum discharge current
    TEMPERATURE = 25 # avg cell temperature
    START_SOC = 100 # start value of SoC
    CAPACITY = 40 * 3600 # Battery capacity in Ampere seconds (Ah * 3600 s)
    # generate dataset for virtual HPPC test
    virtualdata = generate_data(HIST_LEN, DURATION, MAX_DISC_CURRENT,
        TEMPERATURE, START_SOC, CAPACITY
    )
    if DEBUG:
        print("Generated virtual dataset:")
        print(virtualdata.shape)
        print(virtualdata[9:22])
    # normalize data (min-max feature scaling)
    feature_current = scale_minmax(virtualdata[:, :(HIST_LEN + 1)],
        MIN_CURRENT, MAX_CURRENT
    )
    feature_temp = scale_minmax(virtualdata[:, (HIST_LEN + 1)],
        MIN_TEMP, MAX_TEMP
    )
    feature_soc = scale_minmax(virtualdata[:, (HIST_LEN + 2)],
        MIN_SOC, MAX_SOC
    )
    virtualdata_norm = numpy.column_stack(
        (feature_current, feature_temp, feature_soc)
    )
    if DEBUG:
        print("Normalized virtual dataset:")
        print(virtualdata_norm.shape)
        print(virtualdata_norm[9:22])
    # predict voltage with ML-models and generated feature array
    predMLP = regrMLP.predict(virtualdata_norm)
    predSVM = regrSVM.predict(virtualdata_norm)
    if DEBUG: print(predMLP.shape)
    # calculate discharge resistance from current pulse
    resistanceMLP = calc_disc_resistance(predMLP[9], predMLP[20],
        MAX_DISC_CURRENT
    )
    print(f"Discharge resistance (MLP): {resistanceMLP} Ohm")
    resistanceSVM = calc_disc_resistance(predSVM[9], predSVM[20],
        MAX_DISC_CURRENT
    )
    print(f"Discharge resistance (SVM): {resistanceSVM} Ohm")
    # plot virtual hppc test
    plot_virtual_hppc_test(predMLP, virtualdata, resistanceMLP, "MLPhist")
    plot_virtual_hppc_test(predSVM, virtualdata, resistanceSVM, "SVM")
    plot_voltage_comparison(predMLP, predSVM, "MLPhist", "SVM")
    if EXPORT_LATEX:
        plot_virtual_hppc_test_Latex(predMLP, virtualdata, resistanceMLP,
            "MLPhist"
        )
        plot_virtual_hppc_test_Latex(predSVM, virtualdata, resistanceSVM,
            "SVM"
        )
        plot_voltage_comparison_Latex(predMLP, predSVM, "MLPhist", "SVM")
    if interactive: matplotlib.pyplot.show(block = True)

#############################################################################

def main(argv):
    if len(argv) == 2:
        version.set(argv[1])
        result = virtual_hppc_test()
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

