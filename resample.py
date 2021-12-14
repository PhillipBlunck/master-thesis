#!/usr/bin/python3

#############################################################################
##                                                                         ##
## resample.py: Resample microsecond signal section to set sampling rate   ##
##              and print out results as csv-file.                         ##
##                                                                         ##
## Phillip Blunck, 2021-10-06                                              ##
##                                                                         ##
#############################################################################

NAME = "resample"

interactive = False

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

import math
import scipy
import scipy.signal
import scipy.interpolate

#############################################################################

COMMA = ","
EMPTY = ""
SPACE = " "
DELIMITER = COMMA + SPACE

WIDTH = 1920
HEIGHT = 1080
RES = 100
SIZE = (WIDTH/RES, HEIGHT/RES)


#############################################################################

class version:
   
   hashtext = "unknown"

   def set(filename):
      """Set the hash of the used dataset.
      """
      with open(filename) as hashfile:
         version.hashtext = hashfile.read()

   def get(string=EMPTY):
      """This function returns version information of the project.
      The returned string contains: the current date, the hash of
      the current commit and the hash of the used dataset.
      In case of modification of the project source data, the commit hash
      is marked as modified.
      """
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
      if subprocess.check_output (["git", "status", "--porcelain"]):
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

def writecsvheader(signals):
   """The function prints out a csv-file header for specific signalnames
   and adds an empty line afterwards
   """
   header = "Logger" + COMMA + SPACE + "Timestamp"
   for signal in signals:
      header += COMMA + SPACE
      header += signal + SPACE
   print(header)
   print()
    
#############################################################################

def writecsvdata(logger, time, signals, data):
    """Print signal values with delimiter to the standard output.
    """
    for idx, val in enumerate(time):
        line = f"{logger}" + COMMA + SPACE + f"{val:.1f}"
        for idx_s, sig in enumerate(signals):
            line += COMMA + SPACE
            if "UBAT" in sig:
                line += f"{data[idx, idx_s]:.1f}" 
            elif "STROM" in sig:
                line += f"{data[idx, idx_s]:.3f}"
            else:
                line += f"{int(data[idx, idx_s])}"
        print(line)

#############################################################################

def plotdifference(logger, time_o, data_o,
                   time_r, data_r, signals, t_range
    ):
    """Display section of the original and the resampled
       signal sections in one plot.
    """
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    for num, signal in enumerate(signals):
        matplotlib.pyplot.subplot(len(signals), 1, num+1)
        if num == 0: matplotlib.pyplot.title(
            "Resampled signal section detailed"
        )
        matplotlib.pyplot.ylabel(signal)
        matplotlib.pyplot.plot(time_o[num][:],
            data_o[num][:], '.-',
            label='Original'
        )
        matplotlib.pyplot.plot(time_r[:],
            data_r[:, num], '*-',
            label='Resampled'
        )
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.legend()
        matplotlib.pyplot.xlim(t_range)
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.text(
      0.02, 0.02, version.get(f"Logger {logger}"),
      transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 5 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")
        
#############################################################################

def plotdifferenceLatex(logger, time_o, data_o,
                   time_r, data_r, signals, t_range
    ):
    """Display section of the original and the resampled
       signal sections in one plot and save it as pdf for LaTex documents.
    """
    figure = matplotlib.pyplot.figure(figsize=(8, 6), dpi=RES)
    for num, signal in enumerate(signals):
        matplotlib.pyplot.subplot(len(signals), 1, num+1)
        #if num == 0: matplotlib.pyplot.title(
        #    "Resampled signal section detailed"
        #)
        matplotlib.pyplot.ylabel(signal)
        matplotlib.pyplot.plot(
            time_o[num][:],
            data_o[num][:], 'o-',
            label='Original'
        )
        matplotlib.pyplot.plot(time_r[:],
            data_r[:, num], '*--',
            label='Resampled'
        )
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlim(t_range)
        if num == 0: matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.tight_layout()
    #if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 5 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.pdf")

#############################################################################

def plotsection(logger, time, data, signals, string=EMPTY):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    for num, signal in enumerate(signals):
        matplotlib.pyplot.subplot(len(signals), 1, num+1)
        if (num == 0) and (string): matplotlib.pyplot.title(string)
        matplotlib.pyplot.ylabel(signal)
        matplotlib.pyplot.plot(time[:], data[:, num], '.')
        matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.text(
      0.02, 0.02, version.get(f"Logger {logger}"),
      transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 5 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")

#############################################################################

def plotsectiontiny(logger, time, data, signals, t_range):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    for num, signal in enumerate(signals):
        matplotlib.pyplot.subplot(len(signals), 1, num+1)
        if num == 0: matplotlib.pyplot.title("Signal section detailed")
        matplotlib.pyplot.ylabel(signal)
        matplotlib.pyplot.plot(time[:], data[:, num], '.')
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlim(t_range)
    matplotlib.pyplot.xlabel("time (s)")
    matplotlib.pyplot.text(
      0.02, 0.02, version.get(f"Logger {logger}"),
      transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 5 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")

#############################################################################

def plot_raw(logger, time, data, signals, string=EMPTY):
    figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)
    for num, signal in enumerate(signals):
        x = time
        y = data[:, num]
        matplotlib.pyplot.subplot(len(signals), 1, num+1)
        if (num == 0) and (string): matplotlib.pyplot.title(string)
        matplotlib.pyplot.ylabel(signal)
        matplotlib.pyplot.plot(x, y, '.')
        matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel("Index of value")
    matplotlib.pyplot.text(
      0.02, 0.02, version.get(f"Logger {logger}"),
      transform=matplotlib.pyplot.gcf().transFigure
    )
    if interactive: matplotlib.pyplot.show(block = False)
    # saving figure as png
    number = (figure.number - 1) % 5 + 1
    matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")

#############################################################################

def getsection(data, lo_lim, up_lim):
    """Returns indices of time limits for signal section.
    """
    section = []
    for idx, value in enumerate(data):
        if (data[idx, 1] > float(lo_lim)) and (not section):
            section.append(idx)
            continue
        if (data[idx, 1] > float(up_lim)) and (section):
            section.append(idx - 1)
            break
    return section

#############################################################################

def resample(stdin, lo_lim, up_lim, sampling_rate):
    try:
        # read csv
        data = pandas.read_csv(stdin, delimiter=DELIMITER, engine="python")
        # get signal names
        signals = data.columns[2:]
        # convert csv data to numpy array
        data = data.to_numpy()
        
        # print out header of csv-file
        writecsvheader(signals)
        # set range for detailed plots
        t_range = [10576, 10577.5]
        
        # plot signals for each logger
        loggers = {* data[:, 0]}
        for logger in sorted(loggers):
            logdata = numpy.array([d for d in data if d[0] == logger])
            # get indices of signal section
            section = getsection(logdata, lo_lim, up_lim)
            # collect signal section out of logdata
            sect_x = logdata[section[0]:(section[1] + 1), 1]
            sect_y = logdata[section[0]:(section[1] + 1), 2:]
            # show signal section
            plotsection(int(logger), sect_x, sect_y, signals,
                "Signal section"
            )
            # show detailed plot of signal section
            plotsectiontiny(int(logger), sect_x, sect_y, signals,
                t_range
            )
            # resample signal section
            # set intervals to whole seconds
            itv_1 = math.ceil(logdata[section[0], 1])
            itv_2 = int(logdata[section[1], 1])
            
            # get all non NaN values
            collect_x = []
            collect_y = []
            for col, sig in enumerate(signals):
                temp_x = []
                temp_y = []
                for idx, val in enumerate(sect_y[:, col]):
                    if not math.isnan(val):
                        temp_x.append(sect_x[idx])
                        temp_y.append(val)
                collect_x.append(temp_x)
                collect_y.append(temp_y)
                        
            # interpolate all signals into range of signal section
            sr_scnds = 1/int(sampling_rate) # set sampling rate in seconds
            rsmpl_t = numpy.arange(itv_1, itv_2, sr_scnds)
            rsmpl_y = []
            for idx, vals in enumerate(collect_x):
                intp_f = scipy.interpolate.interp1d(collect_x[idx][:],
                    collect_y[idx][:]
                )
                rsmp_val = intp_f(rsmpl_t)
                rsmpl_y.append(rsmp_val)
            rsmpl_y = numpy.array(rsmpl_y).T
            # show resampled signal section
            plotsection(int(logger), rsmpl_t, rsmpl_y, signals,
                "Resampled signal section"
            )
            # show detailed differences between original and resampled
            plotdifference(int(logger), collect_x, collect_y,
                   rsmpl_t, rsmpl_y, signals, t_range
            )
            # plot raw values with index
            time_raw = numpy.arange(0, len(rsmpl_y))
            plot_raw(int(logger), time_raw, rsmpl_y, signals, 
                "Raw resampled signals"
            )
            # export plot for latex documents
            if EXPORT_LATEX: plotdifferenceLatex(int(logger), collect_x,
                    collect_y, rsmpl_t, rsmpl_y, signals, t_range
                )
            # print resampled data as csv file to standard output
            writecsvdata(int(logger), rsmpl_t, signals, rsmpl_y)
        if interactive: matplotlib.pyplot.show(block = True)
          
        result = 0
    except pandas.errors.EmptyDataError:
    	print("#No data found.", file=sys.stderr)
    	result = 2
    return result

#############################################################################

def main(argv):
    if len(argv) == 5: # TODO: check if limits are correct
        version.set(argv[4])
        lo_lim, up_lim = argv[1], argv[2]
        sig_fs = argv[3]
        result = resample(sys.stdin, lo_lim, up_lim, sig_fs)
    else:
        program = argv[0] if argv else __file__
        print(f"Usage: <pipeline> | {program} <lower lim> <upper lim>" +
            f" <sampling rate> <hashfile>",
            file = sys.stderr
        )
        result = 1
    return result

#############################################################################

if __name__ == "__main__":
    STATUS = main(sys.argv)
    sys.exit(STATUS)

#############################################################################

