#!/usr/bin/python3

#############################################################################
##                                                                         ##
## signals_micro.py: Displays all microsecond signals                      ##
##                   from the saved csv-file                               ##
##                                                                         ##
## Phillip Blunck, 2021-10-05                                              ##
##                                                                         ##
#############################################################################

NAME = "signals-micro" # Name for picture file

interactive = True # Flag for interactive mode

#############################################################################

import datetime
import matplotlib
if not interactive: matplotlib.use("agg")
import matplotlib.pyplot
import numpy
import pandas
import subprocess
import sys

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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

def plotdata_date(logger, time, data, signals):
   """Plot datapoints for every signal with date timeline.
   """
   figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)

   for num, signal in enumerate(signals):
      x = time
      y = data[:, num]
      # plot data
      matplotlib.pyplot.subplot(len(signals), 1, num+1)
      matplotlib.pyplot.ylabel(signal)
      matplotlib.pyplot.plot(x, y, '.')
      matplotlib.pyplot.grid(True)

   matplotlib.pyplot.xlabel("Time (s)")
   matplotlib.pyplot.text(
      0.02, 0.02, version.get(f"Logger {logger}"),
      transform=matplotlib.pyplot.gcf().transFigure
   )

   if interactive: matplotlib.pyplot.show(block = False)
   # saving figure as png
   number = (figure.number - 1) % 2 + 1
   matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")

#############################################################################

def plotdata_raw(logger, time, data, signals):
   """Plot datapoints for every signal with raw timeline.
   """
   figure = matplotlib.pyplot.figure(figsize=SIZE, dpi=RES)

   for num, signal in enumerate(signals):
      x = time
      y = data[:, num]
      # plot data
      matplotlib.pyplot.subplot(len(signals), 1, num+1)
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
   number = (figure.number - 1) % 2 + 1
   matplotlib.pyplot.savefig(f"{NAME}-{logger}-{number:02d}.png")

#############################################################################

def plotlogger(logger, data, signals):
   """Generate timelines for raw data point and for every second.
   Plot signals of every logger for each timeline.
   """
   # generate timelines
   time_raw = numpy.arange(0, len(data))
   #time_date = [
   #   datetime.datetime.strptime(t, "%S.%f")
   #   for t in data[:, 1]
   #]
   time_date = data[:, 1]
   data = data[:, 2:]

   # plot signals for both timelines
   plotdata_raw(logger, time_raw, data, signals)
   plotdata_date(logger, time_date, data, signals)

#############################################################################

def display(stdin):
   """Try to open the csv file and plot the signals in following formats:
   First: Raw signals
   Second: Signals for each second.
   """
   try:
      # read csv
      data = pandas.read_csv(stdin, delimiter=DELIMITER, engine="python")
      # get signalnames
      signals = data.columns[2:]
      # convert csv data to numpy array
      data = data.to_numpy()
      
      # plot signals for each logger
      loggers = {* data[:, 0]}
      for logger in sorted(loggers):
         logdata = numpy.array([d for d in data if d[0] == logger])
         plotlogger(int(logger), logdata, signals)

      if interactive: matplotlib.pyplot.show(block = True)
      result = 0
   except pandas.errors.EmptyDataError:
      print("# No data found.", file=sys.stderr)
      result = 2
   return result

#############################################################################

def main(argv):
   """Check the input argument vector. If correct start the script,
   otherwise print usage message.
   """
   if len(argv) == 2:
      version.set(argv[1])
      result = display(sys.stdin)
   else:
      program = argv[0] if argv else __file__
      print(f"Usage: <pipeline> | {program} <hashfile>", file=sys.stderr)
      result = 1
   return result

#############################################################################

if __name__ == "__main__":
   STATUS = main(sys.argv)
   sys.exit(STATUS)

#############################################################################

