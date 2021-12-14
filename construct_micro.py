#!/usr/bin/python3

#############################################################################
##                                                                         ##
## construct_micro.py: Generates a csv file with data in microseconds.     ##
##                     For missing values are set to NaN at a specific     ##
##                     timepoint                                           ##
##                                                                         ##
## Phillip Blunck, 2021-10-05                                              ##
##                                                                         ##
#############################################################################

import numpy as np
import re
import sys

#############################################################################

EMPTY = ""
COLON = ":"
COMMA = ","
DEGREE_SIGN = "\N{DEGREE SIGN}"
SOLIDUS = "/"
SPACE = " "
NAN = "NaN"

MICRO_SCALE = 1000000

startflag = False

#############################################################################

def writecsvheader(signals, units):
    """Prints out a csv-file header with specific signals
       and adds an empty line for the following data.
    """
    # Logger, timestamp, sig1 / unit2, ..., sigN / unitN
    header = "Logger" + COMMA + SPACE + "Timestamp"
    for signal in signals:
        header += COMMA + SPACE
        header += signal + SPACE
        if units[signal] == "degC":
            header += SOLIDUS + SPACE + DEGREE_SIGN + "C"
        elif units[signal] is not EMPTY:
            header += SOLIDUS + SPACE + units[signal]
    print(header)
    print(EMPTY)

#############################################################################

def writecsvline(logger, timestamp, signals, values, units):
   """The function writeline prints out the logger, signalvalues
   for each timestamp and can print out the column names for the csv-format.
   """
   global startflag
   line = logger + COMMA + SPACE + timestamp
   for signal in signals:
      if signal in values:
         line += COMMA + SPACE
         line += values[signal]
      else:
         line = EMPTY
         break

   if line:
      if not startflag:
         writecsvheader(signals, units)
         startflag = True
      print(line)

#############################################################################

def convert_time(time):
    """Converts the timevalues from 0 to n microseconds
    """
    converted = [0] * len(time)
    for idx, t in enumerate(time[:-1]):
        converted[(idx + 1)] = abs(int(time[idx + 1]) - int(t)) + converted[idx]
    return [str(time / MICRO_SCALE) for time in converted]

#############################################################################

def construct(stdin, container, signals):
    """Searches in each line of the logfile for the signals.
       When a signal is missing a NaN value is set.
       The corresponding time value of the found signal and missing signals
       is also set.
    """
    # compile all search pattern
    # Start of a logfile
    string = "^###"
    start = re.compile(string)

    # Timestamp
    string = "^# (....-..-.. ..:..:..) .* \[(....)\]"
    timestamp = re.compile(string)

    # telegram time in micro seconds
    string = SPACE + "([0-9]+)" + SPACE + "([0-9]+)"
    microsec = re.compile(string)

    # signal message pattern
    # chain up all signalnames in one string
    string = EMPTY
    for signal in signals:
        if string: string += "|"
        string += signal

    string = SPACE + f"({string})" + COLON + "([-0-9.]+)" + "([^ ]*)"
    message = re.compile(string)

    # container pattern
    string = "^[A-Z]+" + container + SPACE
    contpattern = re.compile(string)

    # search signals
    values = {}
    units = {}
    time = []
    data = [] # contains dictionaries of values
    
    for line in stdin:
        # start pattern
        match = start.match(line)
        if match:
            continue

        # timestamp
        match = timestamp.match(line)
        if match:
            logger = match[2]
            continue

        # container
        if container:
            if not contpattern.match(line):
                continue

        # find and save signal values and units
        matches = message.findall(line)
        for match in matches:
            # 0=signal 1=wert 2=unit
            for signal in signals:
                values[signal] = NAN # clear value dict with NaN
            values[match[0]] = match[1]
            units[match[0]] = match[2]
            match_t = microsec.search(line)
            if match_t:
                time.append(match_t[1] + match_t[2])
                data.append(values.copy()) # save signal values for time 

    # write collected data to stdout as csv file
    cvrt_time = convert_time(time) # convert time into seconds
    for idx, time_n in enumerate(cvrt_time):
        writecsvline(logger, time_n, signals, data[idx], units)

#############################################################################

def main(argv):
    """Check argument vector and call contruct function if successfull.
       Otherwise print out instruction for the user.
    """
    if len(argv) > 2 and argv[1] in ["1","2"]:
        container = argv[1]
        argv.pop(1)
    else:
        container = EMPTY
    if len(argv) > 1:
        signals = argv[1:]
        construct(sys.stdin, container, signals)
        result = 0
    else:
        program = argv[0] if argv else __file__
        print(f"Usage: <pipeline> | {program} [1|2] <signal> ...",
            file = sys.stderr
        )
        result = 1
    return result

#############################################################################

if __name__ == "__main__":
    STATUS = main(sys.argv)
    sys.exit(STATUS)

#############################################################################

