#!/usr/bin/python3

#############################################################################
##                                                                         ##
## construct.py: Collect signals out of a logfile and save them            ##
##               as a csv-file.                                            ##
##                                                                         ##
## Phillip Blunck, 2021-07-05                                              ##
##                                                                         ##
#############################################################################

import re
import sys

#############################################################################

EMPTY = ""
COLON = ":"
COMMA = ","
DEGREE_SIGN = "\N{DEGREE SIGN}"
SOLIDUS = "/"
SPACE = " "

startflag = False

#############################################################################

def writecsvheader(signals, units):
   """The function prints out a csv-file header for specific signalnames
   and adds an empty line afterwards
   """
   header = "Logger" + COMMA + SPACE + "Timestamp"
   for signal in signals:
      header += COMMA + SPACE
      header += signal + SPACE
      if units[signal] == "degC":
         header += SOLIDUS + SPACE + DEGREE_SIGN + "C"
      elif units[signal] is not EMPTY:
         header += SOLIDUS + SPACE + units[signal]
   print(header)
   print()

#############################################################################

def writecsvline(logger, timestamp, signals, values, oldvalues, units):
   """The function writeline prints out the logger, signalvalues
   for each timestamp and can print out the column names for the csv-format.
   """
   global startflag
   line = logger + COMMA + SPACE + timestamp
   for signal in signals:
      if signal in values:
         line += COMMA + SPACE
         line += values[signal]
      elif signal in oldvalues:
         line += COMMA + SPACE
         line += oldvalues[signal]
      else:
         line = EMPTY
         break

   if line:
      if not startflag:
         writecsvheader(signals, units)
         startflag = True
      print(line)

#############################################################################

def construct(stdin, container, signals):
   """This function constructs the input textfiles into a cvs-file.
   It searches for the user signals, container signals, timestamps,
   container IDs and prints it out to the console line by line seperated
   with delimiters.
   """
   
   # Start of logfile pattern
   string = "^###"
   start = re.compile(string)

   # Timestamp pattern
   string = "^# (....-..-.. ..:..:..) .* \[(....)\]"
   timestamp = re.compile(string)

   # Signal message pattern
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

   # search and print signals of logfile from stdin
   values = {}
   oldvalues = {}
   units = {}

   for line in stdin:
      # Start
      match = start.match(line)
      if match:
         values = {}
         oldvalues = {}
         continue

      # timestamp
      match = timestamp.match(line)
      if match:
         logger = match[2]
         time = match[1]
         writecsvline(logger, time, signals, values, oldvalues, units)
         oldvalues = values
         values = {}
         continue

      # check for specific container messages
      # TODO: For future -> Comparison of different container signals
      if container:
         if not contpattern.match(line):
            continue

      # find and save signals values and units
      matches = message.findall(line)
      for match in matches:
         values[match[0]] = match[1]
         units[match[0]] = match[2]

#############################################################################

def main(argv):
   """Checks all command line arguments and stdinputs.
   Then starts the construction of an CSV-file."""

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

