#############################################################################
##                                                                         ##
## makefile: Makefile for the master-thesis                                ##
##                                                                         ##
## Phillip Blunck, 2021-12-14                                              ##
##                                                                         ##
#############################################################################

PACKAGES := python3 python3-pip python3-numpy python3-pandas
PACKAGES += jupyter-notebook python3-virtualenv

PIPPAGES := scikit-learn tensorflow

#############################################################################

LST := data.lst

DATA := $(shell find -L data -name "*.txt.bz2" | LANG=C sort)
LIST := $(shell cat $(LST) 2> /dev/null)

ifneq ($(DATA), $(LIST))
$(shell echo $(DATA) > $(LST))
endif

ifeq ($(DATA), )
$(shell > $(LST))
endif

SHA := data.sha
CSV := data.csv
CSV_M := data_m.csv
DCHRES_SHA := discharge_resistances.sha

#############################################################################

OUT_SIGNALS := UBAT
IN_SIGNALS := STROM TCAVG STTOFCHRG
CONTAINER := 1

LO_LIM := 10571 # Lower limit for signal section in seconds
UP_LIM := 84238 # Upper limit for signal section in seconds
SIG_FS := 10 # Resample sampling rate (Hz)

#############################################################################

all: $(SHA) $(CSV) $(CSV_M)

prepare:
	@ for PACKAGE in $(PACKAGES); do \
	dpkg --status $$PACKAGE 1>/dev/null 2>/dev/null || \
	sudo apt install --yes $$PACKAGE; done
	@ for PIPPAGE in $(PIPPAGES); do \
	pip3 show $$PIPPAGE 1>/dev/null 2>/dev/null || \
	sudo pip3 install --quiet $$PIPPAGE; done
	@ touch $@

$(SHA): $(LST) $(DATA)
	@ bzcat $(DATA) 2> /dev/null | \
	sha1sum > $@.tmp
	@ mv $@.tmp $@

$(CSV): $(LST) $(DATA)
	@ bzcat $(DATA) 2> /dev/null | \
	python3 construct.py $(CONTAINER) \
	$(OUT_SIGNALS) $(IN_SIGNALS) > $@.tmp
	@ mv $@.tmp $@

$(CSV_M): $(LST) $(DATA)
	@ bzcat $(DATA) 2> /dev/null | \
	python3 construct_micro.py $(CONTAINER) \
	$(OUT_SIGNALS) $(IN_SIGNALS) > $@.tmp
	@ mv $@.tmp $@

$(DCHRES_SHA):
	@ cat discharge_resistances.csv 2> /dev/null | \
	sha1sum > $@.tmp
	@ mv $@.tmp $@

signals: prepare $(SHA) $(CSV) $(CSV_M)
	@ python3 signals.py $(SHA) < $(CSV)
	@ python3 signals_micro.py $(SHA) < $(CSV_M)
	@ touch $@

resample: prepare $(SHA) $(CSV_M)
	@ cat $(CSV_M) 2> /dev/null | \
	python3 resample.py $(LO_LIM) $(UP_LIM) $(SIG_FS) $(SHA) > $@.csv.tmp
	@ mv $@.csv.tmp $@.csv
	@ touch $@

#	@ python3 trainmlpbasic.py $(SHA) < resample.csv
#	@ python3 trainmlptemp.py $(SHA) < resample.csv
#	@ python3 trainmlphist.py $(SHA) < resample.csv
#	@ python3 trainsvm.py $(SHA) < resample.csv
training: prepare $(SHA) resample
	@ python3 trainmlphist.py $(SHA) < resample.csv
	@ python3 trainsvm.py $(SHA) < resample.csv
	@ touch $@

simulation: prepare $(SHA) $(CSV)
	@ python3 simulation.py $(SHA)
	@ touch $@

resistances: $(DCHRES_SHA)
	@ python3 resistances.py $(DCHRES_SHA) < discharge_resistances.csv
	@ touch $@

view: signals
	@ eog *.png

notebook: prepare
	@ virtualenv notebook-env
	@ jupyter-notebook

clean:
	rm -f signals-*.png signals
	rm -f resample-*.png resample-*.pdf resample
	rm -f training-*.png training-*.pdf training
	rm -f simulation-*.png simulation-*.pdf simulation
	rm -rf .ipynb_checkpoints
	rm -rf __pycache__

distclean: clean
	rm -f $(LST) $(SHA).tmp $(SHA) $(CSV).tmp $(CSV) \
	$(CSV_M).tmp $(CSV_M) resample.csv.tmp resample.csv
	rm -f prepare
	rm -rf notebook-env

#############################################################################

