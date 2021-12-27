# master-thesis
Repository meiner Masterarbeit "Entwicklung und Evaluierung eines Modells zur Prognose des State of Health von Traktionsbatterien auf Basis künstlicher neuronaler Netze"

Repository of my master’s thesis "Development and evaluation of a model for predicting the state of health of traction batteries based on artificial neural networks"

# Running instructions
To use this project, you have to install a linux distribution. In this case you can use Ubuntu 20.04.3 LTS.
It comes with the build automation tool make. It is used to run the commands in the terminal.

## First start
After you cloned this repository, you should use the following command to prepare your system.  

```make prepare```  

It installs all necessery libraries and Python packages.  
After this you need to  generate three directories.  

```mkdir models data```

## Example
After the preperation you can start using the main steps to calculate the internal resistance of a specific battery.
1. Place a single logfile into the directory ```data```
2. Run ```make signals```
3. Set the limits of the signal section in seconds in the makefile (```LO_LIM``` and ```UP_LIM```)
4. Run ```make resample``` to resample the signal section
5. Run ```make training``` to train the MLP and SVM with the resampled signal section
6. Run ```make simulation``` to calculate the internal resistance.

## Predicting the state of health
If you want to predict the SoH, you have to repeat the steps shown in the example. Each time you calculate the interal resistance, save the values into the ```discharge_resistances.csv``` file.
Remember to only place one logfile at a time in the directory ```data```.
After you calculated enough values you can run the command ```make resistance``` to  get a diagram with all the calculated internal resistances of the csv-file.  
Now you can make a prediction about the SoH based on the course of the internal resistances.
