# A spiking neural program for sensorimotor control during foraging in flying insects

This repository contains all accompanying code and allows to re-generate the results for the following paper:

*H. Rapp, MP. Nawrot, A spiking neural program for sensorimotor control during foraging in flying insects.*

If you use any parts of this code for your own work, please acknowledge us by citing the above paper.


If you have questions or encounter any problems while using this code base, feel free to file a Github issue here and we'll be in touch !

# project layout
This project uses a mixed code base of Python and MATLAB scripts. Python and BRIAN2 is used for the spiking neural network (SNN) models and simulations thereof. The simulation results (spike trains) are dumped as numpy pickled files (NPZ) and MATLAB (MAT) files.

MATLAB is used for all learning and memory experiments to train the Multispike Tempotron readout neuron on the dumped spike trains from the model simulations and for most data analysis and figures.

All script files are commented and/or self-explanatory.

* `./` root folder contains all Python and BASH scripts to run the SNN simulations to re-generate the data used for the paper (Note: this requires several TB of disk space and large amount of RAM !)
* `olnet/models/` contains the BRIAN2 Mushroom body SNN model definitions
* `olnet/plotting/` contains matplotlib scripts to plot SNN network activity
* `matlab/` contains all the MATLAB code for fitting the readout neuron, data analysis and figures


# Using the model
The BRIAN2 model definition is located in the file `olnet/droso_mushroombody.py` for the model without APL neuron and `olnet/droso_mushroombody_apl.py` for the model with APL neuron. If you want to use our model for your own study, just import the model definitions from these files.
Both models use the same interface such that you can easily swap out the implementations by modifying your `import` statement. For an example on how to use it see the `run_model` function in `mkDataSet_DrosoCustomProtocol.py`.


# usage
Detailed description of usage of the individual script files will be published here shortly.