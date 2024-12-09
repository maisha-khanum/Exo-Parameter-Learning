This script runs a simulated opportunistic optimization using covariance matrix adaptation. For simulation, a set of "optimal" parameters are provided based on random sampling. Running cma_test.py will perform an optimization that converges towards these optimal parameters. To emulate opportunistic optimization, the optimization performs update to 3 sets of parameters based on walking speed range. Here the walking speed is sampled from a normal distribution following real-world data. After installing the program and supporting modules listed below, run the program in the command line by typing "python3 cma_test.py".

The software used to test this script was Python 3.7.7 with additional packages detailed below. Python is an open-source and freely available software.
Additional Python modules required to run script:
- numpy (1.18.2)
- matplotlib (3.2.1)
- scipy (1.4.1)

