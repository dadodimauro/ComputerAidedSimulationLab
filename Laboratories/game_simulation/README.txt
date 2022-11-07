two files:

- "game_simulation_script.py"
is the script used to run the simulation, it has 4 parameters passed with command line:
	--nRuns (default=30)
	--area (default=5)
	--initialPlayers (default=4)
	--movementSpeed (default=1)
it outputs a csv file names "outuput_simulation.csv" containing the results of the simulation

- "visualization_game_simulation.py"
is the script to plot all information about the simulation. It also save the output as a .png 
file named "visualization_game_simulation.png"