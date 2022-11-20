# Cooperative Control in Multi Agent Pacman
All the modifications are on top of the baseline pacman API from http://inst.eecs.berkeley.edu/~cs188/fa18/projects.html

Use minicontest1 for Qlearning agents and minicontest5 for MultiActor-CentralizedCritic agents

## Dependencies
Dependancy  | Install Command
------------- | -------------
Numpy  | python -m pip install numpy
Pickle  | python -m pip install pickle-mixin
Tkinter | sudo apt-get install python-tk

## Running Game
To train the model use

` python pacman.py -n 10010 -l layoutName -g DirectionalGhost -r -q `

This will add record files under ` minicontest1/records/ ` or ` minicontest5/records/ `

Options | Description
-------------|--------------
  -n GAMES, --numGames=GAMES | the number of GAMES to play, GAMES-numTraining will be used for testing (numTraining is a hyperparamter in the classes of agents in myAgents.py) [Default: 1]
  -l LAYOUT_FILE, --layout=LAYOUT_FILE | the LAYOUT_FILE from which to load the map layout [Options: In layouts/ directory, Default: test51]
  -q, --quietTextGraphics | Generate minimal output and no graphics
  -g TYPE, --ghosts=TYPE | the ghost agent TYPE in the ghostAgents module to use [Options: DirectionalGhost, RandomGhost, Default: RandomGhost]
  -r, --recordActions |  Writes game histories to a file (last 10 episodes will be recorded)

## Replaying Records
Use Bash script ` script.sh ` to run all the records under the ` minicontest1/records/ ` or ` minicontest5/records/ ` directory

                                                OR
                                                
Use the following python command

` python pacman.py --replay=FILNAME -l layoutName `

` layoutName ` should be consistent with training layout

## Changing Parameters
* To change parameters for agents change the parameters change paramters in ` __init__ ` function of the corresponding agent class in myAgent.py
* To change the mini-batch size change ` minibatch_size ` variable in ` pacman.py ` (line number: 694)
