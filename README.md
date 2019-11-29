# Cooperative Control in Multi Agent Pacman

## Dependancies
Dependancy  | Install Command
------------- | -------------
Numpy  | python -m pip install numpy
Pickle  | python -m pip install pickle-mixin
Tkinter | sudo apt-get install python-tk

## Running Game
Use the following command to get all the possible options for pacman.py

` python pacman.py --help `

To train the model use

` python pacman.py -n 10010 -l layoutName -g DirectionalGhost -r -q `

This will create record files under ` minicontest1/records/ ` or ` minicontest5/records/ `

Options | Description
-------------|--------------
  -h, --help         |   show this help message and exit
  -n GAMES, --numGames=GAMES + 10 | the number of GAMES to play [Default: 1]
  -l LAYOUT_FILE, --layout=LAYOUT_FILE | the LAYOUT_FILE from which to load the map layout [Options: In layouts/ directory, Default: test51]
  -q, --quietTextGraphics | Generate minimal output and no graphics
  -g TYPE, --ghosts=TYPE | the ghost agent TYPE in the ghostAgents module to use [Options: DirectionalGhost, RandomGhost, Default: RandomGhost]
  -r, --recordActions |  Writes game histories to a file (named by the time they were played)

## Replaying Records
Use Bash script ` script.sh ` to run all the records under the ` minicontest1/ ` or ` minicontest5/ ` directory

                                                OR
                                                
Use the following python command
` python pacman.py --replay=FILNAME -l layoutName `
