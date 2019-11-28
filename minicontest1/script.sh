for entry in "./records/"*
do
    python3 pacman.py --replay="$entry" -l smallClassic
done
