#!/bin/bash

echo "Localization evaluation"
echo
echo "Name of files: $( ls ./data/ev_rooms )"
echo 

for i in $( ls ./data/ev_rooms )
do
    echo "Testing $i..."
    python main.py --config configs/ev_rooms_privacy_up.ini --override "scene_names=$i" --log log/loc_ev_rooms_$i --method privacy_recon_loc
done
