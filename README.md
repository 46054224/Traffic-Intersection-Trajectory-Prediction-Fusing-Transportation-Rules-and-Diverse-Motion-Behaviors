# Traffic-Intersection-Trajectory-Prediction-Fusing-Transportation-Rules-and-Diverse-Motion-Behaviors

the model bases on pytorch and python3.6
pip install -r requirements.txt

the files of dataset include 
Veh_tracks_meta.csv
road_information.npz
Veh_smoothed_tracks.csv
TrafficLight_8_02_1.csv
Ped_smoothed_tracks.csv

the visual tool can be downloaded from
https://github.com/SOTIF-AVLab/SinD

generate_graph.py produces the multi-graph
train.py trains the model
test.py evaluate the model
