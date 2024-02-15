This is a code supplement to ["Analysis on the Vulnerability of Multi-Server Federated Learning Against Model Poisoning Attacks"](https://repository.tudelft.nl/islandora/object/uuid%3A60f7bc45-2550-4e03-a570-ae2a4bb01b14).

It is done as part of the 2024 winter edition of the [Research Project](https://github.com/TU-Delft-CSE/Research-Project) of [TU Delft](https://github.com/TU-Delft-CSE)

It is based on the code from:
https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/tree/main/cifar10
with added modifications to simulate a FedMes network and the two novel attacks, discussed in the paper.

It is recommended to run the code in a python venv.

To install the requierments use
```
pip install -r requirements.txt
```

Before running it for the first time use the following command to obtain the datasets
```
python data.py
```

Afterwards the parameters can be adjusted by modifing the `arguments.py` file and multiple experiments can set to run in sequence by adjusting the `main.py` file.

To start it use:
```
python main.py
```
