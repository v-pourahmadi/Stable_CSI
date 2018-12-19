# Stable CSI Extraxtion 

## Source code of:
Deep Neural Networks Meet CSI-Based Authentication

## Abstract

The first step of a secure communication is authenticating legible users and detecting the malicious ones. In the last recent years some promising schemes proposed using wireless medium network's features, in particular channel state information (CSI) as a means for authentication. These schemes mainly compare user's previous CSI with the new received CSI to determine if the user is in fact what it is claiming to be. 
Despite high accuracy, these approaches lack the stability in authentication when the users rotate in their positions. This is due to significant change in CSI when a user rotates which mislead the authenticator when it compares the new CSI with the previous  ones. Our approach presents a way of extracting features from raw CSI measurements which are stable towards rotation. We extract these features by the means of deep neural network. We also present a scenario in which users can be {efficiently} authenticated while they are at certain locations in an environment (even if they rotate); and, they will be rejected if they change their location. Also experimental results are presented to show the performance of the proposed scheme.

## LocNet Architecture

![Architecture of LocNet](Images/ANN.jpg?raw=true "Architecture of LocNet")


## Architecture of Feature Extractor

![Feature Extractor’s architecture](Images/featureextractor.jpg?raw=true "Feature Extractor’s architecture")


## Datasets
To access datasets contact us: v.pourahmadi@aut.ac.ir
