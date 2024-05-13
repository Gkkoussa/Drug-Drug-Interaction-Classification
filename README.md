# Drug-Drug-Interaction-Classification
Drug to Drug Interaction Classifier
![287417229-b56a6ee4-3aab-47c1-8116-481339bfb7bb](https://github.com/Gkkoussa/Drug-Drug-Interaction-Classification/assets/70414077/c55b87ae-f27f-4285-8b7c-5bbc1eeed922)


Our solution utilizes CatBoost and link prediction to classify predict drug-drug interactions in the absence of information about each individual drug's other interactions. CatBoost predicts whether pairs of drugs interact, and we use these predictions as adjacency matrix entries for link prediction. 

This method appears to slighly outperform standalone CatBoost.

# Code

parse.ipynb contains the code for collecting features from the DrugBank database and saving them as a csv.
The catboost folder contains all the code we used to train our CatBoost model.
link_prediction.ipynb contains the code for evaluating standalone link prediction, standalone CatBoost, and our stacked model.
