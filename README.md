
# Margo_hackathon_JMSM
This project was carried out by a team of four  as part of a hackathon organized by Margo, IBM, and Qubitpharmaceuticals at école Polytechnique on 24-26/01/2025. 

Starting with no prior knowledge in Machine Learning (ML), our goal was to use ML  models to predict molecular toxicity using chemical and biological information.

**Our team ranked 3rd out of 14 teams**, achieving a Cohen-Kappa score of 0.599 on the testing data provided by the organizers. 


## Methods/models used
- Dimensionality reduction
- threshold "fine tuning" for classification tasks with a high cost of false negatives 
- Convolutional Neural Networks
- Graph Neural Newtoks
- Random Forests

## Code organisation
- **GNN+MLP.ipynb** is a notebook implementing a GNN and a MLP to leverage the geometric data on molecules through a graph representation, inspired by the approach of a paper in Nature: Corso, G., Stark, H., Jegelka, S. et al. Graph neural networks. Nat Rev Methods Primers 4, 17 (2024). https://doi.org/10.1038/s43586-024-00294-7. It also implements our procedure for choosing the right threshold for the classification of molecules.
- **MLP_fingerprints_reduction.py** implements a naïve Multi-layer-perceptron using only Morgan fingerprints of the molecule as features, and reducing the dimensionality of the fingerprints through principal component analysis (PCA) (from 2048 to 50 features).
- **Rand_forest(final_sub).py** is our final submission, which got us our best results. It implements a random forest model.
- GNN+MLP.ipynb is a notebook implementing a GNN and a MLP to leverage the geometric data on molecules through a graph representation, inspired by the approach of a paper in Nature: Corso, G., Stark, H., Jegelka, S. et al. Graph neural networks. Nat Rev Methods Primers 4, 17 (2024). https://doi.org/10.1038/s43586-024-00294-7. It also implements our procedure for choosing the right threshold for the classification of molecules.
- **MLP_fingerprints_reduction.py** implements a naïve Multi-layer-perceptro (MLP) n using only Morgan fingerprints of the molecule as features, and reducing the dimensionality of the fingerprints through principal component analysis (PCA) (from 2048 to 50 features). 
- **Basic_train_MLP** also implements a naïve MLP, but using all features. This was our first approach. It manually splits the training data in training and validation data through the script in **data_retrieve_processing.py**

## Installation
To install and run this project, follow the steps below:
1. Retrieve the data from the the following Article:  Karim, A., Lee, M., Balle, T. et al. CardioTox net: a robust predictor for hERG channel blockade based on deep learning meta-feature ensembles. J Cheminform 13, 60 (2021). https://doi.org/10.1186/s13321-021-00541-z
2. put the data in the data directory
3. Install the requirements using pip instlal -r requirements.txt

## Usage
Simply run the python scripts corresponding to the approach you are interested in. The training takes less than a minute for random forests, and less than 10 minutes for the GNN and the MLPs. 

