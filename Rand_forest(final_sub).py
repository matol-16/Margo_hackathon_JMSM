import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import os

# to run the code, you only need the three files provided (train.csv, test_1.csv, test_2.csv)

# the code returns the files pred_1.csv and pred_2.csv with the predictions for (test_1.csv and test_2.csv)

# load the data
df = pd.read_csv('data/train.csv')  
df_test1 = pd.read_csv('data/test_1.csv')
df_test2 = pd.read_csv('data/test_2.csv')

X_train = df.iloc[:, 1:-1]  # Features (without smiles)
y_train = df.iloc[:, -1]    # labels

df_test1_ws = df_test1.iloc[:, 1:]
df_test2_ws = df_test2.iloc[:, 2:]

# create a random forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=7)

# train the model on the training data set 
rf_model.fit(X_train, y_train)

# predictions on dataset test1
y_pred1 = rf_model.predict(df_test1_ws)

# predictions on dataset test2
y_pred2 = rf_model.predict(df_test2_ws)

# Convert in pandas Series
y_pred1_series = pd.Series(y_pred1)

test1_df_predictions = pd.DataFrame({
    'smiles': df_test1['smiles'],
    'class': y_pred1_series
})

# Convert in pandas Series
y_pred2_series = pd.Series(y_pred2)

test2_df_predictions = pd.DataFrame({
    'smiles': df_test2['smiles'],
    'class': y_pred2_series
})

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save the predictions to a new CSV file with only smiles and predicted class
test2_df_predictions.to_csv(os.path.join(output_dir, 'pred_2.csv'), index=False)

# to find the 200 molecules for which predictions are the most reliable.
all_tree_predictions = np.array([tree.predict(df_test1_ws) for tree in rf_model.estimators_]).T

def proportion_of_trees_voting_for_predicted_class(tree_predictions, all_tree_predictions):
    proportions = []
    for i in range(len(tree_predictions)):
        # Count how many trees voted for the predicted class
        votes_for_predicted_class = np.sum(all_tree_predictions[i] == tree_predictions[i])
        # Calculate the proportion of trees that voted for the predicted class
        proportion = votes_for_predicted_class / len(all_tree_predictions[i])
        proportions.append(proportion)
    return np.array(proportions)

proportions = proportion_of_trees_voting_for_predicted_class(y_pred1, all_tree_predictions)

sorted_indices = np.argsort(proportions)[::-1] 

top_200_samples = test1_df_predictions.iloc[sorted_indices, :]

top_200_samples.to_csv(os.path.join(output_dir, 'pred_1.csv'), index=False)

print("Predictions saved to output/pred_1.csv and output/pred_2.csv")










