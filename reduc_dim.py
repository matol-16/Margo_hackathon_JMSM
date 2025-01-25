from sklearn.decomposition import PCA

from Basic_train import CustomDataset


def pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


if __name__ == "__main__":
    # Load datasets
    train_dataset = CustomDataset('data/training_4.csv')

    # Reduce dimensionality of the 1-200 features 

    data_to_be_reduced = [row[1:201] for row in train_dataset.data]  # Correct slicing
    n_components = 50
    reduced_data= pca(data_to_be_reduced, n_components)

    for i, row in enumerate(train_dataset.data):
        row[1:201] = reduced_data[i]

    print(len(train_dataset.data[0]))
    

