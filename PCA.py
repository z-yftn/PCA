import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

datasets = ['Iris Dataset', 'Abalone Dataset', 'Seeds Dataset']
# Load data set
temp_data = [0, 0, 0]
temp_data[0] = pd.read_csv(
    filepath_or_buffer='iris.csv',
    header=None,
    sep=',')
temp_data[1] = pd.read_csv(
    filepath_or_buffer='abalone.csv',
    header=None,
    sep=',')
temp_data[2] = pd.read_csv(
    filepath_or_buffer='seeds_dataset.csv',
    header=None,
    sep=',')

for num in range(len(datasets)):
    print('current dataset: ', datasets[num])
    df = temp_data[num]
    # split data table into data X and class labels y
    if num == 1:
        x = df.loc[:, df.columns != 0]
        y = df.iloc[:, 0]
    else:
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    # print(x)
    # print(y)

    # normalization of the data to work properly
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    dfx = pd.DataFrame(data=x, columns=df.columns[1:])
    # print(dfx)

    # importing and instantiating PCA with 2 components
    pca = PCA(n_components=2)
    pct = pca.fit_transform(dfx)
    principal_df = pd.DataFrame(pct, columns=['pc1', 'pc2'])
    final_df = pd.concat([principal_df, y], axis=1)
    final_df.head()
    print(final_df)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.set_title('2 component PCA in ' + datasets[num], fontsize=20)
    if num == 0:
        targets = ['Setosa', 'Versicolor', 'Virginica']
    elif num == 1:
        targets = ['M', 'F', 'I']
    else:
        targets = [1, 2, 3]
    colors = ['r', 'y', 'c']
    for target, color in zip(targets, colors):
        indicesToKeep = y == target
        ax.scatter(final_df.loc[indicesToKeep, 'pc1']
                   , final_df.loc[indicesToKeep, 'pc2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # returns a vector of the variance explained by each dimension

    print(pca.explained_variance_ratio_)
    show_explain = pca.explained_variance_ratio_
    print('in ' + datasets[num] + ' the first component is able to explain ' + str(
        show_explain[0] * 100) + '% and the second ' + str(show_explain[1] * 100) + '%')

plt.show()
