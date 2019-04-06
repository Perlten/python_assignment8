import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import kaggle


if __name__ == "__main__":
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'PromptCloudHQ/imdb-data', path='./', unzip=True)

    data = pd.read_csv("IMDB-Movie-Data.csv")

    data = data[pd.notnull(data["Metascore"])]
    data = data[pd.notnull(data["Revenue (Millions)"])]

    xs = data["Metascore"]
    ys = data["Revenue (Millions)"]

    xs_reshape = np.array(xs).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(xs_reshape, ys)

    model = sklearn.linear_model.LinearRegression()

    model.fit(x_train, y_train)

    print(model.score(x_test, y_test))
    y_predict = model.predict(x_test)

    plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_predict, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
