import pandas as pd


def read_known_data(path, alpha1, alpha2):
    df = pd.read_csv(path, header=None)
    print(df.keys())
    df = df.loc[df[df.keys()[0]] > alpha1]
    df = df.loc[df[df.keys()[0]] < alpha2]
    print(df)
    print(df.describe())


def read_unknown_data(path, alpha1, alpha2):
    df = pd.read_csv(path, header=0, index_col=0)
    print(df)
    print(df.keys())
    df = df.loc[df[df.keys()[0]] > alpha1]
    df = df.loc[df[df.keys()[0]] < alpha2]
    print(df.head(10))
    print(df.describe())


if __name__ == '__main__':
    # read_known_data('./data/polblogs_byGEMSEC.csv', 0.2, 0.55)
    read_known_data('./data/football_byDANMF.csv', 0.2, 1)
