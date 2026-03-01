import pandas as pd


def load_data(path):

    df = pd.read_csv(path, sep=";")

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Price"])

    df["price_class"] = pd.qcut(
        df["Price"],
        q=3,
        labels=["low", "mid", "high"]
    )

    y = df["price_class"]

    X = df.drop(columns=["price_class", "Price"])

    columns_id = [c for c in X.columns if "id" in c.lower()]
    X = X.drop(columns=columns_id)

    return X, y