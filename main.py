import sys

from data import load_data
from preprocess import preprocess_data
from train import train_model


def main():

    if len(sys.argv) < 3:
        print(
            "Uso: python main.py <Modelo> <Hyperparam>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    hyperparam = sys.argv[2]

    X, y = load_data(
        "airbnb-listings-extract.csv"
    )

    (
        preprocess,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = preprocess_data(X, y)

    train_model(
        model_name,
        hyperparam,
        preprocess,
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()