from dependencies import *
from dataload import *
from model import *
from helpers import *
import argparse


def train():
    Custom_model = CNN(args.lr, 5, args.opt).build_model()
	# TODO: remove print statement
    print(Custom_model.summary())

    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1)
    train_model = Custom_model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=5,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[es],
    )
    score = Custom_model.evaluate(X_train, y_train, verbose=1)
    print("Training loss:", score[0])
    print("Training accuracy:", score[1])

    score = Custom_model.evaluate(X_val, y_val, verbose=1)
    print("\nValidation loss:", score[0])
    print("Validation accuracy:", score[1])

    plot = plot_loss(train_model)
    plot.savefig("Training_loss_{}_{}.png".format(args.lr, args.opt), bbox_inches='tight', dpi=200)

def test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="​The aim of this assignment is to train and test Convolutional Neural Networks for image classification on the Fashion-MNIST dataset using Keras Module."
    )
    parser.add_argument(
        "-lr",
        "--lr",
        type=float,
        metavar="",
        required=True,
        help="Learning Rate of CNN Model",
    )
    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        metavar="",
        required=True,
        help="Batch Size per Epoch",
    )
    parser.add_argument(
        "-opt",
        "--opt",
        type=int,
        metavar="",
        required=True,
        help="1 for Adam, 2 for RMSProp",
    )
    parser.add_argument("-save_dir", "--save_dir", metavar="", help="Directory path")
    parser.add_argument("-aug", "--dataug", action="store_true", help="Augment Data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-train", "--train", action="store_true", help="Train mode")
    group.add_argument("-test", "--test", action="store_true", help="Test mode")
    args = parser.parse_args()
    print(args)

    # Load training set data
    input = DataLoader()

    if args.train == True:

        X, y = input.load_data("train")
        y = y.reshape(y.shape[0], 1)

        X = X.reshape(-1, 28, 28, 1)
        X = X / 255
        # One-Hot Encoding of labels
        y = keras.utils.to_categorical(y, 10)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        # TODO: Remove console print statements
        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

        # Call the CNN model and train under train()
        train()

    # Print results for the test set under the best hyperparamter setting under test()

