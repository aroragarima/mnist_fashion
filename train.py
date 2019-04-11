from dependencies import *
from dataload import DataLoader
from model import CNN
from helpers import *
import argparse


def train():
    # mode = "train"
    input = DataLoader()
    X, y = input.load_data(mode="train")
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

    Custom_model = CNN(args.lr, 5, args.opt).build_model()

    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1)
    if args.dataug == 1:

        from keras.preprocessing.image import ImageDataGenerator

        X_train_aug = np.array(X_train, copy=True)
        y_train_aug = np.array(y_train, copy=True)

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
        )

        datagen.fit(X_train)

        # Concatenating the old data with the augmented data
        res_X_train = np.concatenate((X_train, X_train_aug), axis=0)
        res_y_train = np.concatenate((y_train, y_train_aug), axis=0)

        # fits the model on batches with real-time data augmentation:
        train_model = Custom_model.fit_generator(
            datagen.flow(res_X_train, res_y_train, batch_size=128),
            epochs=50,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[es],
        )

    elif args.dataug == 2:
        train_model = Custom_model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=50,
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
    if args.opt == 1:
        plot.savefig(
            "Training_loss_{}_{}.png".format(args.lr, "Adam"),
            bbox_inches="tight",
            dpi=200,
        )
        print("Training Complete. Training_loss_{}_{}.png".format(args.lr, "Adam"))
    elif args.opt == 2:
        plot.savefig(
            "Training_loss_{}_{}.png".format(args.lr, "RMSProp"),
            bbox_inches="tight",
            dpi=200,
        )
        print("Training Complete. Training_loss_{}_{}.png".format(args.lr, "RMSProp"))
    Custom_model.save(
        "{}train_model.h5".format(args.save_dir)
    )  # creates a HDF5 file 'my_model.h5'


def test():
    input = DataLoader()
    X_test, y_test = input.load_data(mode="test")
    print(X_test.shape, y_test.shape)
    X_test = X_test.reshape(-1, 28, 28, 1)
    print(X_test.shape)
    trained_model = load_model(
        "{}train_model.h5".format(args.save_dir),
        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform},
    )
    predicted_classes = trained_model.predict_classes(X_test)

    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(
        y_test, predicted_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(
        cnf_matrix,
        classes=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle Boot",
        ],
        title="Confusion matrix, with normalization",
        normalize=True,
    ).show()


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
    parser.add_argument(
        "-aug",
        "--dataug",
        type=int,
        metavar="",
        default=2,
        help="1 for True, 2 for False",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-train", "--train", action="store_true", help="Train mode")
    group.add_argument("-test", "--test", action="store_true", help="Test mode")
    args = parser.parse_args()
    print(args)

    # Load training set data
    if args.train == True:
        # Call the CNN model and train under train()
        train()

    # Print results for the test set under the best hyperparamter setting under test()
    if args.test == True:
        test()
