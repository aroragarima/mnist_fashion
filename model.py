from dependencies import *


class CNN(object):
    def __init__(self, lr, kernel_size, opt):
        self.lr = lr
        self.kernel_size = kernel_size
        self.opt = opt

    def cnn_models(self):
        model = Sequential()

        # Adding Convolutional layer with 32 outputs which indicates number of filters to be used are 32 with each of them of dimension 5 x 5
        model.add(
            Conv2D(
                32,
                kernel_size=(self.kernel_size, self.kernel_size),
                activation="relu",
                kernel_initializer="random_uniform",
                input_shape=(28, 28, 1),
            )
        )
        # Adding MAX Pooling Layer 2X2
        model.add(MaxPooling2D((2, 2)))
        # Adding Padding = 1
        model.add(ZeroPadding2D((1, 1)))
        # Adding Convolutional layer with 32 inputs from CONV1, 64 outputs which indicates number of filters to be used are 64 with each of them of dimension 5 x 5
        model.add(
            Conv2D(
                64, kernel_size=(self.kernel_size, self.kernel_size), activation="relu"
            )
        )
        # Adding MAX Pooling Layer 2X2
        model.add(MaxPooling2D((2, 2)))
        # Adding Padding = 1
        model.add(ZeroPadding2D((1, 1)))
        # Adding 2D Convolution Layer with 128 Outputs Filter size 3x3
        model.add(Conv2D(128, (3, 3), activation="relu"))
        # 	Adding Convolution Layer with 128 Outputs Filter size 5x5
        model.add(Conv2D(128, (3, 3), activation="relu"))
        # Adding MAX Pooling Layer 2X2
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        # Adding Fully Connected Layer1
        model.add(Dense(256, activation="relu"))
        # Adding Fully Connected Layer2
        model.add(Dense(256, activation="relu"))
        # S oftMax Layer for Classification
        model.add(Dense(10, activation="softmax"))

        if self.opt == "Adam":
            optimizer = keras.optimizers.Adam(lr=self.lr)
        elif opt == "RMSprop":
            optimizer = keras.optimizers.RMSprop(lr=self.lr)
        else:
            print("Optimiser unrecognised, Please check help section for help")
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

if __name__ == "__main__":
	Custom_model = CNN(0.001, 5, "Adam")
	print(Custom_model.cnn_models().summary())