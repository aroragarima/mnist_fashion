from dependencies import *


class CNN(object):
    def __init__(self, lr, kernel_size, opt):
        self.lr = lr
        self.kernel_size = kernel_size
        self.opt = 1  # Adam

    def build_model(self):
        self.model = Sequential()

        # Adding Convolutional layer with 32 outputs which indicates number of filters to be used are 32 with each of them of dimension 5 x 5
        self.model.add(
            Conv2D(
                32,
                kernel_size=(self.kernel_size, self.kernel_size),
                activation="relu",
                kernel_initializer="random_uniform",
                input_shape=(28, 28, 1),
            )
        )
        # Adding MAX Pooling Layer 2X2
        self.model.add(MaxPooling2D((2, 2)))
        # Adding Padding = 1
        self.model.add(ZeroPadding2D((1, 1)))
        # Adding Convolutional layer with 32 inputs from CONV1, 64 outputs which indicates number of filters to be used are 64 with each of them of dimension 5 x 5
        self.model.add(
            Conv2D(
                64, kernel_size=(self.kernel_size, self.kernel_size), activation="relu"
            )
        )
        # Adding MAX Pooling Layer 2X2
        self.model.add(MaxPooling2D((2, 2)))
        # Adding Padding = 1
        self.model.add(ZeroPadding2D((1, 1)))
        # Adding 2D Convolution Layer with 128 Outputs Filter size 3x3
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        # 	Adding Convolution Layer with 128 Outputs Filter size 5x5
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        # Adding MAX Pooling Layer 2X2
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        # Adding Fully Connected Layer1
        self.model.add(Dense(256, activation="relu"))
        # Adding Fully Connected Layer2
        self.model.add(Dense(256, activation="relu"))
        # SoftMax Layer for Classification
        self.model.add(Dense(10, activation="softmax"))

        if self.opt == 1:
            print("Adam Optimiser running")
            optimizer = keras.optimizers.Adam(lr=self.lr)
        elif self.opt == 2:
            print("RMSprop Optimiser running")
            optimizer = keras.optimizers.RMSprop(lr=self.lr)
        else:
            print("Optimiser unrecognised, Please check help section for help")
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return self.model

    def print_summary(self):
        return self.model.summary()


if __name__ == "__main__":
    Custom_model = CNN(0.001, 5, "Adam")
    Custom_model.build_model()
    print(Custom_model.print_summary())
