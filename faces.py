import keras.utils as utils
from keras.models import Sequential
import keras.layers as layers
import keras.losses as losses
import keras.optimizers as optimizers
import time

start = time.time()

train = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (300, 300),
    seed = 21,
    validation_split = 0.3,
    subset = "training",
    )

test = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (300, 300),
    seed = 21,
    validation_split = 0.3,
    subset = "validation",
)

class Net():
    def __init__(self, input_shape):
        self.model = Sequential()

        # self.model.add(layers.ZeroPadding2D(
        #     padding = ((0,0), (0,0)),
        # ))

        self.model.add(layers.Conv2D(
            8, # filters
            15, # kernals
            strides = 5, # step size
            activation = "relu",
            input_shape = input_shape,
        )) # output 58 x 58 x 8

        self.model.add(layers.MaxPool2D(pool_size = 2))
        # output 29 x 29 x 8

        self.model.add(layers.Conv2D(
            8, # filters
            3, # kernal
            strides = 1,
            activation = "relu"
        )) # output 27 x 27 x 8

        self.model.add(layers.ZeroPadding2D(
             padding = ((1,0), (1,0)),
         ))# output 28 x 28 x 8

        self.model.add(layers.MaxPool2D(pool_size = 2))
        # output 14 x 14 x 8

        self.model.add(layers.Flatten())
        # output 1568

        # self.model.add(layers.Dense(1024, activation = "relu"))
        self.model.add(layers.Dense(256, activation = "relu"))
        self.model.add(layers.Dense(64, activation = "relu"))
        self.model.add(layers.Dense(16, activation = "relu"))
        self.model.add(layers.Dense(5, activation = "softmax"))

        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer, 
            metrics = ["accuracy"],
        )
    
    def __str__(self):
        self.model.summary()
        return ""

net = Net((300, 300, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 40,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)

net.model.save("faces_model_save_0")

end = time.time()

print ("finished in" + str(end-start))