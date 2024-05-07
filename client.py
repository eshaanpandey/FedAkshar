import flwr as fl
import tensorflow as tf

#model definition
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), strides=1, activation="relu", input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
        tf.keras.layers.Conv2D(32, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
        tf.keras.layers.Conv2D(64, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
        tf.keras.layers.Conv2D(64, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer="uniform", kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu", kernel_initializer="uniform", kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(46, activation="softmax", kernel_initializer="uniform")
    ])

# Load and preprocess your data
train_path = "trainPath"
test_path =  "testPath"

train_data = tf.keras.preprocessing.image_dataset_from_directory(train_path, image_size=(32, 32), batch_size=32, label_mode='categorical', shuffle=True, interpolation="lanczos5")
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(32, 32), batch_size=32, label_mode='categorical', shuffle=True, interpolation="lanczos5")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


class myClient(fl.client.NumPyClient):
    
        
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_data, epochs=1, validation_data=test_data)
        print(f"Client Weights After Training:")
        # Print all weight layers (modify for specific layer printing if needed)
        for layer in model.layers:
            print(layer.name, layer.get_weights())
        return model.get_weights(), len(train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_data)
        return loss, len(test_data), {"accuracy":accuracy}

fl.client.start_numpy_client(
    server_address="Server's IP",
    client=myClient(),
)
