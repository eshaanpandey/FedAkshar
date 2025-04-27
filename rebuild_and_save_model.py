import tensorflow as tf
import pickle

def build_model():
    """Recreate the exact CNN architecture used during training."""
    model = tf.keras.models.Sequential([
        # Conv block 1
        tf.keras.layers.Conv2D(32, (3,3), strides=1, activation="relu", input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

        # Conv block 2
        tf.keras.layers.Conv2D(32, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

        # Conv block 3
        tf.keras.layers.Conv2D(64, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

        # Conv block 4
        tf.keras.layers.Conv2D(64, (3,3), strides=1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

        # Dense head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_initializer="uniform",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01),
                              bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(64, activation="relu",
                              kernel_initializer="uniform",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01),
                              bias_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(46, activation="softmax", kernel_initializer="uniform"),
    ])
    return model

if __name__ == "__main__":
    # 1. Build model
    model = build_model()

    # 2. Load aggregated weights
    with open("aggregated_weights.pickle", "rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    print("✅ Weights loaded into model.")

    # 3. Save full model
    model.save("federated_devanagari_model.h5")
    print("✅ Model saved as federated_devanagari_model.h5")
