import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
import flwr as fl
from flwr.client import NumPyClient
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# === Dataset Loading === #
def load_dataset_from_folder(folder_path, img_size=(224, 224), test_size=0.2, random_state=42):
    data, labels = [], []
    class_names = sorted(os.listdir(folder_path))  # Get class names from subdirectories
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize to match model input
                    data.append(img)
                    labels.append(label)

    data = np.array(data, dtype=np.float32) / 255.0  # Normalize
    labels = np.array(labels)
    
    return train_test_split(data, labels, test_size=test_size, random_state=random_state), class_names

folder_path = "C:/Users/Carms/Downloads/Cleaned_Data"
(x_train, x_test, y_train, y_test), class_names = load_dataset_from_folder(folder_path)

# Encode labels and convert to categorical
y_train = to_categorical(LabelEncoder().fit_transform(y_train), len(class_names))
y_test = to_categorical(LabelEncoder().fit_transform(y_test), len(class_names))

# === Model Definition === #
def create_densenet_model(input_shape, num_classes):
    base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:400]:  
        layer.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.6)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# === Federated Learning Client === #
class FlowerClient(NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, model_path):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model_path = model_path

        if os.path.exists(model_path):
            print(f"🔄 Loading latest global model from {model_path}")
            self.model = load_model(model_path)
        else:
            print("⚠️ No global model found, initializing a new model")
            self.model = create_densenet_model(x_train.shape[1:], len(y_train[0]))

    def get_parameters(self, config):
        return self.model.get_weights()  # Returns a list of NumPy arrays

    def fit(self, parameters, config):
        print("🔄 Fetching latest global model weights...")
        self.model.set_weights(parameters_to_ndarrays(parameters))

        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                       epochs=10, batch_size=32,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

        self.model.save(self.model_path)
        print(f"✅ Local model saved as {self.model_path}")

        return ndarrays_to_parameters(self.model.get_weights()), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters_to_ndarrays(parameters))
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}

# === Start Client === #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")

    args = parser.parse_args()
    print(f"🚀 Starting client {args.client_id}")

    model_path = "models/global_model_latest.keras"
    client = FlowerClient(x_train, y_train, x_test, y_test, model_path)

    fl.client.start_client(server_address=f"{args.server_ip}:8081", client=client.to_client())

