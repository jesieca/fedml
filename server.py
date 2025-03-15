import os
import flwr as fl
from typing import List, Tuple
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from flwr.server import ServerConfig

# === Model Definition === #
def create_densenet_model(input_shape, num_classes):
    base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:400]:  # Freeze first 400 layers
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

# === Custom Strategy with Model Loading === #
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.global_model = create_densenet_model((224, 224, 3), 6)
        self.num_rounds = num_rounds
        self.save_dir = "models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.save_dir, "global_model_latest.keras")

        if os.path.exists(self.model_path):
            print(f"🔄 Loading saved global model from {self.model_path}")
            self.global_model.load_weights(self.model_path)
        else:
            print("⚠️ No saved global model found. Starting fresh.")

    def initialize_parameters(self):
        """Ensure server initializes with pre-trained model weights."""
        return ndarrays_to_parameters(self.global_model.get_weights())

    def aggregate_fit(self, rnd: int, results: List[Tuple[Parameters, fl.common.FitRes]], failures: List[str]):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters:
            self.save_global_model(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def save_global_model(self, parameters):
        """Save aggregated weights to file."""
        weights = parameters_to_ndarrays(parameters)
        self.global_model.set_weights(weights)
        self.global_model.save(self.model_path)
        print(f"✅ Latest global model saved as {self.model_path}")

# === Run Server === #
if __name__ == "__main__":
    print("🚀 Starting Flower Server...")
    strategy = CustomFedAvg(min_available_clients=3, min_fit_clients=3)

    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )
