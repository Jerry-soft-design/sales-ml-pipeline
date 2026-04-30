# Paso 1 – Imports
import logging
import pandas as pd
import numpy as np
import argparse
import os
import yaml
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def setup_logging(logging_config):  
    log_level = getattr(logging, logging_config.get("level", "INFO"), logging.INFO)
    log_file = logging_config.get("file", "outputs/pipeline.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Paso 1 – Cargar configuración
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Paso 2 – Cargar datos
def load_data(path):
    logging.info(f"Cargando datos desde: {path}")

    if not os.path.exists(path):
        logging.error(f"No existe el archivo: {path}")
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)
    df["Order.Date"] = pd.to_datetime(df["Order.Date"], errors="coerce")
    df["Ship.Date"] = pd.to_datetime(df["Ship.Date"], errors="coerce")

    logging.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def apply_derived_features(df, derived_config):
    logging.info("Aplicando columnas derivadas desde configuración")

    for feature_name, params in derived_config.items():
        feature_type = params["type"]
        output_col = params["output_name"]

        if feature_type == "difference_days":
            start = params["start_date"]
            end = params["end_date"]

            df[output_col] = (
                df[start] - df[end]
            ).dt.days

        elif feature_type == "extract_year":
            source = params["source_date"]
            df[output_col] = df[source].dt.year

        else:
            raise ValueError(f"Tipo de feature derivada no soportado: {feature_type}")

        logging.info(f"Columna derivada creada: {output_col}")

    return df


# Paso 3 – Feature engineering
def create_features(df, feature_config):
    logging.info("Creando features desde configuración")

    selected_features = (
        feature_config["numeric"] + feature_config["categorical"]
    )

    features = df[selected_features]
    target = df["Sales"]

    features_encoded = pd.get_dummies(
        features,
        columns=feature_config["categorical"],
        drop_first=True
    )

    return features_encoded, target

# Paso 4 – KPIs automáticos
def calculate_kpis(df):
    kpis = {
        "total_sales": df["Sales"].sum(),
        "total_profit": df["Profit"].sum(),
        "avg_profit_margin": (df["Profit"] / df["Sales"]).mean(),
        "avg_shipping_days": (
            df["Ship.Date"] - df["Order.Date"]
        ).dt.days.mean(),
    }

    return pd.DataFrame([kpis])

def build_neural_network(input_dim, nn_params):    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
    except Exception as e:
        raise RuntimeError(
            "TensorFlow no puede cargarse en este entorno. "
            "La red neuronal es opcional."
        ) from e

    model = Sequential()

    for units in nn_params["hidden_layers"]:
        model.add(Dense(units, activation="relu"))

    model.add(Dense(1))  # salida para regresión

    model.compile(
        optimizer=Adam(learning_rate=nn_params["learning_rate"]),
        loss="mean_absolute_error",
        metrics=["mae"]
    )

    return model

def get_model(model_config, input_dim=None):
    model_type = model_config["type"]
    params = model_config["params"][model_type]

    if model_type == "lightgbm":
        return LGBMRegressor(**params)

    elif model_type == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            **params
        )

    elif model_type == "neural_network":
        return build_neural_network(input_dim, params)

    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

#  Paso 5 –Entrenar modelo
def train_model(X_train, y_train, model, model_type, model_params):
    logging.info(f"Entrenando modelo: {model_type}")

    if model_type == "neural_network":
        model.fit(
            X_train,
            y_train,
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            validation_split=model_params["validation_split"],
            verbose=0
        )
    else:
        model.fit(X_train, y_train)

    logging.info("Entrenamiento finalizado")
    return model

#  Paso 6 –Evaluar modelo
def evaluate_model(model, X_test, y_test, model_type):
    logging.info("Evaluando modelo")

    preds = model.predict(X_test)

    if model_type == "neural_network":
        preds = preds.flatten()

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

    logging.info(
        f"Métricas: MAE={metrics['MAE']:.2f}, "
        f"RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}"
    )

    return metrics, preds



#M  Paso 7 – AIN PIPELINE
def main(config):
    logging.info("Pipeline iniciado")
    logging.info(f"Modelo seleccionado: {config['model']['type']}")
    logging.info(f"Test size: {config['split']['test_size']}")
    # Load data
    df = load_data(config["data"]["path"])
    df = apply_derived_features(
    df,
    config["derived_features"]
    )

    # KPIs
    kpis = calculate_kpis(df)
    kpis.to_csv(config["outputs"]["kpis"], index=False)

    # Features
    X, y = create_features(df, config["features"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"],
    )

    # Train
    
    model_type = config["model"]["type"]
    model_params = config["model"]["params"][model_type]

    model = get_model(
        config["model"],
        input_dim=X_train.shape[1]
    )

    
    # Escalado SOLO para NN
    if model_type == "neural_network":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = train_model(
        X_train,
        y_train,
        model,
        model_type,
        model_params
    )

    # Evaluate
    metrics, predictions = evaluate_model(
    model,
    X_test,
    y_test,
    model_type
    )

    # Save predictions
    pred_df = pd.DataFrame(
        {"Sales_real": y_test, "Sales_predicted": predictions}
    )
    pred_df.to_csv(config["outputs"]["predictions"], index=False)

    # Save metrics    
    with open(config["outputs"]["metrics"], "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    logging.info("Pipeline finalizado correctamente")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de ventas con Machine Learning"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo config.yaml"
    )    

    return parser.parse_args()

# Paso 8 – Ejecutar
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["logging"])
    main(config)
