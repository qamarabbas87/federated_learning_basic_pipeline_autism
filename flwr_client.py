# flwr_client.py
import argparse
import json
import numpy as np
import flwr as fl

from fed_shared_pipeline import (
    load_arff, preprocess_dataframe, split_sites,
    build_models, site_pipeline, evaluate_models, train_for_predictions
)

ARFF_PATH = r"D:\MRes Computing\7006 Dissertation Thesis\Datasets Autism\Code\flwr-hello\asd.arff"
TARGET = "Class/ASD"
SITE_COL = None  # Set to column name if your df has an explicit site identifier

class SklearnMetricsClient(fl.client.NumPyClient):
    def __init__(self, site_idx, X_site, y_site):
        self.site_idx = site_idx
        self.X_site = X_site
        self.y_site = y_site

    def get_parameters(self, config=None):
        # We are not aggregating sklearn parameters; return empty
        return []

    def fit(self, parameters, config):
        # Apply parameters from server (not used for sklearn models)
        #self.set_parameters(parameters)

        # Run your site pipeline
        models = build_models()
        X_sel, y_res, selected_features, scaler, selector = site_pipeline(self.X_site, self.y_site)

        # Evaluate via CV
        results = evaluate_models(models, X_sel, y_res)

        # Train models for predictions (optional)
        fitted, preds = train_for_predictions(models, X_sel, y_res)

        # Prepare metrics dictionary
        metrics = {
            "site": int(self.site_idx),
            "cv_results": results,
            "n_samples": int(len(y_res))
        }

        # Instead of sending full prediction arrays, send safe summaries
        pred_summary = {}
        for name, arr in preds.items():
            pred_summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }

        # Return parameters (empty), number of examples, and metrics
        return [], len(y_res), {
            "metrics_json": json.dumps(metrics),          # string
            "pred_summary_json": json.dumps(pred_summary) # string
        }
    
    def evaluate(self, parameters, config):
        # Optional: return a placeholder; primary reporting is in fit
        return 0.0, 0, {"msg": "evaluation not used"}

def main(site_idx: int):
    # Load and preprocess once
    df = load_arff(ARFF_PATH)
    df = preprocess_dataframe(df)
    sites = split_sites(df, target_col=TARGET, site_col=SITE_COL, num_sites=3, random_state=42)
    X_site, y_site = sites[site_idx]
    client = SklearnMetricsClient(site_idx, X_site, y_site)
    #fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=int, required=True, help="Client site index: 0, 1, or 2")
    args = parser.parse_args()
    main(args.client)
