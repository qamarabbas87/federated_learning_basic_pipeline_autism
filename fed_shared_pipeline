# shared_pipeline.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE

# -----------------------------
# Load and preprocess (shared)
# -----------------------------

def load_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode bytes -> strings
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")
    return df

def preprocess_dataframe(df):
    # Replace missing markers
    df = df.replace("?", np.nan)
    # Encode all object columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def split_sites(df, target_col="Class/ASD", site_col=None, num_sites=3, random_state=42):
    # If a site column exists, use it; else even split
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop ID-like columns
    id_like_cols = [c for c in X.columns if X[c].nunique()/len(X) > 0.95]
    X = X.drop(columns=id_like_cols)

    # Optional explicit site split
    sites = []
    if site_col and site_col in df.columns:
        unique_sites = sorted(df[site_col].unique())
        if len(unique_sites) < num_sites:
            raise ValueError(f"Found {len(unique_sites)} unique sites in '{site_col}', need {num_sites}.")
        for s in unique_sites[:num_sites]:
            mask = (df[site_col] == s)
            sites.append((X[mask].copy(), y[mask].copy()))
    else:
        # Even split by stratified index
        X_temp = X.copy()
        y_temp = y.copy()
        n = len(X_temp)
        idx = np.arange(n)
        np.random.seed(random_state)
        np.random.shuffle(idx)
        chunks = np.array_split(idx, num_sites)
        for ch in chunks:
            sites.append((X_temp.iloc[ch].copy(), y_temp.iloc[ch].copy()))

    return sites  # List of (X_site, y_site)

def build_models():
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    svm = SVC(C=1, kernel="rbf", probability=True, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, use_label_encoder=False)
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)

    estimators = [("rf", rf), ("svm", svm), ("xgb", xgb)]
    stack = StackingClassifier(estimators=estimators,
                               final_estimator=RandomForestClassifier(random_state=42),
                               cv=5)
    models = {
        "Random Forest": rf,
        "SVM": svm,
        "XGBoost": xgb,
        "Naive Bayes": nb,
        "KNN": knn,
        "Stacking Ensemble": stack
    }
    return models

def site_pipeline(X_site, y_site, k_features=10, random_state=42):
    # Imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="most_frequent")
    X_imp = pd.DataFrame(imputer.fit_transform(X_site), columns=X_site.columns)

    print(f"Site {len(y_site)} samples before balancing.")
    #print("X_Site shape : " + X_site.shape)
    #print("X_imp shape : " + X_imp.shape)

    # Balance
    #sm = BorderlineSMOTE(random_state=42)
    #X_res, y_res = sm.fit_resample(X_imp, y_site)
    X_res, y_res = X_imp, y_site

    print(f"Site {len(y_res)} samples after balancing.")
    #print("X_res shape : " + X_res.size)

    # Scale + select
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    selector = SelectKBest(score_func=f_classif, k=min(k_features, X_scaled.shape[1]))
    X_sel = selector.fit_transform(X_scaled, y_res)
    selected_features = X_imp.columns[selector.get_support()].tolist()

    return X_sel, y_res, selected_features, scaler, selector

def evaluate_models(models, X_sel, y_res, random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }
    results = []
    for name, model in models.items():
        scores = cross_validate(model, X_sel, y_res, cv=cv, scoring=scoring)
        results.append({
            "Model": name,
            "Accuracy": float(np.mean(scores["test_accuracy"])),
            "Precision": float(np.mean(scores["test_precision"])),
            "Recall": float(np.mean(scores["test_recall"])),
            "F1": float(np.mean(scores["test_f1"])),
            "ROC-AUC": float(np.mean(scores["test_roc_auc"]))
        })
    return results

def train_for_predictions(models, X_sel, y_res):
    # Fit each model and return predict_proba on X_sel (used for federated stacking)
    fitted = {}
    preds = {}
    for name, model in models.items():
        model.fit(X_sel, y_res)
        if hasattr(model, "predict_proba"):
            preds[name] = model.predict_proba(X_sel)[:, 1]
        else:
            # Fallback: use decision_function scaled to [0,1]
            dec = model.decision_function(X_sel)
            dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-8)
            preds[name] = dec
        fitted[name] = model
    return fitted, preds
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
# flwr_server.py
import json
import flwr as fl

class MetricsCollectStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=3,
            min_evaluate_clients=0,
            min_available_clients=3
        )

    def aggregate_fit(self, rnd, results, failures):
        # Collect metrics from clients
        collected = []
        for _, fitres in results:
            metrics_json = fitres.metrics.get("metrics_json", "{}")
            try:
                parsed = json.loads(metrics_json)
            except Exception:
                parsed = {}
            collected.append(parsed)

        # Print progress per round
        print(f"\n=== Round {rnd} completed ===")
        for site in collected:
            print(f"Site {site.get('site')} | Samples: {site.get('n_samples')} | CV Results:")
            for res in site.get("cv_results", []):
                print(f"  {res['Model']}: Acc={res['Accuracy']:.3f}, F1={res['F1']:.3f}, AUC={res['ROC-AUC']:.3f}")

        return [], {}

def main():
    strategy = MetricsCollectStrategy()
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5)  # run 5 rounds
    )

if __name__ == "__main__":
    main()
