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
