import flwr as fl
from flwr.common import parameters_to_ndarrays

class Strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        parameters_aggregate , metric_aggregate = super().aggregate_fit(
            server_round,
            results,
            failures
        )
        
        self.aggr_weights = parameters_to_ndarrays(parameters_aggregate)
        print("randomShit " , parameters_aggregate, " " , metric_aggregate)
        return parameters_aggregate , metric_aggregate

strategy = Strategy()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

#print("aggrigated weights " , strategy.aggr_weights)

import pickle

# Load aggregated weights from file
with open("aggregated_weights.pickle", "wb") as f:
    pickle.dump(strategy.aggr_weights, f)
