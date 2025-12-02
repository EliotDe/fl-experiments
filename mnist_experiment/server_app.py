import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from mnist_experiment.task import ConvNN


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    fraction_train = context.run_config["fraction-train"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["lr"]

    global_model = ConvNN() 
    arrays = ArrayRecord(global_model.state_dict())
    
    strategy = FedAvg(fraction_train=fraction_train)

    result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds
    )

    print("\nSaving Final Model:")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "mnist_model.pt")



