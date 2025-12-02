import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from mnist_experiment.task import ConvNN, GetPartition
from mnist_experiment.task import test as test_fn 
from mnist_experiment.task import train as train_fn 


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    model=ConvNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict()) # Initialise with received weights
    device = torch.device("cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = GetPartition(partition_id, num_partitions) # Don't need testloader

    # Call the training function
    train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
    )


    # Construct and return the reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics_record = MetricRecord({"train_loss": train_loss, "num-examples": len(trainloader.dataset) })
    content = RecordDict({"arrays": model_record, "metrics": metrics_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # Load Model and Initialise
    model = ConvNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")
    model.to(device)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = GetPartition(partition_id, num_partitions)

    # Call evaluation function
    eval_loss, eval_acc = test_fn(
            model,
            valloader,
            device
    )

    # Construct and return Reply Message
    metrics = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
