from flwr_datasets import FederatedDataset 
from flwr_datasets.partitioner import IidPartitioner
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor 


""" The Model  """
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        

        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)#reshape(out.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 


fds = None

def GetPartition(partition_id: int, num_partitions: int):
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
                dataset="mnist", 
                partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5,), (0.5,))]
    )

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader



def train(model, trainloader, epochs, lr, device):
    model.to(device)
    
    # Initialise optimiser and loss mechanism
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimiser.zero_grad()
            # Get initial loss
            loss = criterion(model(images), labels)
            # Backpropagation
            loss.backward()
            optimiser.step()
            # Add to loss
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss



def test(model, testloader, device):
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"]
            labels = batch["label"]
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
