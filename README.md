# mnist-experiment: A Flower / PyTorch app

Following the flower docs I have implemented a simple federated learning simulation to collaboratively train a model on the MNIST dataset. 

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```
## Run with the Simulation Engine

In the `mnist-experiment` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## MNIST-Results
Current model achieved an aggregated accuracy of ~97% over 3 server rounds and 3 local epochs.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
=======
