from TripletLossTrain import train
import torch
import json


if __name__ == "__main__":
    optims = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    bools = {"True": True, "False": False}

    f = open('config.json', 'r')
    hyperparameters = json.load(f)

    hyperparameters["optimizer"] = optims[hyperparameters["optimizer"]]
    hyperparameters["max_violation"] = bools[hyperparameters["max_violation"]]
    hyperparameters["fine_tune"] = bools[hyperparameters["fine_tune"]]

    train(hyperparameters, limit_descending=hyperparameters["limit_descending"],
          limit_patience=hyperparameters["limit_patience"], fine_tune=hyperparameters["fine_tune"])
