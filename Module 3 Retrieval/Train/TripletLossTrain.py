import matplotlib.pyplot as plt
import torch
from utils import TripletLoss, l2norm
from models.GarnmentsNet import GarnmentsMobileNet, GarnmentsResNet
from ComposeDataset import FeedCoupleDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import platform
from neptuneLogger import NetpuneLogger
import time


def train(hyper, limit_descending, limit_patience, fine_tune: bool = False):

    neptune = NetpuneLogger(hyper)

    category = hyper['category']

    if platform.system() == 'Windows':
        dataset_path = 'C:\\Users\\pales\\Desktop\\DatasetCV'
        annotation_file_train = dataset_path + '\\' + category + '\\train_pairs.txt'
        annotation_file_test = dataset_path + '\\' + category + '\\test_pairs_paired.txt'
        root_dir = 'C:\\Users\\pales\\Desktop\\DatasetCV\\' + category + '\\images'
        seconds = time.time()
        local = time.ctime(seconds)
        weights_path = "data\\weights-pytorch-" + category + f"{local}.pth"
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        dataset_path = '../DatasetFolder/DatasetCV'
        annotation_file_train = dataset_path + f'/{category}/train_pairs.txt'
        annotation_file_test = dataset_path + f'/{category}/test_pairs_paired.txt'
        root_dir = f'../DatasetFolder/DatasetCV/{category}/images'
        seconds = time.time()
        local = time.ctime(seconds)
        weights_path = "data/weights-pytorch-" + category + f"{local}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Fine Tuning mode : ", fine_tune)
    print("model : ", hyper["model"])

    if hyper['model'] == "ResNet":
        model = GarnmentsResNet(fine_tune).to(device)
    elif hyper['model'] == "MobileNet":
        model = GarnmentsMobileNet(fine_tune).to(device)

    if not fine_tune:
        for param in model.ConvLayer.parameters():
            param.requires_grad = False
        model.ConvLayer.eval()

    preprocess_for_items = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_for_parsed = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training_data = FeedCoupleDataset(annotations_file=annotation_file_train,
                                      root_dir=root_dir, transform_items=preprocess_for_items,
                                      transform_parsed=preprocess_for_parsed)

    validation_data = FeedCoupleDataset(annotations_file=annotation_file_test,
                                        root_dir=root_dir, transform_items=preprocess_for_items,
                                        transform_parsed=preprocess_for_parsed)

    train_loader = DataLoader(
        training_data, batch_size=hyper["batch_size"], shuffle=True, num_workers=4
    )

    validation_loader = DataLoader(
        validation_data, batch_size=hyper["batch_size"], shuffle=True, num_workers=4
    )

    if fine_tune:
        optimizer = hyper["optimizer"](model.parameters(), lr=hyper["learning_rate"], weight_decay=hyper["wd"])
    elif not fine_tune:
        optimizer = hyper["optimizer"](model.Linear.parameters(), lr=hyper["learning_rate"], weight_decay=hyper["wd"])

    triplet_loss = TripletLoss(margin=hyper["margin"], max_violation=hyper["max_violation"])

    triplet_loss_validation = TripletLoss(margin=hyper["margin"], max_violation=hyper["max_violation"])

    num_epochs = hyper["num_epochs"]

    train_count = len(training_data)

    print(f'Number of images in the training set: {train_count}')

    training_data.set_flag(False)
    best_loss = 0xffffffffffffffff
    patience = 0
    best_weights = model.state_dict()
    keep_descending = 0
    for epoch in range(int(num_epochs)):

        torch.cuda.empty_cache()
        if not fine_tune:
            model.Linear.train()
            model.ConvLayer.eval()
        elif fine_tune:
            model.train()

        l = 0.0
        counter = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc="Training...")):
            x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()

            out1, out2 = model.forward(x1, x2)
            out1 = l2norm(out1, dim=-1)
            out2 = l2norm(out2, dim=-1)
            score = torch.mm(out1, out2.t())
            loss = triplet_loss(score)
            loss.backward()
            optimizer.step()
            l += loss.item()
            counter += 1

        neptune.set_train(l / counter)
        print("Epoch: {} sum of loss :{} best :{} ".format(epoch, l / counter, best_loss))

        l = 0.0
        counter = 0.0
        model.eval()
        with torch.no_grad():
            for ii, batch in enumerate(tqdm(validation_loader, desc="Validating...")):
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)

                out1, out2 = model.forward(x1, x2)
                out1 = l2norm(out1, dim=-1)
                out2 = l2norm(out2, dim=-1)
                score = torch.mm(out1, out2.t())
                loss_valid = triplet_loss_validation(score)
                l += loss_valid.item()
                counter += 1

        neptune.set_validation(l / counter)



        if best_loss < (l / counter):
            patience += 1
            keep_descending = 0
        elif best_loss > (l / counter):
            if keep_descending == 0:
                keep_initial_weights = model.state_dict()
                keep_initial_loss = l / counter

            keep_descending += 1

            if 0 < patience <= limit_patience:
                print(f"UPDATING WEIGHTS DUE TO PATIENCE : {patience} WITH LOSS : {best_loss} ...")
                # saving...
                torch.save(best_weights, weights_path)
                neptune.update(weights_path, category)
                neptune.logfile(best_loss)
                # saving...
            patience = 0
            best_loss = l / counter
            best_weights = model.state_dict()

        print('keep descending : ', keep_descending)
        if keep_descending == limit_descending:
            print(f"UPDATING WEIGHTS DUE TO KEEP DESCENDING...WITH LOSS : {keep_initial_loss}")
            # saving...
            torch.save(keep_initial_weights, weights_path)
            neptune.update(weights_path, category)
            neptune.logfile(keep_initial_loss)
            # saving...
            keep_initial_weights = model.state_dict()
            keep_initial_loss = l / counter
            keep_descending = 1

        if patience >= limit_patience:
            print(f"LIMIT OF PATIENCE REACHED...EXIT")
            break



    neptune.destroy()
