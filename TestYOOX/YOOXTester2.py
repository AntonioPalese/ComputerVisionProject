import torch
import  numpy as np
from ComposeDataset import FeedImagesToInput
from torchvision import transforms
from models.GarnmentsNet import ClassNetGarnments
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import  tqdm
from utils import l2norm
from UtilsExtractParsed import make_csv, make_input_csv
import os
import cv2


class YOOXTester(torch.nn.Module):
    def __init__(self, root, imgs_path, category, recall, platform, weights_path):
        super(YOOXTester, self).__init__()

        self.root = root
        self.imgs_path = imgs_path
        self.category = category
        self.csv_path = 'D:\datasetCV_nomanichini_classificato' + os.sep + 'final_yoox-' + self.category + '.csv'
        self.preprocess_for_items = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.preprocess_for_parsed = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model = ClassNetGarnments().to(self.device)
        self.model.load_state_dict(torch.load(weights_path))

        self.model.eval()



        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.mean = torch.tensor(self.mean)
        self.std = torch.tensor(self.std)

        self.std.resize_(1, 1, 3)
        self.mean.resize_(1, 1, 3)

        self.recall = recall

        self.platform = platform

    def forward(self, verbose : bool = False):
        test_data = FeedImagesToInput(annotations_file=self.csv_path, root_dir=self.root, category="denim",
                                      filtering=True, transform_items=self.preprocess_for_items)
        batch_size = 16
        test_loader = torch.utils.data.DataLoader(dataset=test_data, num_workers=4, shuffle=True, batch_size=batch_size)

        file_csv = make_input_csv(self.category)
        data = pd.read_csv(file_csv, header=None, sep=',')

        test_data.set_flag(False)

        for i in range(data.shape[0]):
            immagine = data.iloc[i, 0]
            es = plt.imread(immagine)
            # es = np.delete(es, 3, 2)
            es_toprint = es.copy()
            matches = torch.ones((1, 224, 224, 3))
            matches_scores = torch.tensor([[1]])
            matches_items = []

            plt.imshow(es)
            plt.show()

            es = torch.from_numpy(es.copy()).permute(2, 0, 1).type(torch.float32)

            es = self.preprocess_for_parsed(es)
            es = es.to(self.device)
            es = es.unsqueeze(0)

            for img in tqdm(test_loader, desc='Testing...'):
                if verbose:
                    im = (img[0].permute(1, 2, 0) * self.std + self.mean).type(torch.uint8)
                    plt.imshow(im)
                    plt.show()

                x = img.to(self.device)

                out1, out2 = self.model.forward(es, x)
                out1 = l2norm(out1, dim=-1).cpu()
                out2 = l2norm(out2, dim=-1).cpu()
                score = torch.mm(out1, out2.t())

                score = score.squeeze(0)
                m = (score > 0.20)
                x = x[m]
                score = score[m]
                score = score.detach().cpu()
                for j in range(score.shape[0]):
                    vis = (x[j].detach().cpu().squeeze().permute(1, 2, 0) * self.std + self.mean).type(torch.uint8)

                    matches = torch.cat((matches, vis.unsqueeze(0)), 0)
                    s = score[j]
                    s = s.unsqueeze(0).unsqueeze(0)
                    matches_scores = torch.cat((matches_scores, s), 0)

            matches = matches[1:].numpy()
            matches_scores = matches_scores[1:].numpy()

            # print('matches score prima dell order : ', matches_scores)
            matches_scores = np.squeeze(matches_scores, 1)
            matches_scores = np.argsort(matches_scores)[::-1]
            # print('matches score dopo dell order : ', matches_scores)

            print(f"Immagini confrontate : {len(test_data)}")

            # es_toprint = cv2.applyColorMap(es_toprint, cv2.COLORMAP_JET)
            es_toprint = cv2.cvtColor(es_toprint, cv2.COLOR_BGR2RGB)
            if verbose:
                if self.platform == 'Windows':
                    cv2.imshow("Immagine", es_toprint)
                    if cv2.waitKey(0) == ord('q'):
                        cv2.destroyAllWindows()
                        continue
                    cv2.destroyAllWindows()
                elif self.platform == 'Linux':
                    plt.imshow(es_toprint)
                    plt.show()

            for ii in range(self.recall):
                if (ii == len(matches)):
                    break
                index = matches_scores[ii]
                im = matches[index]
                im = im.astype(np.uint8)
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                plt.imsave(f'outputs\\group{i}_img{ii}.jpg', im)
                if verbose:
                    if self.platform == 'Windows':
                        cv2.imshow("Immagine Output", im)
                        if cv2.waitKey(0) == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    elif self.platform == 'Linux':
                        plt.imshow(im)
                        plt.show()

            if self.platform == 'Windows':
                cv2.destroyAllWindows()





