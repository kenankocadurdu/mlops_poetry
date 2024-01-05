import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from utils import data_generator
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from mlxtend.plotting import plot_confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class executor:
    def __init__(self, path_testset, save_path, path_pth:str, normalize: bool, path_trainset: str, product: bool) -> None:
        self.path_testset = path_testset
        self.save_path = save_path
        self.path_pth = path_pth
        self.normalize = normalize
        self.path_trainset = path_trainset
        self.product = product

        logging.info("-"*50)
        logging.info("Parameters...")
        logging.info("save_path : {}".format(self.save_path))
        logging.info("path_pth : {}".format(self.path_pth))
        logging.info("normalize : {}".format(self.normalize))
        logging.info("path_trainset : {}".format(self.path_trainset))
        logging.info("product : {}".format(self.product))

    def execute(self, model_generator, batch_size):
        tt_transforms = transforms.Compose(
            [
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.ToTensor()
            ]
        )
        if self.normalize:
            ds_train_1 = data_generator.Folder_Dataset(self.path_trainset, transform=tt_transforms)
            dl_train_1 = DataLoader(ds_train_1, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
            mean, std = batch_mean_and_sd(dl_train_1)
            tt_transforms = transforms.Compose(
                [
                    transforms.Resize((model_generator.image_size, model_generator.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean.tolist(), std.tolist()),
                ]
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_generator.model
        model.load_state_dict(torch.load(os.path.join(self.path_pth, model_generator.name)+ '.pth'))
        model.eval()
        model = model.to(device)

        cls_names = []
        for i in range(0, int(model_generator.num_class)):
            cls_names.append(str(i))
        idx_to_class = {}
        for i in range(0, len(cls_names), 1):
            idx_to_class[i] = cls_names[i]


        ds_val = data_generator.Folder_Dataset(self.path_testset, transform=tt_transforms)
        dl_val = DataLoader(ds_val, batch_size, num_workers = 0, pin_memory = True, drop_last=True,  shuffle=False)
        y_preds = []
        y_trues = []
        y_falses = []
        y_f_preds = []
        y_f_trues = []
        with torch.no_grad():
            for i, (image, label) in enumerate(tqdm(dl_val, 0)):
                image = image.to(device)
                output = model(image)
                for k in range(0, batch_size):
                    y_pred = torch.argmax(output[k]).cpu().numpy().item()
                    y_true = label[k].numpy().item()
                    if y_pred != y_true:
                        fname, _ = dl_val.dataset.images[i * batch_size + k]
                        y_falses.append(fname)
                        y_f_preds.append(y_pred)
                        y_f_trues.append(y_true)
                    y_preds.append(y_pred)
                    y_trues.append(y_true)
        #y_falses_df = pd.DataFrame({'filename': y_falses, 'y_true': y_f_trues, 'y_pred': y_f_preds})
        #y_falses_df.to_csv(self.save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'.csv', index=False)

        y_preds_ = list(map(lambda x: idx_to_class[x], y_preds))
        y_trues_ = list(map(lambda x: idx_to_class[x], y_trues))


        sns.set(style='white')
        cm = confusion_matrix(y_trues_, y_preds_)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 40},
                    cmap=sns.color_palette(['tomato', 'palegreen'], as_cmap=True), cbar=False,
                    yticklabels=cls_names, xticklabels=cls_names, ax=ax)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelsize=15, length=0)

        ax.set_title('Confusion Matrix '+ str(model_generator.name) +'_'+ str(model_generator.image_size), size=20, pad=20)
        ax.set_xlabel('Predicted Values', size=15)
        ax.set_ylabel('Actual Values', size=15)

        additional_texts = ['(True 0)', '(False 1)', '(False 0)', '(True 1)']
        for text_elt, additional_text in zip(ax.texts, additional_texts):
            ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                    ha='center', va='top', size=18)
        plt.tight_layout()
        plt.savefig(self.save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_conf_mtrx_1.png', dpi=600)
        plt.clf()

def batch_mean_and_sd(dl_val):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    logging.info("-"*50)
    logging.info("Normalization started")
    #for images, _ in dl_val:
    for i, (images, label) in enumerate(tqdm(dl_val, 0)):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2) 
    logging.info("mean: {} - std: {} ".format(mean, std))
    logging.info("Normalization finished")       
    return mean, std
