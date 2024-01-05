import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from utils import model_arch
from utils import data_generator
import logging
from tqdm.auto import tqdm

class executor:
    def __init__(self, path_dataset_train, path_dataset_val, batch_size: int, num_threads: int, device_id: int, 
                 num_epochs: int, lr: float, patience: int, opt_func: str, 
                 criterion: str, normalize: bool,
                 neptune_ai: bool, neptune_ai_desc: str, tensor_board: bool, 
                 path_testset: str, save_path: str, path_pth:str , load_w: bool) -> None:
        
        self.path_dataset_train = path_dataset_train
        self.path_dataset_val = path_dataset_val
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.num_epochs = num_epochs
        self.lr =  lr
        self.patience = patience
        self.opt_func = opt_func
        self.criterion = criterion
        self.normalize = normalize
        self.neptune_ai = neptune_ai
        self.neptune_ai_desc = neptune_ai_desc
        self.tensor_board = tensor_board
        self.path_testset = path_testset
        self.save_path = save_path
        self.path_pth = path_pth
        self.load_w = load_w

        logging.info("Parameters...")
        logging.info("path_dataset_train : {}".format(self.path_dataset_train))
        logging.info("path_dataset_val : {}".format(self.path_dataset_val))
        logging.info("batch_size : {}".format(self.batch_size))
        logging.info("num_threads : {}".format(self.num_threads))
        logging.info("device_id : {}".format(self.device_id))
        logging.info("num_epochs : {}".format(self.num_epochs))
        logging.info("lr : {}".format(self.lr))
        logging.info("patience : {}".format(self.patience))
        logging.info("opt_func : {}".format(self.opt_func))
        logging.info("criterion : {}".format(self.criterion))
        logging.info("normalize : {}".format(self.normalize))
        logging.info("neptune_ai : {}".format(self.neptune_ai))
        logging.info("neptune_ai_desc : {}".format(self.neptune_ai_desc))
        logging.info("tensor_board : {}".format(self.tensor_board))
        logging.info("path_testset : {}".format(self.path_testset))
        logging.info("save_path : {}".format(self.save_path))
        logging.info("path_pth : {}".format(self.path_pth))
        logging.info("load_w : {}".format(self.load_w))


    def execute(self, model_generator):
        tt_transforms_train = transforms.Compose(
            [
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.ToTensor(),
            ]
        )        
        tt_transforms = transforms.Compose(
            [
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.ToTensor(),
            ]
        )
            
        if self.normalize:
            ds_train_1 = data_generator.Folder_Dataset(self.path_dataset_train, transform=tt_transforms)
            dl_train_1 = DataLoader(ds_train_1, self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
            mean, std = batch_mean_and_sd(dl_train_1)

            tt_transforms_train = transforms.Compose(
                [
                    transforms.Resize((model_generator.image_size, model_generator.image_size)),
                    transforms.RandomRotation(degrees=(0, 15)),
                    transforms.RandomVerticalFlip(p=0.3), 
                    transforms.RandomHorizontalFlip(p=0.3),
                    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean.tolist(), std.tolist()),
                ]
            )
            tt_transforms = transforms.Compose(
                [
                    transforms.Resize((model_generator.image_size, model_generator.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean.tolist(), std.tolist()),
                ]
            )

        ds_train = data_generator.Folder_Dataset(self.path_dataset_train, transform=tt_transforms_train)
        ds_val = data_generator.Folder_Dataset(self.path_dataset_val, transform=tt_transforms)

        dl_train = DataLoader(ds_train, self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        dl_val = DataLoader(ds_val, self.batch_size, num_workers = 0, pin_memory = True, shuffle=False)

        pth_name = model_generator.name
        history, model_fit = model_arch.fit(self.num_epochs, self.lr, model_generator, dl_train, dl_val, self.opt_func, self.patience, self.criterion,  
                                            pth_name, self.neptune_ai, self.neptune_ai_desc, self.tensor_board,
                                            self.path_testset, self.save_path, self.path_pth, self.batch_size, tt_transforms, 
                                            self.load_w)
        

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
