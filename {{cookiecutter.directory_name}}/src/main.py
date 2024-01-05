
import cv2
import glob
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from utils import model_generator
import torch

@hydra.main(config_path="../config", config_name="preprocessing", version_base=None)
def do_preprocessing(config: DictConfig):
    logging.info("-"*50)
    logging.info("Image preprocessing started...")
    pre_processor = instantiate(config.preprocessing)
    dir_root = os.getcwd()
    sum_saved_img = 0
    for split_1 in config.data["dirs"]:
        logging.info("Path : {}...".format(split_1))
        for split_2 in config.data["classes"]:
            path_c = split_1 + "/" + split_2
            logging.info("{}".format(path_c))
            image_paths = glob.glob(os.path.join(dir_root, config.data["raw_path"], path_c + "/*."+ config.data["file"]))          
            for _indx, path in enumerate(tqdm(image_paths)):
                path_split = path.split("/")
                save_path = os.path.join(dir_root, config.data["processed_path"], path_c, path_split[-1])
                image = cv2.imread(path, 0)
                output_image = pre_processor.do_process(image)
                cv2.imwrite(save_path, output_image)
                sum_saved_img = sum_saved_img + 1
    logging.info("{} image saved".format(sum_saved_img))  
    logging.info("Image preprocessing finished...")   
    logging.info("-"*50)

@hydra.main(config_path="../config", config_name="training", version_base=None)
def do_training(config: DictConfig):
    logging.info("-"*50)
    executer_training = instantiate(config.training)
    model_names = OmegaConf.to_container(config.model, resolve=True)
    model_params = OmegaConf.to_container(config.model_params, resolve=True)
    for model_name in model_names:
        model_gen = model_generator.generator(model_name, model_params['num_class'], model_params['image_size'])
        logging.info("Process training with "+ model_gen.name +" started...")
        executer_training.execute(model_gen)
        logging.info("Process training with "+ model_gen.name +" finished...")

@hydra.main(config_path="../config", config_name="training", version_base=None)
def do_evaluation(config: DictConfig):
    
    executer_training = instantiate(config.training)
    executer_prediction = instantiate(config.evaluation)
    model_names = OmegaConf.to_container(config.model, resolve=True)
    model_hyperparams = OmegaConf.to_container(config.model_params, resolve=True)
    for model_name in model_names:
        model_gen = model_generator.generator(model_name,  model_hyperparams['num_class'], model_hyperparams['image_size'])
        logging.info("Process prediction with "+ model_gen.name +" started...")
        executer_prediction.execute(model_gen, executer_training.batch_size)
        logging.info("Process prediction with "+ model_gen.name +" finished...")

if __name__ == "__main__":
    torch.manual_seed(43)
    do_preprocessing()
    do_training()
    #do_evaluation()