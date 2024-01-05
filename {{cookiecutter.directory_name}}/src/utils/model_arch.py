import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm.auto import tqdm
import neptune
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from mlxtend.plotting import plot_confusion_matrix
from utils import data_generator
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

class Module(nn.Module):
    def training_step(self, batch: tuple, criterion: str, model_generator):
        images, labels = batch 
        acc = 0
        out = self(images)
        if criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()

        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch: tuple, criterion: str, model_generator):
        images, labels = batch 
        acc = 0
        out = self(images)
        if criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs: list, criterion: str):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() 
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() 
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, criterion):
        logging.info("train_loss: {:.10f}, val_loss: {:.10f}, val_acc: {:.10f}".format(result['train_loss'], result['val_loss'], result['val_acc']))
        
@torch.no_grad()
def evaluate(model, val_loader, criterion: str, model_generator):
    model.eval()
    outputs = [model.validation_step(batch, criterion, model_generator) for batch in val_loader]
    return model.validation_epoch_end(outputs, criterion)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0000001, path='../models/', trace_func=print, pth_name="checkpoint.pth", lr=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.counter_c = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.getcwd() + "/models/"+pth_name
        self.trace_func = trace_func
        self.lr = lr
        
    def __call__(self, val_loss: float, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter % 10 == 0:
                self.counter_c = self.counter_c + 1
                self.lr = self.lr / 10
                if self.counter_c < 3:
                    self.counter = 0
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            logging.info("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model):
        #if self.verbose:
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ...')
        logging.info("Validation loss decreased ({:.10f} --> {:.10f}).  Saving model ...".format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.path + ".pth")
        self.val_loss_min = val_loss

def fit(epochs: int, lr: float, model_generator, train_loader, val_loader, opt_func: str, patience: int, criterion: str, 
        pth_name: str, neptune_ai: bool, neptune_ai_desc: str, tensor_board: bool, path_testset: str, save_path: str, path_pth: str, 
        batch_size: int, tt_transforms, load_w: bool):
    if neptune_ai:
        run = neptune.init_run(
            project="kocadurdu/TEST",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMjlmMzM3NC04NmI0LTRlYzktOTU3NS1kNjhlNTFmNDJlZTAifQ==",
            description = model_generator.name + " -  {}".format(neptune_ai_desc)
        )

        params = {
            "lr": lr,
            "bs": batch_size,
            "input_sz": model_generator.image_size,
            "n_classes": model_generator.num_class,
            "criterion" : criterion,
            "opt_func" : opt_func
        }
        run["parameters"] = params

    history = []
    if tensor_board:
        writer = SummaryWriter()

    device = get_default_device()
    model = model_generator.model
    if load_w:
        model.load_state_dict(torch.load(path_pth))
    model = to_device(model, device)
    

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    early_stopping = EarlyStopping(patience=patience, verbose=True, pth_name=pth_name, lr=lr)
    

    if opt_func == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), early_stopping.lr, alpha=0.9)
    elif opt_func == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif opt_func == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0.001, weight_decay=0.0)
    elif opt_func == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif opt_func == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif opt_func == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), early_stopping.lr, weight_decay=0.01)

    for epoch in range(epochs):
        logging.info("-"*50)
        logging.info("Epoch {} : lr = {:.10f}".format(epoch+1, early_stopping.lr))

        model.train()
        train_losses = []
        valid_losses = []
        pbar = tqdm(train_loader, desc='Epoch', leave=False, disable=False)
        for _indx, batch in enumerate(pbar):
            optimizer.zero_grad()
            loss, acc = model.training_step(batch, criterion, model_generator)
            if tensor_board:
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Acc/train", acc, epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
            if _indx % 10==0:
                pbar.set_description('Epoch {:03d} - loss: {:.10f}'.format(epoch+1, torch.stack(train_losses).mean() ))
                
            if neptune_ai:
                run["metrics/batch_loss"].append(loss)
        result = evaluate(model, val_loader, criterion, model_generator)
        
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result, criterion)
        history.append(result)
        
        train_loss = np.average(result['train_loss'])
        valid_loss = np.average(result['val_loss'])

        if neptune_ai:       
            run["metrics/train_loss"].append(train_loss)
            run["metrics/valid_loss"].append(valid_loss)
            if criterion != "MSELoss":
                run["metrics/accuracy"].append(np.average(result['val_acc']))
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            if neptune_ai:
                run["early_stopping"] = epoch

            logging.info("Epoch {} : Early stopped".format(epoch))
            logging.info("-"*50)
            break

    if tensor_board:
        writer.flush()
        writer.close()
    val_loss = [entry['val_loss'] for entry in history]
    #val_acc = [entry['val_acc'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    epochs = range(1, len(history) + 1)

    plt.clf()
    plt.subplot(1, 1, 1)
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.plot(epochs, train_loss, 'go-', label='Training Loss')
    plt.title('Validation Loss and Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/' + model_generator.name +'_'+ str(model_generator.image_size) +'_val_train_loss.png', dpi=600)


    logging.info("Evaluation started...")

    model = model_generator.model
    model.load_state_dict(torch.load(os.path.join(path_pth, model_generator.name)+ '.pth'))
    model.eval()
    model = model.to(device)
    
    cls_names = []
    for i in range(0, int(model_generator.num_class)):
        cls_names.append(str(i))
    idx_to_class = {}
    for i in range(0, len(cls_names), 1):
        idx_to_class[i] = cls_names[i]
        
    ds_val = data_generator.Folder_Dataset(path_testset, transform=tt_transforms)
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
    #y_falses_df.to_csv(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'.csv', index=False)

    y_preds_ = list(map(lambda x: idx_to_class[x], y_preds))
    y_trues_ = list(map(lambda x: idx_to_class[x], y_trues))

    sns.set(style='white')
    cm = confusion_matrix(y_trues_, y_preds_)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 40},
                cmap=sns.color_palette(['tomato', 'palegreen'], as_cmap=True), cbar=False,
                yticklabels=['Benign', 'Malignant'], xticklabels=['Benign', 'Malignant'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=15, length=0)

    ax.set_title('Confusion Matrix '+ str(model_generator.name) +'_'+ str(model_generator.image_size), size=20, pad=20)
    ax.set_xlabel('Predicted Values', size=15)
    ax.set_ylabel('Actual Values', size=15)

    additional_texts = ['(True Benign)', '(False Malignant)', '(False Benign)', '(True Malignant)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                ha='center', va='top', size=18)
    plt.tight_layout()
    plt.savefig(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_conf_mtrx.png', dpi=600)
    plt.clf()

    """
    conf_mtrx = confusion_matrix(y_trues_, y_preds_)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mtrx,
                                    colorbar=False,
                                    show_absolute=True,
                                    show_normed=False,
                                    class_names = cls_names,
                                    figsize=(9,9)
                                    )
    plt.xlabel('Predicted Label') 
    plt.ylabel('True Label') 
    plt.title("Confussion Matrix - " + model_generator.name)
    plt.savefig(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_conf_mtrx.png', dpi=600)
    """

    clf_report = classification_report(y_preds_, y_trues_, output_dict=True)
    ax_ = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='Blues', linewidths=0.9, fmt=".2f")
    ax_.get_figure().savefig(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_creport.png', dpi=600)

    roc_disp = RocCurveDisplay.from_predictions(y_preds, y_trues, name="roc_curve")
    roc_disp.figure_.savefig(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_roc_curve.png', dpi=600)
    
        
    if neptune_ai:
        run["eva/val_train_loss"].upload(save_path + '/' + model_generator.name +'_'+ str(model_generator.image_size) +'_val_train_loss.png')
        run["eva/conf_matrix"].upload(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_conf_mtrx.png')
        run["eva/clf_report"].upload(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_creport.png')
        run["eva/roc_curve"].upload(save_path + '/'+ model_generator.name +'_'+ str(model_generator.image_size) +'_roc_curve.png')

        run.stop()
    return history, model

