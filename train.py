import sys
path_list=['/raid/lcq/LLib']
sys.path.append(path_list[0])
from logger.Logger import Logger
from torch.utils.data import DataLoader
from data import CrackDataSet
from net import resnet18 as Net
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()#acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)#per-class accuracy acc = np.diag(hist) / hist.sum(1)，np.nanmean(acc)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu) #iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))，np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum() #freq = hist.sum(1) / hist.sum() (freq[freq > 0] * iu[freq > 0]).sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



def create_train_val_test_list(src_dir,suffix='.jpg'):
    import os
    from sklearn.model_selection import train_test_split
    imgs=os.listdir(src_dir)
    imgs = [img for img in imgs if img.endswith(suffix)]
    train, val=train_test_split(imgs,test_size=2/6, random_state=42)
    val, test=train_test_split(val,test_size=1/2, random_state=42)
    return train,val,test

def train_one_epoch(model,dataloader,criterion,opt,GPU,epoch,current_lr,n_class=2,lr_change=None):
    model.train()
    if lr_change and epoch in lr_change:
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr / 10
        current_lr = current_lr / 10

    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0

    for index,data in enumerate(dataloader):
        im,target=data
        if GPU:
            im=im.to(GPU)
            target=target.to(GPU)
        opt.zero_grad()
        feat=model(im)
        loss=criterion(feat,target)
        loss.backward()
        opt.step()
        pre=feat.argmax(1).cpu().numpy()
        label=target.cpu().numpy()
        train_loss += im.size(0)*loss.item()
        for p,l in zip(pre,label):
            acc, acc_cls, mean_iu, fwavacc=label_accuracy_score(l,p,n_class=n_class)
            train_acc+=acc
            train_acc_cls+=acc_cls
            train_mean_iu+=mean_iu
            train_fwavacc+=fwavacc
    return train_loss,train_acc,train_acc_cls,train_mean_iu,train_fwavacc,current_lr

def val(model,dataloader,criterion,GPU,n_class=2):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    with torch.no_grad():
        for data in dataloader:
            im, target = data
            if GPU:
                im = im.to(GPU)
                target = target.to(GPU)
            feat = model(im)
            loss = criterion(feat,target)
            eval_loss += im.size(0)*loss.item()

            pre = feat.argmax(1).cpu().numpy()
            label = target.cpu().numpy()
            for p, l in zip(pre, label):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(l, p, n_class=n_class)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc
    return eval_loss,eval_acc,eval_acc_cls,eval_mean_iu,eval_fwavacc

def main():
    img_dir=r'/raid/lcq/dataset/crack/imgs'
    label_dir=r'/raid/lcq/dataset/crack/labels'
    save_path='./weights/result_without_pre.plt'
    bach_size=64
    GPU_ID=1
    epoches=400
    lr=0.001
    lr_change=[]
    log_dir='./log'
    logger=Logger(log_dir, '', 'crack')
    logger.train()
    if GPU_ID>=0:
        GPU=torch.device('cuda:{}'.format(GPU_ID))
    else:
        GPU=None
    train_list,val_list,test_list=create_train_val_test_list(img_dir)

    train_set=CrackDataSet(img_dir=img_dir,imgs=train_list,label_dir=label_dir)
    train_data_loader=DataLoader(train_set,bach_size,shuffle=True)
    len_train=len(train_set)

    val_set=CrackDataSet(img_dir=img_dir,imgs=val_list,label_dir=label_dir)
    val_data_loader=DataLoader(val_set,32,False)
    len_val=len(val_set)


    model=Net(pretrained=True,num_classes=2)
    opt=SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    if GPU:
        model=model.to(GPU)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epoches):
        train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc,current_lr=train_one_epoch(model,train_data_loader,criterion,opt,GPU,epoch,current_lr=lr,lr_change=lr_change)
        eval_loss, eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc=val(model,val_data_loader,criterion,GPU)
        log='[{}/{}] Train Loss:{:.5f},Train Acc:{:.5f},Train cls acc:{:.5f},Train Mean IU:{:.5f},Train fwavacc:{:.5f}'.format(epoch,epoches,train_loss/len_train,train_acc/len_train,eval_acc_cls/len_train,train_mean_iu/len_train,train_fwavacc/len_train)
        log2='[{}/{}] Val Loss:{:.5f},Val Acc:{:.5f},Val cls acc:{:.5f},Val mean iu:{:.5f},Val fwavacc:{:.5f}'.format(epoch,epoches,eval_loss/len_val, eval_acc/len_val, eval_acc_cls/len_val, eval_mean_iu/len_val, eval_fwavacc/len_val)
        logger.info(log)
        logger.info(log2)

    torch.save(model.state_dict(), save_path)
if __name__ == '__main__':
    main()





