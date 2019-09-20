from train import label_accuracy_score
from data import CrackDataSet
from torch.utils.data import DataLoader
from net import resnet18 as Net
import numpy as np
from train import create_train_val_test_list
import torch

def val(model, dataloader,GPU,n_class=2):
    model.eval()
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
            pre = feat.argmax(1).cpu().numpy()
            label = target.cpu().numpy()
            for p, l in zip(pre, label):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(l, p, n_class=n_class)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc
    return eval_acc,eval_acc_cls,eval_mean_iu,eval_fwavacc



def main():
    img_dir=r'/raid/lcq/dataset/crack/imgs'
    label_dir=r'/raid/lcq/dataset/crack/labels'
    weight_path=r'./weights/result.plt'
    GPU_ID=1
    GPU = torch.device('cuda:{}'.format(GPU_ID))
    _, _, test_list = create_train_val_test_list(img_dir)
    test_set = CrackDataSet(img_dir=img_dir, imgs=test_list, label_dir=label_dir)
    test_data_loader = DataLoader(test_set, 32, False)
    len_test = len(test_set)
    model = Net(pretrained=True, num_classes=2)
    model.load_state_dict(torch.load(weight_path))
    model = model.to(GPU)
    eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc=val(model, test_data_loader, GPU, n_class=2)
    log='Test Acc:{:.5f},Test cls acc:{:.5f},Test mean iu:{:.5f} Test fwavacc:{:.5f}'.format(
        eval_acc/len_test,
        eval_acc_cls/len_test,
        eval_mean_iu/len_test,
        eval_fwavacc/len_test
    )
    print(log)


if __name__ == '__main__':
    main()