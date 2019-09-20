from data import CrackDataSetTest
from net import resnet18 as Net
import cv2
import torch
import numpy as np
from train import create_train_val_test_list

if __name__ == '__main__':
    img_dir=r'/raid/lcq/dataset/crack/imgs'
    label_dir=r'/raid/lcq/dataset/crack/labels'
    weight_path=r'./weights/result.plt'
    GPU_ID=0
    #22,42,52
    img_index=52

    GPU=torch.device('cuda:{}'.format(GPU_ID))
    _,_,test_list=create_train_val_test_list(img_dir)
    test_crackDataSet = CrackDataSetTest(img_dir=img_dir, imgs=test_list, label_dir=label_dir)
    test_img,test_label,img_path=test_crackDataSet[img_index]
    save_img=(test_img*255).permute(1,2,0).byte().numpy()
    test_img=test_img.unsqueeze(0)
    model = Net(pretrained=True, num_classes=2)
    model.load_state_dict(torch.load(weight_path))
    model = model.to(GPU)
    test_img=test_img.to(GPU)
    feat = model(test_img)
    pre = feat.argmax(1)
    pre=pre.squeeze(0)
    pre=pre.cpu().byte().numpy()*255
    test_label=test_label.cpu().byte().numpy()*255

    print(img_path)
    cv2.imwrite('./src.png',save_img)
    cv2.imwrite('./gt.png',test_label)
    cv2.imwrite('pre.png',pre)

    # cv2.imshow('label', test_label)
    # cv2.imshow('pre',pre)
    # cv2.waitKey()

