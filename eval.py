import os
import torch
from net import RINet_attention_cir_pad,RINet_attention_cons_pad
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
import sys
import time
import argparse

from MAE import *
from database import seq2pc_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fast_eval(seq='00', model_file=None,model_mae_file=None, velo_desc_folder=None, img_desc_folder=None,pair_folder=None, use_l2_dis=False):
    if seq=='08':
        net=RINet_attention_cir_pad()
    else:
        net = RINet_attention_cons_pad()
    net.load(model_file)
    net.to(device=device)
    net.eval()

    mask_ratio = 0.1
    model_mae = MAE_ViT(mask_ratio = mask_ratio).to(device)
    checkpoint = torch.load(model_mae_file)
    model_mae.load_state_dict(checkpoint['state_dict'])
    model_mae.eval() 
    
    #导入数据
    img_desc_folder_0=os.path.join(img_desc_folder,'0')
    img_desc_folder_5=os.path.join(img_desc_folder,'5')
    img_desc_folder_10=os.path.join(img_desc_folder,'10')
    img_desc_folder_15=os.path.join(img_desc_folder,'15')

    velo_desc_folder_0=os.path.join(velo_desc_folder,'0')
    velo_desc_folder_1=os.path.join(velo_desc_folder,'1')
    velo_desc_folder_2=os.path.join(velo_desc_folder,'2')
    velo_desc_folder_3=os.path.join(velo_desc_folder,'3')
    velo_desc_folder_4=os.path.join(velo_desc_folder,'4')
    velo_desc_folder_5=os.path.join(velo_desc_folder,'5')
    velo_desc_folder_6=os.path.join(velo_desc_folder,'6')
    velo_desc_folder_7=os.path.join(velo_desc_folder,'7')

    pair_file=os.path.join(pair_folder,seq+'.txt')

    test_dataset = seq2pc_test(seq=seq,
                            pair_file= pair_file,
                            img_desc_folder_0= img_desc_folder_0,
                            img_desc_folder_5= img_desc_folder_5,
                            img_desc_folder_10= img_desc_folder_10,
                            img_desc_folder_15= img_desc_folder_15,
                            velo_desc_folder_0= velo_desc_folder_0,
                            velo_desc_folder_1= velo_desc_folder_1,
                            velo_desc_folder_2= velo_desc_folder_2,
                            velo_desc_folder_3= velo_desc_folder_3,
                            velo_desc_folder_4= velo_desc_folder_4,
                            velo_desc_folder_5= velo_desc_folder_5,
                            velo_desc_folder_6= velo_desc_folder_6,
                            velo_desc_folder_7= velo_desc_folder_7
                            ) 
    
    batch_size=1024
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    

    #inference
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test", leave=False):

            input=torch.cat((sample_batch["img_descs_0"].unsqueeze(1),sample_batch["img_descs_5"].unsqueeze(1),sample_batch["img_descs_10"].unsqueeze(1),sample_batch["img_descs_15"].unsqueeze(1)),1)
            input=input.to(device) #b*4*12*90

            seq_contour_matrix, mask=model_mae(input)
            seq_contour_matrix=torch.clamp(seq_contour_matrix,min=0.0,max=1.0)

            #保存生成的轮廓矩阵
            pad = (135, 135)  # 在 最后1 维度上左侧补充 135 个 0，右侧补充 135 个 0
            seq_contour_matrix=torch.nn.functional.pad(seq_contour_matrix, pad, mode='constant', value=0) #64, 1, 12, 360

            img_contour_matrix=torch.nn.functional.pad(sample_batch["img_descs_0"].unsqueeze(1), pad, mode='constant', value=0) #64, 1, 12, 360

            out, diff,out_cat = net(seq_contour_matrix.squeeze(1)-img_contour_matrix.squeeze(1).to(device=device),
                                    img_contour_matrix.squeeze(1).to(device=device),
                sample_batch["desc2_0"].to(device=device),
                sample_batch["desc2_1"].to(device=device),
                sample_batch["desc2_2"].to(device=device),
                sample_batch["desc2_3"].to(device=device),
                sample_batch["desc2_4"].to(device=device),
                sample_batch["desc2_5"].to(device=device),
                sample_batch["desc2_6"].to(device=device),
                sample_batch["desc2_7"].to(device=device),
                )
            out = out.cpu()
            outlabel = out
            label = sample_batch['label']
            mask = (label > 0.9906840407) | (label < 0.0012710163)
            label = label[mask]
            label[label < 0.5] = 0
            label[label > 0.5] = 1
            pred.extend(outlabel[mask])
            gt.extend(label)

        pred = np.array(pred, dtype='float32')
        gt = np.array(gt, dtype='float32')
        print('pred',pred)
        print('gt',gt)
        pred = np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        print("F1_split:", testaccur)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='00',
                        help='Sequence to eval. [default: 08]')
    parser.add_argument('--dataset', default="kitti",
                        help="Dataset")
    parser.add_argument('--model', default=None,
                        help='Model file.')
    parser.add_argument('--model_mae', default=None,
                        help='Model file.')

    parser.add_argument('--velo_desc_folder', default='./lidar_desc',
                        help='folder of lidar descriptors. ')
    parser.add_argument('--img_desc_folder', default='./img_desc',
                        help='folder of image descriptors. ')                    

    parser.add_argument('--pair_folder', default='./pairs_fold/neg_100',
                        help='Candidate pairs. ')
    parser.add_argument('--eval_type', default="f1",
                        help='Type of evaluation')
    cfg = parser.parse_args()
    if cfg.dataset == "kitti" and cfg.eval_type == "f1":
        fast_eval(seq=cfg.seq, 
                  model_file=cfg.model,
                  model_mae_file=cfg.model_mae,
                  velo_desc_folder=cfg.velo_desc_folder, 
                  img_desc_folder=cfg.img_desc_folder,
                  pair_folder=cfg.pair_folder)
    else:
        print("Error")
