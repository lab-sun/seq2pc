import os
import torch
from net import RINet_attention_cir_pad, RINet_attention_cons_pad
from database import seq2pc_test_recall
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
import sys
import time
import argparse

from MAE import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recall(seq='00', model_file=None,model_mae_file=None,
             velo_desc_folder=None, img_desc_folder=None,
            pose_file="./data/pose_kitti/00.txt"):
            
    img_desc_folder_0=os.path.join(img_desc_folder,'0')
    img_desc_folder_5=os.path.join(img_desc_folder,'5')
    img_desc_folder_10=os.path.join(img_desc_folder,'10')
    img_desc_folder_15=os.path.join(img_desc_folder,'15')
    img_desc_folder_cb=os.path.join(img_desc_folder,'combine')

    velo_desc_folder_0=os.path.join(velo_desc_folder,'0')
    velo_desc_folder_1=os.path.join(velo_desc_folder,'1')
    velo_desc_folder_2=os.path.join(velo_desc_folder,'2')
    velo_desc_folder_3=os.path.join(velo_desc_folder,'3')
    velo_desc_folder_4=os.path.join(velo_desc_folder,'4')
    velo_desc_folder_5=os.path.join(velo_desc_folder,'5')
    velo_desc_folder_6=os.path.join(velo_desc_folder,'6')
    velo_desc_folder_7=os.path.join(velo_desc_folder,'7')


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


    img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
    img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
    img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
    img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
    img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')

    velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
    velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
    velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
    velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
    velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
    velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
    velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
    velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')

    img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)#img_desc_0,img_desc_5,img_desc_10,img_desc_15的长度是一样的
    img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
    img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
    img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)
    img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

    velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
    velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
    velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
    velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
    velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
    velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
    velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
    velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)
    
    #转化为tensor 上传到GPU中，喂给dataset再进一步喂给dataloader
    img_desc_0=torch.from_numpy(img_desc_0).to(device)
    img_desc_5=torch.from_numpy(img_desc_5).to(device)
    img_desc_10=torch.from_numpy(img_desc_10).to(device)
    img_desc_15=torch.from_numpy(img_desc_15).to(device)
    img_desc_cb=torch.from_numpy(img_desc_cb).to(device)
    velo_desc_0=torch.from_numpy(velo_desc_0).to(device)
    velo_desc_1=torch.from_numpy(velo_desc_1).to(device)
    velo_desc_2=torch.from_numpy(velo_desc_2).to(device)
    velo_desc_3=torch.from_numpy(velo_desc_3).to(device)
    velo_desc_4=torch.from_numpy(velo_desc_4).to(device)
    velo_desc_5=torch.from_numpy(velo_desc_5).to(device)
    velo_desc_6=torch.from_numpy(velo_desc_6).to(device)
    velo_desc_7=torch.from_numpy(velo_desc_7).to(device)
    

    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    id_pos = np.argwhere(dis <= 5)
    
    id_pos = id_pos[id_pos[:, 0]-id_pos[:, 1] > 50]
    pos_dict = {}
    

    #删除一些假回环，由车辆停滞时间过长导致的
    for v in id_pos: #id_pos的最小值是0
        for ii in range(v[0]-v[1]):
            if dis[v[0],v[0]-ii-1]>5:
                #去掉一些超过选择范围的地点
                if v[0]>=len(img_desc_0):
                    continue
                if v[0] in pos_dict.keys():
                    pos_dict[v[0]].append(v[1])
                else:
                    pos_dict[v[0]] = [v[1]]


    out_save = []
    recall = np.array([0.]*25)
    for v in tqdm(pos_dict.keys()):
        print('v',v)
        test_dataset = seq2pc_test_recall(seq=seq,
                            v= v,
                            img_desc_0= img_desc_0,
                            img_desc_5= img_desc_5,
                            img_desc_10= img_desc_10,
                            img_desc_15= img_desc_15,
                            img_desc_cb= img_desc_cb,
                            velo_desc_0= velo_desc_0,
                            velo_desc_1= velo_desc_1,
                            velo_desc_2= velo_desc_2,
                            velo_desc_3= velo_desc_3,
                            velo_desc_4= velo_desc_4,
                            velo_desc_5= velo_desc_5,
                            velo_desc_6= velo_desc_6,
                            velo_desc_7= velo_desc_7,
                            )
        batch_size=1024
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
        with torch.no_grad():
            out_list=[]
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
                out_list.append(out)
            out_list=torch.cat(out_list,dim=0)
            out_list = out_list.cpu().numpy()
            ids = np.argsort(-out_list)
            o = [v]
            o += ids[:25].tolist()
            
            for i in range(25):
                if ids[i] in pos_dict[v]:
                    o+=[True]
                else:
                    o+=[False]
            for i in range(25):
                if ids[i] in pos_dict[v]:
                    recall[i:] += 1
                    break

            out_save.append(o)
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result', seq+'_recall_retrieval.txt'), out_save, fmt='%d')
    recall /= len(pos_dict.keys())
    print(recall)
    np.savetxt(os.path.join('result', seq+'_recall_scores.txt'), recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='00',
                        help='Sequence to eval. ')
    parser.add_argument('--dataset', default="kitti",
                        help="Dataset ")
    parser.add_argument('--model', default="./checkpoints/00.ckpt",
                        help='Model file. ')
    parser.add_argument('--model_mae', default="./checkpoints/00_mae.ckpt",
                        help='Model file. ')

    parser.add_argument('--velo_desc_folder', default='./lidar_desc',
                        help='folder of lidar descriptors. ')
    parser.add_argument('--img_desc_folder', default='./img_desc',
                        help='folder of image descriptors. ')  

    parser.add_argument('--pose_file', default="./pose_kitti/00.txt",
                        help='Pose file (eval_type=recall). ')
    parser.add_argument('--eval_type', default="recall",
                        help='Type of evaluation (f1 or recall). [default: f1]')
    cfg = parser.parse_args()
    if cfg.dataset == "kitti" and cfg.eval_type == "recall":
        recall(seq=cfg.seq, model_file=cfg.model, model_mae_file=cfg.model_mae,
                velo_desc_folder=cfg.velo_desc_folder,
                img_desc_folder=cfg.img_desc_folder,
                pose_file=cfg.pose_file)
    else:
        print("Error")


