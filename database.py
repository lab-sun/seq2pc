from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import json
import random  

def load_bin(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    # print('scan',scan)
    # print('scan.shape',scan.shape)
    # os._exit()
    # scan = scan.reshape((4541,12,360))
    return scan


class seq_train(Dataset): 
    '''
    输出img的轮廓矩阵、lidar的轮廓矩阵、以及标签
    '''
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                neg_ratio=1,  
                gt_folder="/workspace/data/gt_kitti/RINet90_forwad_seq", 
                eva_ratio=0.1,
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                velo_desc_folder_0='/workspace/data/kitti_object/desc_kitti_split90/0',
                velo_desc_folder_1='/workspace/data/kitti_object/desc_kitti_split90/1',
                velo_desc_folder_2='/workspace/data/kitti_object/desc_kitti_split90/2',
                velo_desc_folder_3='/workspace/data/kitti_object/desc_kitti_split90/3',
                velo_desc_folder_4='/workspace/data/kitti_object/desc_kitti_split90/4',
                velo_desc_folder_5='/workspace/data/kitti_object/desc_kitti_split90/5',
                velo_desc_folder_6='/workspace/data/kitti_object/desc_kitti_split90/6',
                velo_desc_folder_7='/workspace/data/kitti_object/desc_kitti_split90/7',
                ) -> None:
        super().__init__()
        print(sequs)
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        
        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        self.img_descs_cb = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        for seq in sequs:
            print('train:seq',seq)
            gt_file = os.path.join(gt_folder, seq+'.npz')

            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            # img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')


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
            # img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]

            #截取朝前的90°视场角内的数据 （135-225°），舍弃225那个点
            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            # img_desc_cb=img_desc_cb[:,:,135:225]

            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            # self.img_descs_cb.append(img_desc_cb)

            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            gt = np.load(gt_file)
            pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
            neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)


    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {
                "img_descs_0": self.img_descs_0[int(id_seq)][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[int(id_seq)][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[int(id_seq)][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[int(id_seq)][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[int(id_seq)][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {
                "img_descs_0": self.img_descs_0[i-1][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[i-1][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[i-1][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[i-1][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[i-1][int(pair[0])]/50.,  
                "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50., 
                "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50., 
                "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50., 
                "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0

class seq_eval(Dataset): 
    '''
    输出img的轮廓矩阵、lidar的轮廓矩阵、以及标签
    '''
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                neg_ratio=1,  
                gt_folder="/workspace/data/gt_kitti/RINet90_forwad_seq", 
                eva_ratio=0.1,
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                velo_desc_folder_0='/workspace/data/kitti_object/desc_kitti_split90/0',
                velo_desc_folder_1='/workspace/data/kitti_object/desc_kitti_split90/1',
                velo_desc_folder_2='/workspace/data/kitti_object/desc_kitti_split90/2',
                velo_desc_folder_3='/workspace/data/kitti_object/desc_kitti_split90/3',
                velo_desc_folder_4='/workspace/data/kitti_object/desc_kitti_split90/4',
                velo_desc_folder_5='/workspace/data/kitti_object/desc_kitti_split90/5',
                velo_desc_folder_6='/workspace/data/kitti_object/desc_kitti_split90/6',
                velo_desc_folder_7='/workspace/data/kitti_object/desc_kitti_split90/7',
                ) -> None:
        super().__init__()
        print(sequs)
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        
        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        self.img_descs_cb = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        
        for seq in sequs:
            print('train:seq',seq)
            gt_file = os.path.join(gt_folder, seq+'.npz')

            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            # img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')


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
            # img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]


            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            # img_desc_cb=img_desc_cb[:,:,135:225]



            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            # self.img_descs_cb.append(img_desc_cb)

            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            gt = np.load(gt_file)
            pos = gt['pos'][-int(len(gt['pos'])*eva_ratio):]
            neg = gt['neg'][-int(len(gt['neg'])*eva_ratio):]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {
                "img_descs_0": self.img_descs_0[int(id_seq)][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[int(id_seq)][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[int(id_seq)][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[int(id_seq)][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[int(id_seq)][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {
                "img_descs_0": self.img_descs_0[i-1][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[i-1][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[i-1][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[i-1][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[i-1][int(pair[0])]/50.,    
                "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50., 
                "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50., 
                "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50., 
                "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0

class seq_test(Dataset): 
    '''
    输出img的轮廓矩阵、lidar的轮廓矩阵、以及标签
    '''
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                neg_ratio=1,  
                gt_folder="/workspace/data/gt_kitti/RINet90_forwad_seq", 
                eva_ratio=0.1,
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                velo_desc_folder_0='/workspace/data/kitti_object/desc_kitti_split90/0',
                velo_desc_folder_1='/workspace/data/kitti_object/desc_kitti_split90/1',
                velo_desc_folder_2='/workspace/data/kitti_object/desc_kitti_split90/2',
                velo_desc_folder_3='/workspace/data/kitti_object/desc_kitti_split90/3',
                velo_desc_folder_4='/workspace/data/kitti_object/desc_kitti_split90/4',
                velo_desc_folder_5='/workspace/data/kitti_object/desc_kitti_split90/5',
                velo_desc_folder_6='/workspace/data/kitti_object/desc_kitti_split90/6',
                velo_desc_folder_7='/workspace/data/kitti_object/desc_kitti_split90/7',
                ) -> None:
        super().__init__()
        print(sequs)
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0

        
        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        # self.img_descs_cb = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        
        for seq in sequs:
            print('train:seq',seq)
            gt_file = os.path.join(gt_folder, seq+'.npz')

            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            # img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')


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
            # img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]


            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            # img_desc_cb=img_desc_cb[:,:,135:225]


            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            # self.img_descs_cb.append(img_desc_cb)

            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            gt = np.load(gt_file)
            pos = gt['pos'][:]
            neg = gt['neg'][:]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {
                "img_descs_0": self.img_descs_0[int(id_seq)][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[int(id_seq)][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[int(id_seq)][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[int(id_seq)][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[int(id_seq)][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {
                "img_descs_0": self.img_descs_0[i-1][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[i-1][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[i-1][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[i-1][int(pair[0])]/50.,  
                # "img_descs_cb": self.img_descs_cb[i-1][int(pair[0])]/50.,   
                "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50., 
                "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50., 
                "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50., 
                "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class seq2pc_test(Dataset): 
    '''
    输出seq的轮廓矩阵、lidar的轮廓矩阵、以及标签
    '''
    def __init__(self, seq='00', 
                pair_file='/workspace/data/RINet/pairs_kitti/neg_100/05.txt', 
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                velo_desc_folder_0='/workspace/data/kitti_object/desc_kitti_split90/0',
                velo_desc_folder_1='/workspace/data/kitti_object/desc_kitti_split90/1',
                velo_desc_folder_2='/workspace/data/kitti_object/desc_kitti_split90/2',
                velo_desc_folder_3='/workspace/data/kitti_object/desc_kitti_split90/3',
                velo_desc_folder_4='/workspace/data/kitti_object/desc_kitti_split90/4',
                velo_desc_folder_5='/workspace/data/kitti_object/desc_kitti_split90/5',
                velo_desc_folder_6='/workspace/data/kitti_object/desc_kitti_split90/6',
                velo_desc_folder_7='/workspace/data/kitti_object/desc_kitti_split90/7',
                ) -> None:
        super().__init__()
        print('Seq:',seq)
        
        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
        img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
        img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
        img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')

        velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
        velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
        velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
        velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
        velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
        velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
        velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
        velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')

        img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)
        img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
        img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
        img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)

        velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]
        velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]

        img_desc_0=img_desc_0[:,:,135:225]
        img_desc_5=img_desc_5[:,:,135:225]
        img_desc_10=img_desc_10[:,:,135:225]
        img_desc_15=img_desc_15[:,:,135:225]

        self.img_descs_0.append(img_desc_0)
        self.img_descs_5.append(img_desc_5)
        self.img_descs_10.append(img_desc_10)
        self.img_descs_15.append(img_desc_15)

        self.velo_descs_0.append(velo_desc_0)
        self.velo_descs_1.append(velo_desc_1)
        self.velo_descs_2.append(velo_desc_2)
        self.velo_descs_3.append(velo_desc_3)
        self.velo_descs_4.append(velo_desc_4)
        self.velo_descs_5.append(velo_desc_5)
        self.velo_descs_6.append(velo_desc_6)
        self.velo_descs_7.append(velo_desc_7)

        

        self.pairs = np.genfromtxt(pair_file, dtype='int32').reshape(-1, 3)
        print('len(self.pairs)',len(self.pairs))
        self.pairs=self.pairs[self.pairs[:,0]<len(img_desc_0)]
        self.pairs=self.pairs[self.pairs[:,1]<len(img_desc_0)]
        print('len(self.pairs)_processed',len(self.pairs))

        self.num=len(self.pairs)
        print('self.num',self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        out = {"img_descs_0": self.img_descs_0[0][int(pair[0])]/50., 
                "img_descs_5": self.img_descs_5[0][int(pair[0])]/50., 
                "img_descs_10": self.img_descs_10[0][int(pair[0])]/50.,  
                "img_descs_15": self.img_descs_15[0][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[0][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[0][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[0][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[0][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[0][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[0][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[0][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[0][int(pair[1])]/50.,
                'label': pair[2]}
        return out


    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class seq2pc_test_recall(Dataset): 
    '''
    输出seq的轮廓矩阵、lidar的轮廓矩阵、以及标签
    '''
    def __init__(self, seq='00', 
                v=0, 
                img_desc_0=None,
                img_desc_5=None,
                img_desc_10=None,
                img_desc_15=None,
                img_desc_cb=None,
                velo_desc_0=None,
                velo_desc_1=None,
                velo_desc_2=None,
                velo_desc_3=None,
                velo_desc_4=None,
                velo_desc_5=None,
                velo_desc_6=None,
                velo_desc_7=None,
                ) -> None:
        super().__init__()
        print(seq)
        # self.gt_pos = []
        # self.gt_neg = []
        # self.pos_nums = [0]
        # self.neg_num = 0
        # self.pos_num = 0
        
        # self.img_descs_0 = []
        # self.img_descs_5 = []
        # self.img_descs_10 = []
        # self.img_descs_15 = []
        # self.img_descs_cb = []

        # self.velo_descs_0 = []
        # self.velo_descs_1 = []
        # self.velo_descs_2 = []
        # self.velo_descs_3 = []
        # self.velo_descs_4 = []
        # self.velo_descs_5 = []
        # self.velo_descs_6 = []
        # self.velo_descs_7 = []


        # img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
        # img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
        # img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
        # img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
        # img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')

        # velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
        # velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
        # velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
        # velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
        # velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
        # velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
        # velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
        # velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')

        # img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)#img_desc_0,img_desc_5,img_desc_10,img_desc_15的长度是一样的
        # img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
        # img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
        # img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)
        # img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

        # velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
        # velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)

        self.velo_descs_0=velo_desc_0[0:v-50,:,:]
        self.velo_descs_1=velo_desc_1[0:v-50,:,:]
        self.velo_descs_2=velo_desc_2[0:v-50,:,:]
        self.velo_descs_3=velo_desc_3[0:v-50,:,:]
        self.velo_descs_4=velo_desc_4[0:v-50,:,:]
        self.velo_descs_5=velo_desc_5[0:v-50,:,:]
        self.velo_descs_6=velo_desc_6[0:v-50,:,:]
        self.velo_descs_7=velo_desc_7[0:v-50,:,:]
        

        #截取朝前的90°视场角内的数据 （135-225°），舍弃225那个点
        self.img_descs_0=img_desc_0[v,:,135:225]
        self.img_descs_5=img_desc_5[v,:,135:225]
        self.img_descs_10=img_desc_10[v,:,135:225]
        self.img_descs_15=img_desc_15[v,:,135:225]
        self.img_descs_cb=img_desc_cb[v,:,135:225]

        # img_desc_0=np.expand_dims(img_desc_0,axis=0)
        # img_desc_5=np.expand_dims(img_desc_5,axis=0)
        # img_desc_10=np.expand_dims(img_desc_10,axis=0)
        # img_desc_15=np.expand_dims(img_desc_15,axis=0)
        # img_desc_cb=np.expand_dims(img_desc_cb,axis=0)

        # self.img_descs_0.append(img_desc_0)
        # self.img_descs_5.append(img_desc_5)
        # self.img_descs_10.append(img_desc_10)
        # self.img_descs_15.append(img_desc_15)
        # self.img_descs_cb.append(img_desc_cb)

        # self.velo_descs_0.append(velo_desc_0)
        # self.velo_descs_1.append(velo_desc_1)
        # self.velo_descs_2.append(velo_desc_2)
        # self.velo_descs_3.append(velo_desc_3)
        # self.velo_descs_4.append(velo_desc_4)
        # self.velo_descs_5.append(velo_desc_5)
        # self.velo_descs_6.append(velo_desc_6)
        # self.velo_descs_7.append(velo_desc_7)

        

        # self.pairs = np.genfromtxt(pair_file, dtype='int32').reshape(-1, 3)
        # print('len(self.pairs)',len(self.pairs))
        # self.pairs=self.pairs[self.pairs[:,0]<len(img_desc_0)]
        # self.pairs=self.pairs[self.pairs[:,1]<len(img_desc_0)]
        # print('len(self.pairs)_processed',len(self.pairs))

        self.num=v-50
        print('self.num',self.num)
        # os._exit()
        # print('img_descs_0.shape',self.img_descs_0[0].shape)
        # print('velo_descs_0.shape',self.velo_descs_0[0].shape)
        # print('type(self.img_descs_0[0])',type(self.img_descs_0[0]))
        # print('type(self.velo_descs_0[0])',type(self.velo_descs_0[0]))
        # os._exit()

            # gt = np.load(gt_file)
            # pos = gt['pos'][:]
            # neg = gt['neg'][:]
            # self.gt_pos.append(pos)
            # self.gt_neg.append(neg)
            # self.pos_num += len(self.gt_pos[-1])
            # self.pos_nums.append(self.pos_num)
        # self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        # return self.pos_num+self.neg_num
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pair = self.pairs[idx]
        # out = {"img_descs_0": self.img_descs_0[0][0]/50., 
        #         "img_descs_5": self.img_descs_5[0][0]/50., 
        #         "img_descs_10": self.img_descs_10[0][0]/50.,  
        #         "img_descs_15": self.img_descs_15[0][0]/50.,  
        #         "img_descs_cb": self.img_descs_cb[0][0]/50., 
        #         "desc2_0": self.velo_descs_0[0][idx]/50.,
        #         "desc2_1": self.velo_descs_1[0][idx]/50.,
        #         "desc2_2": self.velo_descs_2[0][idx]/50.,
        #         "desc2_3": self.velo_descs_3[0][idx]/50.,
        #         "desc2_4": self.velo_descs_4[0][idx]/50.,
        #         "desc2_5": self.velo_descs_5[0][idx]/50.,
        #         "desc2_6": self.velo_descs_6[0][idx]/50.,
        #         "desc2_7": self.velo_descs_7[0][idx]/50.,}
        out = {"img_descs_0": self.img_descs_0/50., 
                "img_descs_5": self.img_descs_5/50., 
                "img_descs_10": self.img_descs_10/50.,  
                "img_descs_15": self.img_descs_15/50.,  
                "img_descs_cb": self.img_descs_cb/50., 
                "desc2_0": self.velo_descs_0[idx]/50.,
                "desc2_1": self.velo_descs_1[idx]/50.,
                "desc2_2": self.velo_descs_2[idx]/50.,
                "desc2_3": self.velo_descs_3[idx]/50.,
                "desc2_4": self.velo_descs_4[idx]/50.,
                "desc2_5": self.velo_descs_5[idx]/50.,
                "desc2_6": self.velo_descs_6[idx]/50.,
                "desc2_7": self.velo_descs_7[idx]/50.,}
        return out





class contour_fusion_train(Dataset): 
    '''
    在主函数中, 从sequs中remove当前测试的seq
    i.e., sequs=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] when tesing on seq '00'

    output:
    LiDAR的描述子,6个分视角的轮廓
    Seq的描述子,4个来自img的轮廓
    当前帧到距离其15m的那一帧的坐标变换
    每一帧的位姿
    '''
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], eva_ratio=0.1,
                velo_desc_folder='/workspace/semcorr/data/kitti_object/desc_kitti_split90/0',
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                neg_ratio=1,
                gt_folder="./data/gt_kitti",
                velo_desc_folder_0=None,velo_desc_folder_1=None,velo_desc_folder_2=None,velo_desc_folder_3=None,
                velo_desc_folder_4=None,velo_desc_folder_5=None,velo_desc_folder_6=None,velo_desc_folder_7=None,
                ) -> None:
        super().__init__()
        print(sequs)
        # self.descs = []
        # self.gt_pos = []
        # self.gt_neg = []
        # self.pos_nums = [0]
        # self.neg_num = 0
        # self.pos_num = 0
        self.num = 0
        self.nums = [0]

        self.velo_descs = []
        # self.img_descs = []

        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        self.img_descs_cb = []

        
        for seq in sequs:
            print('train:seq',seq)
            # desc_file = os.path.join(desc_folder, seq+'.npy')
            # gt_file = os.path.join(gt_folder, seq+'.npz')
            velo_desc_file=os.path.join(velo_desc_folder, seq+'.bin')#velo的数目和img_desc的不一样，因为后者是基于seq生成的，前者是基于img生成的
            # img_desc_file=os.path.join(img_desc_folder, seq+'.bin')


            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')

            
            # img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

            img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)#img_desc_0,img_desc_5,img_desc_10,img_desc_15的长度是一样的
            img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
            img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)
            img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]

            
            # self.img_descs.append(img_desc)

            #截取朝前的90°视场角内的数据 （135-225°），舍弃225那个点
            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            img_desc_cb=img_desc_cb[:,:,135:225]
            velo_desc=velo_desc[:,:,135:225]

            # print('img_desc_0.shape',img_desc_0.shape)

            #选取训练的数据量，前80%。后20%用来验证
            img_desc_0 = img_desc_0[:-int(len(img_desc_0)*eva_ratio)]
            img_desc_5 = img_desc_5[:-int(len(img_desc_5)*eva_ratio)]
            img_desc_10 = img_desc_10[:-int(len(img_desc_10)*eva_ratio)]
            img_desc_15 = img_desc_15[:-int(len(img_desc_15)*eva_ratio)]
            img_desc_cb = img_desc_cb[:-int(len(img_desc_cb)*eva_ratio)]
            velo_desc = velo_desc[:-int(len(velo_desc)*eva_ratio)]

            #多个seq，append到一块
            self.velo_descs.append(velo_desc)
            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            self.img_descs_cb.append(img_desc_cb)

            self.num += len(img_desc_0)
            self.nums.append(self.num)
            # print('len(img_desc_0)',len(img_desc_0))
            # print('self.num',self.num)
            print('self.nums',self.nums)

            # print('type(velo_desc)',type(velo_desc))
            print('velo_desc.shape',velo_desc.shape)
            print('img_desc_0.shape',img_desc_0.shape)
            print('len(velo_desc)',len(velo_desc))
            print('len(img_desc_0)',len(img_desc_0))
            print('len(img_desc_5)',len(img_desc_5))
            print('len(img_desc_10)',len(img_desc_10))
            print('len(img_desc_cb)',len(img_desc_cb))
            # os._exit()

        #     gt = np.load(gt_file)
        #     # print('gt[pos]',len(gt['pos']))
        #     pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
        #     # pos = gt['pos'][:]
        #     # print('pos',len(pos))
        #     neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
        #     # neg = gt['neg'][:]
        #     # print('gt[neg]',len(gt['neg']))
        #     self.gt_pos.append(pos)
        #     self.gt_neg.append(neg)
        #     self.pos_num += len(self.gt_pos[-1])
        #     # print('pos_num',len(self.gt_pos[-1]))
        #     self.pos_nums.append(self.pos_num)
        # self.neg_num = int(neg_ratio*self.pos_num)
        # print('self.nums',self.nums)
        # os._exit()


        # print('train:self.pos_num+self.neg_num',self.pos_num+self.neg_num)

    def __len__(self):
        # return self.pos_num+self.neg_num
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('idx is a tensor')
            idx = idx.tolist()

        for i in range(1, len(self.nums)):
            if self.nums[i] > idx:
                id=idx-self.nums[i-1]
                id_seq=i-1
                # print('id_seq,id',id_seq,id)
                out = {"GT": self.velo_descs[id_seq][id]/50., 
                "img_descs_0": self.img_descs_0[id_seq][id]/50., 
                "img_descs_5": self.img_descs_5[id_seq][id]/50., 
                "img_descs_10": self.img_descs_10[id_seq][id]/50.,  
                "img_descs_15": self.img_descs_15[id_seq][id]/50.,  
                "img_descs_cb": self.img_descs_cb[id_seq][id]/50.,  
                }

                # if random.randint(0, 1) > 0:
                #     self.rand_occ(out["desc1"])
                #     self.rand_occ(out["desc2"])
                #     self.rand_occ(out["desc2_0"])
                #     self.rand_occ(out["desc2_1"])
                #     self.rand_occ(out["desc2_2"])
                #     self.rand_occ(out["desc2_3"])
                #     self.rand_occ(out["desc2_4"])
                #     self.rand_occ(out["desc2_5"])
                #     self.rand_occ(out["desc2_6"])
                #     self.rand_occ(out["desc2_7"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0

class contour_fusion_eval(Dataset): 
    '''
    在主函数中, 从sequs中remove当前测试的seq
    i.e., sequs=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] when tesing on seq '00'
    '''
    # def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], eva_ratio=0.2,
    #             velo_desc_folder='/hdd8tb/semcorr/data/kitti_object/desc_kitti_split90/0',
    #             img_desc_folder_0='/hdd8tb/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
    #             img_desc_folder_5='/hdd8tb/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
    #             img_desc_folder_10='/hdd8tb/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
    #             img_desc_folder_15='/hdd8tb/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
    #             img_desc_folder_cb='/hdd8tb/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
    #             ) -> None:
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], eva_ratio=0.2,
                velo_desc_folder='/workspace/semcorr/data/kitti_object/desc_kitti_split90/0',
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                ) -> None:
        super().__init__()
        print(sequs)
        # self.descs = []
        # self.gt_pos = []
        # self.gt_neg = []
        # self.pos_nums = [0]
        # self.neg_num = 0
        # self.pos_num = 0
        self.num = 0
        self.nums = [0]

        self.velo_descs = []
        # self.img_descs = []

        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        self.img_descs_cb = []

        
        for seq in sequs:
            print('train:seq',seq)
            # desc_file = os.path.join(desc_folder, seq+'.npy')
            # gt_file = os.path.join(gt_folder, seq+'.npz')
            velo_desc_file=os.path.join(velo_desc_folder, seq+'.bin')#velo的数目和img_desc的不一样，因为后者是基于seq生成的，前者是基于img生成的
            # img_desc_file=os.path.join(img_desc_folder, seq+'.bin')


            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')

            
            # img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

            img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)#img_desc_0,img_desc_5,img_desc_10,img_desc_15的长度是一样的
            img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
            img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)
            img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]

            
            # self.img_descs.append(img_desc)

            #截取朝前的90°视场角内的数据 （135-225°），舍弃225那个点
            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            img_desc_cb=img_desc_cb[:,:,135:225]
            velo_desc=velo_desc[:,:,135:225]

            # print('img_desc_0.shape',img_desc_0.shape)

            #选取训练的数据量，前80%。后20%用来验证
            img_desc_0 = img_desc_0[-int(len(img_desc_0)*eva_ratio):]
            img_desc_5 = img_desc_5[-int(len(img_desc_5)*eva_ratio):]
            img_desc_10 = img_desc_10[-int(len(img_desc_10)*eva_ratio):]
            img_desc_15 = img_desc_15[-int(len(img_desc_15)*eva_ratio):]
            img_desc_cb = img_desc_cb[-int(len(img_desc_cb)*eva_ratio):]
            velo_desc = velo_desc[-int(len(velo_desc)*eva_ratio):]

            #多个seq，append到一块
            self.velo_descs.append(velo_desc)
            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            self.img_descs_cb.append(img_desc_cb)

            self.num += len(img_desc_0)
            self.nums.append(self.num)
            # print('len(img_desc_0)',len(img_desc_0))
            # print('self.num',self.num)
            print('self.nums',self.nums)

            # print('type(velo_desc)',type(velo_desc))
            print('velo_desc.shape',velo_desc.shape)
            print('img_desc_0.shape',img_desc_0.shape)
            print('len(velo_desc)',len(velo_desc))
            print('len(img_desc_0)',len(img_desc_0))
            print('len(img_desc_5)',len(img_desc_5))
            print('len(img_desc_10)',len(img_desc_10))
            print('len(img_desc_cb)',len(img_desc_cb))
            # os._exit()

        #     gt = np.load(gt_file)
        #     # print('gt[pos]',len(gt['pos']))
        #     pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
        #     # pos = gt['pos'][:]
        #     # print('pos',len(pos))
        #     neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
        #     # neg = gt['neg'][:]
        #     # print('gt[neg]',len(gt['neg']))
        #     self.gt_pos.append(pos)
        #     self.gt_neg.append(neg)
        #     self.pos_num += len(self.gt_pos[-1])
        #     # print('pos_num',len(self.gt_pos[-1]))
        #     self.pos_nums.append(self.pos_num)
        # self.neg_num = int(neg_ratio*self.pos_num)
        # print('self.nums',self.nums)
        # os._exit()


        # print('train:self.pos_num+self.neg_num',self.pos_num+self.neg_num)

    def __len__(self):
        # return self.pos_num+self.neg_num
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('idx is a tensor')
            idx = idx.tolist()

        for i in range(1, len(self.nums)):
            if self.nums[i] > idx:
                id=idx-self.nums[i-1]
                id_seq=i-1
                # print('id_seq,id',id_seq,id)
                out = {"GT": self.velo_descs[id_seq][id]/50., 
                "img_descs_0": self.img_descs_0[id_seq][id]/50., 
                "img_descs_5": self.img_descs_5[id_seq][id]/50., 
                "img_descs_10": self.img_descs_10[id_seq][id]/50.,  
                "img_descs_15": self.img_descs_15[id_seq][id]/50.,  
                "img_descs_cb": self.img_descs_cb[id_seq][id]/50.,  
                }

                # if random.randint(0, 1) > 0:
                #     self.rand_occ(out["desc1"])
                #     self.rand_occ(out["desc2"])
                #     self.rand_occ(out["desc2_0"])
                #     self.rand_occ(out["desc2_1"])
                #     self.rand_occ(out["desc2_2"])
                #     self.rand_occ(out["desc2_3"])
                #     self.rand_occ(out["desc2_4"])
                #     self.rand_occ(out["desc2_5"])
                #     self.rand_occ(out["desc2_6"])
                #     self.rand_occ(out["desc2_7"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0

class contour_fusion_test(Dataset): 
    '''
    和train/eval不同,在test中,1)seques选择当前序列 2)没有eval_ratio
    '''
    def __init__(self, sequs=['00'],
                velo_desc_folder='/workspace/semcorr/data/kitti_object/desc_kitti_split90/0',
                img_desc_folder_0='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/0',
                img_desc_folder_5='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/5',
                img_desc_folder_10='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/10',
                img_desc_folder_15='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/15',
                img_desc_folder_cb='/workspace/semcorr/data/kitti_object/desc_kitti_image_seq_based_on_dis/combine',
                ) -> None:
        super().__init__()
        print(sequs)
        # self.descs = []
        # self.gt_pos = []
        # self.gt_neg = []
        # self.pos_nums = [0]
        # self.neg_num = 0
        # self.pos_num = 0
        self.num = 0
        self.nums = [0]

        self.velo_descs = []
        # self.img_descs = []

        self.img_descs_0 = []
        self.img_descs_5 = []
        self.img_descs_10 = []
        self.img_descs_15 = []
        self.img_descs_cb = []

        
        for seq in sequs:
            print('train:seq',seq)
            # desc_file = os.path.join(desc_folder, seq+'.npy')
            # gt_file = os.path.join(gt_folder, seq+'.npz')
            velo_desc_file=os.path.join(velo_desc_folder, seq+'.bin')#velo的数目和img_desc的不一样，因为后者是基于seq生成的，前者是基于img生成的
            # img_desc_file=os.path.join(img_desc_folder, seq+'.bin')


            img_desc_file_0=os.path.join(img_desc_folder_0, seq+'.bin')
            img_desc_file_5=os.path.join(img_desc_folder_5, seq+'.bin')
            img_desc_file_10=os.path.join(img_desc_folder_10, seq+'.bin')
            img_desc_file_15=os.path.join(img_desc_folder_15, seq+'.bin')
            img_desc_file_cb=os.path.join(img_desc_folder_cb, seq+'.bin')

            
            # img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

            img_desc_0=np.fromfile(img_desc_file_0, dtype=np.float32).reshape(-1,12,360)#img_desc_0,img_desc_5,img_desc_10,img_desc_15的长度是一样的
            img_desc_5=np.fromfile(img_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            img_desc_10=np.fromfile(img_desc_file_10, dtype=np.float32).reshape(-1,12,360)
            img_desc_15=np.fromfile(img_desc_file_15, dtype=np.float32).reshape(-1,12,360)
            img_desc_cb=np.fromfile(img_desc_file_cb, dtype=np.float32).reshape(-1,12,360)

            velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)[0:len(img_desc_0),:,:]

            
            # self.img_descs.append(img_desc)

            #截取朝前的90°视场角内的数据 （135-225°），舍弃225那个点
            img_desc_0=img_desc_0[:,:,135:225]
            img_desc_5=img_desc_5[:,:,135:225]
            img_desc_10=img_desc_10[:,:,135:225]
            img_desc_15=img_desc_15[:,:,135:225]
            img_desc_cb=img_desc_cb[:,:,135:225]
            velo_desc=velo_desc[:,:,135:225]

            # print('img_desc_0.shape',img_desc_0.shape)

            #选取训练的数据量，前80%。后20%用来验证
            img_desc_0 = img_desc_0[-int(len(img_desc_0)):]
            img_desc_5 = img_desc_5[-int(len(img_desc_5)):]
            img_desc_10 = img_desc_10[-int(len(img_desc_10)):]
            img_desc_15 = img_desc_15[-int(len(img_desc_15)):]
            img_desc_cb = img_desc_cb[-int(len(img_desc_cb)):]
            velo_desc = velo_desc[-int(len(velo_desc)):]

            #多个seq，append到一块
            self.velo_descs.append(velo_desc)
            self.img_descs_0.append(img_desc_0)
            self.img_descs_5.append(img_desc_5)
            self.img_descs_10.append(img_desc_10)
            self.img_descs_15.append(img_desc_15)
            self.img_descs_cb.append(img_desc_cb)

            self.num += len(img_desc_0)
            self.nums.append(self.num)
            # print('len(img_desc_0)',len(img_desc_0))
            # print('self.num',self.num)
            print('self.nums',self.nums)

            # print('type(velo_desc)',type(velo_desc))
            print('velo_desc.shape',velo_desc.shape)
            print('img_desc_0.shape',img_desc_0.shape)
            print('len(velo_desc)',len(velo_desc))
            print('len(img_desc_0)',len(img_desc_0))
            print('len(img_desc_5)',len(img_desc_5))
            print('len(img_desc_10)',len(img_desc_10))
            print('len(img_desc_cb)',len(img_desc_cb))
            # os._exit()

        #     gt = np.load(gt_file)
        #     # print('gt[pos]',len(gt['pos']))
        #     pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
        #     # pos = gt['pos'][:]
        #     # print('pos',len(pos))
        #     neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
        #     # neg = gt['neg'][:]
        #     # print('gt[neg]',len(gt['neg']))
        #     self.gt_pos.append(pos)
        #     self.gt_neg.append(neg)
        #     self.pos_num += len(self.gt_pos[-1])
        #     # print('pos_num',len(self.gt_pos[-1]))
        #     self.pos_nums.append(self.pos_num)
        # self.neg_num = int(neg_ratio*self.pos_num)
        # print('self.nums',self.nums)
        # os._exit()


        # print('train:self.pos_num+self.neg_num',self.pos_num+self.neg_num)

    def __len__(self):
        # return self.pos_num+self.neg_num
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('idx is a tensor')
            idx = idx.tolist()

        for i in range(1, len(self.nums)):
            if self.nums[i] > idx:
                id=idx-self.nums[i-1]
                id_seq=i-1
                # print('id_seq,id',id_seq,id)
                out = {"GT": self.velo_descs[id_seq][id]/50., 
                "img_descs_0": self.img_descs_0[id_seq][id]/50., 
                "img_descs_5": self.img_descs_5[id_seq][id]/50., 
                "img_descs_10": self.img_descs_10[id_seq][id]/50.,  
                "img_descs_15": self.img_descs_15[id_seq][id]/50.,  
                "img_descs_cb": self.img_descs_cb[id_seq][id]/50.,  
                }

                # if random.randint(0, 1) > 0:
                #     self.rand_occ(out["desc1"])
                #     self.rand_occ(out["desc2"])
                #     self.rand_occ(out["desc2_0"])
                #     self.rand_occ(out["desc2_1"])
                #     self.rand_occ(out["desc2_2"])
                #     self.rand_occ(out["desc2_3"])
                #     self.rand_occ(out["desc2_4"])
                #     self.rand_occ(out["desc2_5"])
                #     self.rand_occ(out["desc2_6"])
                #     self.rand_occ(out["desc2_7"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0






if __name__ == '__main__':
    database = SigmoidDataset_train(
        ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 2)
    print(len(database))
    for i in range(0, len(database)):
        idx = random.randint(0, len(database)-1)
        d = database[idx]
        print(i, d['label'])
        plt.subplot(2, 1, 1)
        plt.imshow(d['desc1'])
        plt.subplot(2, 1, 2)
        plt.imshow(d['desc2'])
        plt.show()
