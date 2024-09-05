import os
import string
import torch
from net import RINet_attention_cir_pad, RINet_attention_cons_pad
from database import seq_train,seq_eval,seq_test
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import argparse
from torch.utils.tensorboard.writer import SummaryWriter

from MAE import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def train(cfg):
    writer = SummaryWriter()
    if cfg.seq=='08':
        net=RINet_attention_cir_pad()
    else:
        net = RINet_attention_cons_pad()
    net.to(device=device)
    sequs = cfg.all_seqs
    sequs.remove(cfg.seq)

    
    train_dataset = seq_train(sequs=sequs, 
                            neg_ratio=cfg.neg_ratio,
                            gt_folder=cfg.gt_folder,
                            eva_ratio=cfg.eval_ratio, 
                            img_desc_folder_0=cfg.img_desc_folder_0,
                            img_desc_folder_5=cfg.img_desc_folder_5,
                            img_desc_folder_10=cfg.img_desc_folder_10,
                            img_desc_folder_15=cfg.img_desc_folder_15,
                            img_desc_folder_cb=cfg.img_desc_folder_cb,
                            velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
                            velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7
                                         )
    eval_dataset = seq_eval(sequs=sequs, 
                            neg_ratio=cfg.neg_ratio*100,
                            gt_folder=cfg.gt_folder,
                            eva_ratio=cfg.eval_ratio, 
                            img_desc_folder_0=cfg.img_desc_folder_0,
                            img_desc_folder_5=cfg.img_desc_folder_5,
                            img_desc_folder_10=cfg.img_desc_folder_10,
                            img_desc_folder_15=cfg.img_desc_folder_15,
                            img_desc_folder_cb=cfg.img_desc_folder_cb,
                            velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
                            velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7
                            )            
    
    # test_dataset = seq_test(sequs=[cfg.seq], 
    #                         neg_ratio=cfg.neg_ratio*100,
    #                         gt_folder=cfg.gt_folder,
    #                         eva_ratio=cfg.eval_ratio*0, 
    #                         img_desc_folder_0=cfg.img_desc_folder_0,
    #                         img_desc_folder_5=cfg.img_desc_folder_5,
    #                         img_desc_folder_10=cfg.img_desc_folder_10,
    #                         img_desc_folder_15=cfg.img_desc_folder_15,
    #                         img_desc_folder_cb=cfg.img_desc_folder_cb,
    #                         velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
    #                         velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7
    #                         )

    batch_size = cfg.batch_size
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    # test_loader = DataLoader(
    #     dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=cfg.learning_rate, weight_decay=1e-6)

    #退火学习策略
    import math
    warmup_epoch = 100*0.1
    total_epoch = 500*0.1
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    epoch = cfg.max_epoch
    starting_epoch = 0
    batch_num = 0

    if not cfg.model == "":
        checkpoint = torch.load(cfg.model)
        starting_epoch = checkpoint['epoch']
        batch_num = checkpoint['batch_num']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model_path_base='./checkpoints/model_mae/'
    model_path_sper='seq'
    model_path =  model_path_base+model_path_sper+cfg.seq+'_best.pth'
    pretrained_model_path = model_path

    model_mae = torch.load(pretrained_model_path)
    model_mae.to(device=device)

    base_learning_rate = (1.5e-4)
    weight_decay = 5e-2
    warmup_epoch_mae = 100*0.1
    total_epoch_mae = 500*0.1
    optim_mae = torch.optim.AdamW(model_mae.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
    lr_func_mae = lambda epoch: min((epoch + 1) / (warmup_epoch_mae + 1e-8), 0.5 * (math.cos(epoch / total_epoch_mae * math.pi) + 1))
    lr_scheduler_mae = torch.optim.lr_scheduler.LambdaLR(optim_mae, lr_lambda=lr_func_mae, verbose=True)



    for i in range(starting_epoch, epoch):
        net.train()
        pred = []
        gt = []
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch '+str(i), leave=False):
            optimizer.zero_grad()
            optim_mae.zero_grad()


            input=torch.cat((sample_batch["img_descs_0"].unsqueeze(1),sample_batch["img_descs_5"].unsqueeze(1),sample_batch["img_descs_10"].unsqueeze(1),sample_batch["img_descs_15"].unsqueeze(1)),1)
            input=input.to(device) #b*4*12*90

            seq_contour_matrix, mask=model_mae(input)   #64, 1, 12, 90
            seq_contour_matrix=torch.clamp(seq_contour_matrix,min=0.0,max=1.0)


            #保存生成的轮廓矩阵
            pad = (135, 135)  
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
                sample_batch["desc2_7"].to(device=device),)
            yaw_sec_gt=sample_batch["yaw_e_sec"].to(device=device)
            out_cat=out_cat.permute(1,0)
            labels = sample_batch["label"].to(device=device)
            # print('out',out)
            # print('labels',labels)

            weights=torch.zeros(out_cat.shape[1])
            for fov_i in range(len(weights)):
                #加1是为了防止当所有样本朝向都一直时，算出的权重为0
                weights[fov_i]=(1+np.sum(labels.cpu().numpy() == 1)
                                -np.sum((yaw_sec_gt.cpu().numpy() == fov_i) * (labels.cpu().numpy() == 1)))/np.sum(labels.cpu().numpy() == 1)
            # print('weights',weights)
            weights=weights.to(device=device)
            loss_ce_func = torch.nn.CrossEntropyLoss(weight=weights,reduce=False)
            loss_ce = loss_ce_func(out_cat, yaw_sec_gt.long())
            loss_ce=torch.mean(loss_ce*labels)

            loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
                out, labels)  #相当于先对输入求sigmoid，再与label对比求交叉熵
            loss2 = labels*diff*diff+(1-labels)*torch.nn.functional.relu(
                cfg.margin-diff)*torch.nn.functional.relu(cfg.margin-diff)
            loss2 = torch.mean(loss2)
            # loss = loss1+loss2+loss_ce
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            optim_mae.step()

            with torch.no_grad():
                writer.add_scalar(
                    'total loss', loss.cpu().item(), global_step=batch_num)
                writer.add_scalar('loss1', loss1.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss2', loss2.cpu().item(),
                                  global_step=batch_num)
                # writer.add_scalar('loss_ce', loss_ce.cpu().item(),
                #                   global_step=batch_num)
                batch_num += 1
                outlabel = out.cpu().numpy()
                label = sample_batch['label'].cpu().numpy()
                mask = (label > 0.9906840407) | (label < 0.0012710163)
                label = label[mask]
                label[label < 0.5] = 0
                label[label > 0.5] = 1
                pred.extend(outlabel[mask].tolist())
                gt.extend(label.tolist())

        # lr_scheduler.step()
        lr_scheduler_mae.step()

        pred = np.array(pred, dtype='float32')
        pred = np.nan_to_num(pred)
        gt = np.array(gt, dtype='float32')
        precision, recall, _ = metrics.precision_recall_curve(gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('Train F1:', trainaccur)
        print('i',i)
        writer.add_scalar('train f1', trainaccur, global_step=i)
  

        if i%3==0:
            evalaccur = test(net=net, dataloader=eval_loader,model_mae=model_mae)
            writer.add_scalar('eval f1', evalaccur, global_step=i)
            print('Eval_train F1:', evalaccur)

            # lastaccur = test(net=net, dataloader=test_loader,model_mae=model_mae)
            # writer.add_scalar('etest f1', lastaccur, global_step=i)
            # print('Eval_test F1:', lastaccur)
            torch.save({'epoch': i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(
            ), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.seq, str(i)+'.ckpt'))
            torch.save({'epoch': i, 'state_dict': model_mae.state_dict(), 'optimizer': optim_mae.state_dict(
            ), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.seq, str(i)+'_mae.ckpt'))


def test(net, dataloader,model_mae):
    net.eval()
    model_mae.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Eval", leave=False):
            input=torch.cat((sample_batch["img_descs_0"].unsqueeze(1),sample_batch["img_descs_5"].unsqueeze(1),sample_batch["img_descs_10"].unsqueeze(1),sample_batch["img_descs_15"].unsqueeze(1)),1)
            input=input.to(device) #b*4*12*90
            # os._exit()

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
        return testaccur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log/',
                        help='Log dir.')
    parser.add_argument('--seq', default='00',
                        help='Sequence to test.')
    parser.add_argument('--all_seqs', type=list, default=['00', '01', '02', '03', '04', '05', '06', '07', '08',
                        '09', '10'], help="All sequence. [default: ['00','01','02','03','04','05','06','07','08','09','10'] ]")
    parser.add_argument('--neg_ratio', type=float, default=1,
                        help='The proportion of negative samples used during training. [default: 1]')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='Proportion of samples used for validation. [default: 0.1]')
    
    parser.add_argument('--gt_folder', default="./data/gt_pairs",
                        help='Folder containing groundtruth files. ')
    
    parser.add_argument('--velo_desc_folder_0', default="./data/lidar_desc/0",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_1', default="./data/lidar_desc/1",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_2', default="./data/lidar_desc/2",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_3', default="./data/lidar_desc/3",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_4', default="./data/lidar_desc/4",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_5', default="./data/lidar_desc/5",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_6', default="./data/lidar_desc/6",
                        help='Folder containing lidar-slice descriptors')
    parser.add_argument('--velo_desc_folder_7', default="./data/lidar_desc/7",
                        help='Folder containing lidar-slice descriptors')
    
    parser.add_argument('--img_desc_folder_0', default="./data/img_desc/0",
                        help='Folder containing img descriptors')
    parser.add_argument('--img_desc_folder_5', default="./data/img_desc/5",
                        help='Folder containing img descriptors')
    parser.add_argument('--img_desc_folder_10', default="./data/img_desc/10",
                        help='Folder containing img descriptors')
    parser.add_argument('--img_desc_folder_15', default="./data/img_desc/15",
                        help='Folder containing img descriptors')
    parser.add_argument('--img_desc_folder_cb', default="./data/img_desc/combine",
                        help='Folder containing img descriptors')
  
    parser.add_argument('--model', default="",
                        help='Pretrained model. [default: ""]')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Epoch to run.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size during training. [default: 1024]')
    parser.add_argument('--learning_rate', type=float, default=0.02,
                        help='Initial learning rate. [default: 0.02]')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-6, help='Weight decay. [default: 1e-6]')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin used in contrastive loss. [default: 0.2]')
    cfg = parser.parse_args()
    if(not os.path.exists(os.path.join(cfg.log_dir, cfg.seq))):
        os.makedirs(os.path.join(cfg.log_dir, cfg.seq))
    train(cfg)
