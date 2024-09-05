import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math
from tqdm import tqdm
import copy
import torch

def load_poses(pose_path):
  """ Load ground truth poses (T_w_cam0) from file.
      Args: 
        pose_path: (Complete) filename for the pose file
      Returns: 
        A numpy array of size nx4x4 with n poses as 4x4 transformation 
        matrices
  """
  # Read and parse the poses
  poses = []
  try:
    if '.txt' in pose_path:
      with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
          T_w_cam0 = T_w_cam0.reshape(3, 4)
          T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
          poses.append(T_w_cam0)
    else:
      poses = np.load(pose_path)['arr_0']
  
  except FileNotFoundError:
    print('Ground truth poses are not avaialble.') 
  return np.array(poses)

def load_calib(calib_path):
  """ Load calibrations (T_cam_velo) from file.
  """
  # Read and parse the calibrations
  T_cam_velo = []
  try:
    with open(calib_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        if 'Tr:' in line:
          line = line.replace('Tr:', '')
          T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
          T_cam_velo = T_cam_velo.reshape(3, 4)
          T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
  
  except FileNotFoundError:
    print('Calibrations are not avaialble.')
  
  return np.array(T_cam_velo)

def euler_angles_from_rotation_matrix(R):
  """ From the paper by Gregory G. Slabaugh,
      Computing Euler angles from a rotation matrix
      psi, theta, phi = roll pitch yaw (x, y, z)
      Args:
        R: rotation matrix, a 3x3 numpy array
      Returns:
        a tuple with the 3 values psi, theta, phi in radians
  """
  
  def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)
  
  phi = 0.0
  if isclose(R[2, 0], -1.0):
    theta = math.pi / 2.0
    psi = math.atan2(R[0, 1], R[0, 2])
  elif isclose(R[2, 0], 1.0):
    theta = -math.pi / 2.0
    psi = math.atan2(-R[0, 1], -R[0, 2])
  else:
    theta = -math.asin(R[2, 0])
    cos_theta = math.cos(theta)
    psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
    phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
  return psi, theta, phi


def run(seq='00'):
    pose_file = "/media/l/yp2/KITTI/odometry/dataset/poses/"+seq+".txt"
    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    id_pos = np.argwhere(dis < 3)
    id_neg = np.argwhere(dis > 20)
    # id_pos=id_pos[id_pos[:,0]-id_pos[:,1]>50]
    id_neg = id_neg[id_neg[:, 0] > id_neg[:, 1]]
    id_pos = np.concatenate(
        [id_pos, (id_pos[:, 0]*0+1).reshape(-1, 1)], axis=1)
    id_neg = np.concatenate([id_neg, (id_neg[:, 0]*0).reshape(-1, 1)], axis=1)
    print(id_pos.shape)
    np.savez(seq+'.npz', pos=id_pos, neg=id_neg)

#从矩阵A中截取出B，其中B的尺寸为（m,m)，m是A的最下一行，这一行的最后一列元素A[m,n]>=15
def get_matrix_B(A):
  n = A.shape[0]
  m = 0
  for i in range(n-1, -1, -1):
    if A[i, n-1] >= 15:
      m = i
      break
  B = A[:m+1, :m+1]
  return B

def run_sigmoid(seq='00',save_dir='./',ratio=1):
    '''
    test的时候使用
    label只有1/0没有小数 分别对应正负样本
    正样本,索引差至少50个scan,地点间距小于3m
    负样本,地点间距大于20m
    保存成txt格式 输出为 scan 0 和 scan 1的索引,以及他们是否是相同地点 
    '''
    pose_file = "/workspace/data/SemanticKITTI/extracted_KITTI_data_all/dataset/poses/"+seq+".txt"
    calib_file="/workspace/data/SemanticKITTI/extracted_KITTI_data_all/dataset/sequences/"+seq+"/calib.txt"
    
    # load calibrations
    #关系：T_velo2cam*p_in_velo=p_in_cam
    T_velo2cam = load_calib(calib_file)
    T_velo2cam = np.asarray(T_velo2cam).reshape((4, 4))
    T_cam2velo = np.linalg.inv(T_velo2cam)

    # load poses
    #poses为相机坐标系从t时刻到0时刻的变换，这里，我的理解是到0时刻世界坐标系的变化。理由：可能由于安装或者标定的关系，poses[0]并不是单位矩阵。
    #pose0_inv.dot(pose).dot(T_velo2cam)*p_in_velo=p_in_cam_of_time0
    poses = load_poses(pose_file)
    pose0_inv = np.linalg.inv(poses[0])
    # for KITTI dataset, we need to convert the provided poses
    # from the camera coordinate system into the LiDAR coordinate system 
    #curr lidar 2 init lidar  即poses[i]*point[i]=点在初始位置的lidar坐标系下的坐标
    poses_new = []
    for pose in poses:
      poses_new.append(T_cam2velo.dot(pose0_inv).dot(pose).dot(T_velo2cam))
    poses = np.array(poses_new).reshape((-1,16))
    poses = poses[:, [7,3]] #第一个是y 第二个是x
    
    
    #每一行利用3x4转移矩阵代表左边相机系统位姿，转移矩阵将当前帧左边相机系统中的一个点映射到第0帧的坐标系统中。转移矩阵中平移的部分表示当前相机位置(相对于第0帧)
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    #两两scan之间的距离
    dis = np.sqrt(np.abs(dis))

    #舍弃最后的几个scans，因为他们没有距离他们在前向15m的scan
    dis=get_matrix_B(dis)

    #计算两两的x差距,(p,q)的值，表示位置p到位置q的差距
    poses_x = poses[:, 1]  #第二列对应lidar坐标系x
    poses_x=torch.tensor(poses_x)
    x_diff = poses_x.unsqueeze(1) - poses_x
    x_diff=x_diff.numpy()
    
    #计算两两的y差距
    poses_y = poses[:, 0]  #第一列对应lidar坐标系y
    poses_y=torch.tensor(poses_y)
    y_diff = poses_y.unsqueeze(1) - poses_y
    y_diff=y_diff.numpy()

    x_diff = x_diff[:len(dis), :len(dis)]
    y_diff = y_diff[:len(dis), :len(dis)]

    score = 1.-1./(1+np.exp((10.-dis)/1.5))
    #两两scan之间的距离都赋值1,其余按照公式计算
    score[dis < 3] = 1

    id = np.argwhere(dis > -1)

    id_dis = id[id[:, 0] >= id[:, 1]]
    print('id.shape',id.shape)
    print('id_dis.shape',id_dis.shape)
    label = score[(id_dis[:, 0], id_dis[:, 1])]
    label = label.reshape(-1, 1)

    
    out = np.concatenate((id_dis, label), 1)
    # out_pos = out[out[:, 2] > 0.1]
    # out_neg = out[out[:, 2] <= 0.1]
    #在训练时距离小于3m，为正;大于约20m为负
    out_pos = out[out[:, 2] >= 1]
    out_neg = out[out[:, 2] <= 0.0013]

    print('out_pos.shape',out_pos.shape)
    print('out_neg.shape',out_neg.shape)

    out_pos=out_pos[out_pos[:,0]-out_pos[:,1]>50]
    out_pos[:,2]=1
    out_neg[:,2]=0


    # print(out[out[:, 2] > 0.5].shape)
    # print(out[out[:, 2] >= 1.0].shape)
    print('out_pos.shape',out_pos.shape)
    print('out_neg.shape',out_neg.shape)

    #确定负样本的数量，并随机选择
    num_neg=ratio*len(out_pos)
    print('num_neg',num_neg)
    idx = np.random.choice(len(out_neg), size=int(num_neg), replace=False)
    out_neg = out_neg[idx,:]


    out_pos_and_neg=np.concatenate((out_pos,out_neg),axis=0)
    out_pos_and_neg=out_pos_and_neg.astype(int).astype(str)
    out_pos_and_neg[:,:2]=np.char.zfill(out_pos_and_neg[:,:2],6)
    # print(out_pos_and_neg[:3,:])
    print('out_pos_and_neg.shape',out_pos_and_neg.shape)
    # print(type(out_pos_and_neg[0,0]))

    #保存
    save_dir=os.path.join(save_dir, 'neg_'+str(ratio))
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
      print('create dir:',save_dir)
    save_dir=os.path.join(save_dir, seq+'.txt')
    print('save_dir',save_dir)

    np.savetxt(save_dir,out_pos_and_neg, delimiter=" ",fmt='%s')
    # np.savez(save_dir+seq+'.npz', pos=out_pos, neg=out_neg)


if __name__ == '__main__':
    seq = "00"
    save_dir="/workspace/data/RINet/pairs_kitti_seq/"
    ratio=1
    #如果有超过python3 和 gen_pair.py 以外的输入，后续第一个参数对应seq数，第二个参数对应ground truth的保存地址
    if len(sys.argv) > 1:
        seq = sys.argv[1]
    if len(sys.argv) > 2:
        save_dir=sys.argv[2]
        if os.path.exists(save_dir):
          print('existing dir:',save_dir)
        else:
          os.makedirs(save_dir)
          print('create dir:',save_dir)
    # print(len(sys.argv) )
    if len(sys.argv) > 3:
        ratio = int(sys.argv[3])

    # os._exit()
    run_sigmoid(seq,save_dir,ratio)
    # run(seq)
