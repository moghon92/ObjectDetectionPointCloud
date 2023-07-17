import os
import saliency_utils as utils
import TestModel as tm
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter

path='/home/FinalProject/OpenPCDet/data/kitti/training/velodyne/'
arr = os.listdir(path)
# arr = ['001100.bin', '007251.bin']
target = torch.empty(1, dtype=torch.long).random_(9)

loss_fn = torch.nn.CrossEntropyLoss()
n_p = 20
T = 4

count = 0
for b in arr[0:]:
  b_array = np.fromfile(path + b, np.float32)
  lidar_img = torch.from_numpy(b_array)
  l_img = np.reshape(lidar_img, (-1, 4))[:, :4]
  print(l_img.shape)
  l_img = l_img[None, :]
  img = utils.get_spherical_coor(l_img)
  X = Variable(img, requires_grad=True)
  
  model = tm.TestModel(num_points=img.shape[1])
  optim = torch.optim.SGD(model.parameters(), lr=1e-6)
  for t in range(1):
    out = model(X)
    loss = loss_fn(out, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

  new_lidar = utils.drop_points_from_saliency_map(l_img, X, None, 1, n_p, T)

  final_lidar = new_lidar.cpu().detach().numpy()

  # x = final_lidar[:, 0] * np.cos(final_lidar[:, 2]) * np.sin(final_lidar[:, 1])
  # y = final_lidar[:, 0] * np.sin(final_lidar[:, 2]) * np.sin(final_lidar[:, 1])
  # z = final_lidar[:, 0] * np.cos(final_lidar[:, 1])
  # final_lidar[:, 0] = x
  # final_lidar[:, 1] = y
  # final_lidar[:, 2] = z

  final_lidar.tofile('./train_velo/' + b)
  print(final_lidar.shape)

  print("Finished Lidar " + b)
  print("Lidar " + str(count))
  count += 1
#o /home/FinalProject/OpenPCDet/data/kitti/training/velodyne

