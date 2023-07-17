import torch
# Convert the point cloud data to spherical coordinates based on explanation from authors of original paper
# https://en.wikipedia.org/wiki/Spherical_coordinate_system

def get_spherical_coor(l_img):
  l_img_new = torch.zeros_like(l_img)
  l_img_new[:, 0] = torch.sqrt(l_img[:, 0] ** 2 + l_img[:, 1] ** 2 + l_img[:, 2] ** 2)
  l_img_new[:, 1] = torch.arccos(l_img[:, 2] / l_img_new[:, 0])

  # x > 0
  l_img_new[l_img[:, 0] > 0, 2] = torch.arctan(l_img[l_img[:, 0] > 0, 1] / l_img[l_img[:, 0] > 0, 0])
  # x < 0 and y >= 0
  t1 = l_img[:, 0] < 0 
  t2 = l_img[:, 1] >= 0
  idxs = torch.logical_and(t1, t2)
  l_img_new[idxs, 2] = torch.arctan(l_img[idxs, 1] / l_img[idxs, 0]) + torch.pi
  # x < 0 and y < 0
  t1 = l_img[:, 0] < 0 
  t2 = l_img[:, 1] < 0
  idxs = torch.logical_and(t1, t2)
  l_img_new[idxs, 2] = torch.arctan(l_img[idxs, 1] / l_img[idxs, 0]) - torch.pi
  # x == 0 and y > 0
  t1 = l_img[:, 0] == 0 
  t2 = l_img[:, 1] > 0
  idxs = torch.logical_and(t1, t2)
  l_img_new[idxs, 2] = torch.pi / 2
  # x == 0 and y < 0
  t1 = l_img[:, 0] == 0 
  t2 = l_img[:, 1] > 0
  idxs = torch.logical_and(t1, t2)
  l_img_new[idxs, 2] = - torch.pi / 2

  return l_img_new

# Algorithm 1 https://arxiv.org/pdf/1812.01687.pdf
# Keep n_p/T to 5 as is done in the paper for final project

def drop_points_from_saliency_map(coor_X, sphere_X, m_weights, alpha, n_p, T):
  d = int(n_p / T)
  batch_size = sphere_X.shape[0]
  loss_wrt_in = sphere_X.grad
  for t in range(0, T):
    ri = sphere_X[:, :, 0]
    m_tuple = torch.median(coor_X, dim=1)
    medians = m_tuple[0]
    subt = torch.sub(coor_X, medians)
    partial_loss_wrt_r_center = torch.einsum('bpd,bpd->bp', subt, loss_wrt_in)
    saliency = - (ri ** alpha) * partial_loss_wrt_r_center
    sort, indices = torch.sort(saliency, descending=False)
    drop_idx = indices[:, 0:d]
    new_samples = []
    new_grads = []
    new_coors = []
    for s in range(0, batch_size):
      sample = sphere_X[s, :, :]
      grad_sample = loss_wrt_in[s, :, :]
      coor_one = coor_X[s, :, :]
      points = drop_idx[s, :]
      for i in points:
        sample = torch.cat((sample[0:i, :], sample[i+1:, :]))
        grad_sample = torch.cat((grad_sample[0:i, :], grad_sample[i+1:, :]))
        coor_one = torch.cat((coor_one[0:i, :], coor_one[i+1:, :]))
      new_samples.append(sample)
      new_grads.append(grad_sample)
      new_coors.append(coor_one)
    sphere_X = torch.stack(new_samples)
    loss_wrt_in = torch.stack(new_grads)
    coor_X = torch.stack(new_coors)
  # return sphere_X
  return coor_X

