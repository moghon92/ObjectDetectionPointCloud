import os
import numpy as np

def main():
    # path='/home/FinalProject/OpenPCDet/data/kitti/testing/velodyne/'
    path='/home/chris_procak_gmail_com/helper/debug/'
    arr = os.listdir(path)
    for b in arr[0:1]:
      b_array = np.fromfile(path + b, np.float32)
      print(b_array.shape)
      l_img = np.reshape(b_array, (-1, 4))[:, :4]
      print(l_img.shape)
      x = l_img[:, 0] * np.cos(l_img[:, 2]) * np.sin(l_img[:, 1])
      y = l_img[:, 0] * np.sin(l_img[:, 2]) * np.sin(l_img[:, 1])
      z = l_img[:, 0] * np.cos(l_img[:, 1])
      l_img[:, 0] = x
      l_img[:, 1] = y
      l_img[:, 2] = z
      l_img = np.reshape(l_img, (-1,))
      print(l_img.shape)
      l_img.tofile('./velodyne_debug/' + b)
    
main()
