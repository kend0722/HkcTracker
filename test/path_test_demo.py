# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/25 下午2:01
@Author  : Kend
@FileName: path_test_demo.py
@Software: PyCharm
@modifier:
"""
import os.path as osp

save_folder = r"D:\kend\WorkProject\Hk_Tracker\visualization\vis_folder\demo_output\2024_11_25_14_00_40"
path = r"D:\kend\WorkProject\Hk_Tracker\data\videos\palace.mp4"
print("path", path.split("/")[-1])  # path D:\kend\WorkProject\Hk_Tracker\data\videos\palace.mp4
save_path = osp.join(save_folder, path.split("/")[-1])
print(save_path)    # D:\kend\WorkProject\Hk_Tracker\data\videos\palace.mp4

