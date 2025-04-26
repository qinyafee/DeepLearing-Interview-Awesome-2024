https://zhuanlan.zhihu.com/p/642735557

https://www.bilibili.com/video/BV1JC4y1A7hp/?spm_id_from=333.337.search-card.all.click&vd_source=8b0b4338476123356ac8c9e506a70b45
# 带lidar

1. 多传感器交叉验证
   1. 2d 模型和 3d模型结果比对，只保留一致性好的数据
   2. 动态OD可以做到100%自动化
    ![alt text](image.png)
2. lidar camera重投影误差 < 3pix
   ![alt text](image-1.png)
   ![alt text](image-2.png)
3. 单趟建图
   1. 高速行驶初始化难，引入lidar-seg
   ![alt text](image-3.png)
4. 多趟：
   1. 提升回环：引入lidar-seg语义
   ![alt text](image-4.png)
5. maptr大模型预刷
   1. 输入语义、intensity，还有lidar+cam pixel融合，提供纹理图，应对intensity对比度不高的问题
   ![alt text](image-5.png)
   ![alt text](image-6.png)
   ![alt text](image-7.png)
6.  泊车预刷模型：车位、限位器、墙壁
    1.  人工构建回环评测
    ![alt text](image-8.png)
7.  动态OD
    1. 3d不一定全都检测出来，60-70%就够了；图像2d要全都有
    ![alt text](image-9.png)
8.  静态：比如锥桶
    1.  lidar分割大模型->3d proposal很多噪声
    2.  2d锥桶大模型，交叉验证可以解决。【比较依赖这个大模型】
    ![alt text](image-10.png)
9.  occ：动静态点云分割，不依赖白名单
    1.  静态建图
    2.  动态：ICP+scene flow
    ![alt text](image-11.png)

# 纯视觉重建
10. 纯视觉重建
    1.  比原始MVS，使用odom作为初值，提升效率和egopose精度
    ![alt text](image-12.png)
    ![alt text](image-13.png)
11. 稠密点云：用了MVS方案+语义【质量很高，但不会一直做，耗资源，还是要根据任务来】
    ![alt text](image-14.png)
12. 在sfm的准确pose+稀疏点云基础上，nerf做路面重建：
    1.  2.5d，平面+mlp高度
    2.  mesh重建监督：rgb, 语义
    3.  3d -- 2d投影一致性非常高
    4.  ![alt text](image-15.png)
    5.  ![alt text](image-16.png)
    6.  ![alt text](image-17.png)
    7.  ![alt text](image-18.png)
13. TSR：依赖2d 3d大模型
    1.  以SFM提供的准确的视觉pose和稀疏3D点为基础，获取3D空间中交通牌的proposal，之后和2D图像预刷结果进行联合优化，角点重投影误差<2px
    2.  优化TSR朝向
    3.  ![alt text](image-19.png)
    4.  ![alt text](image-20.png)
14. 纯视觉动态
    1.  bev大模型出bbox，精度不如lidar；但是对online模型有效果提升就可以
    ![alt text](image-21.png)
    ![alt text](image-22.png)
15. 纯视觉occ，vidar
    1.  模型是DRO，依赖depth
    2.  仿真数据，可提升动态物体边缘的depth锐利程度
    3.  ![alt text](image-23.png)
    4.  ![alt text](image-24.png)
    5.  ![alt text](image-25.png)
16. ![alt text](image-26.png)
17. ![挖掘](image-27.png)
18. 数据合成
    1.  ![alt text](image-28.png)
    2.  ![alt text](image-29.png)
    3.  ![alt text](image-30.png)
    4.  ![alt text](image-31.png)
19. ![alt text](image-32.png)