# Video_Classification_ResNet3D_Pytorch
Using ResNet3D to train on Kinetics form scratch or fine-tune on UCF-101(or others) with Kinetics pretrained model.



此repo中的代码和模型来源于（https://github.com/kenshohara/3D-ResNets-PyTorch） ，没有原创性。





### 一、利用ResNet3D-34在Kinetics训练好的模型，进行UCF-101的微调

1.环境配置

（1）pytorch1.1，cuda9.0，python3.7；（一定要python3.7以上，否则后续抽帧会有问题，涉及到一个subprocess的函数，3.7以下的版本中subprocess函数参数设置和3.7不一样）

（2）ffmpeg：处理视频文件的库，将视频文件处理成帧；

安装了anaconda的可以直接conda install安装：

```shell
conda install ffmpeg
```

否则就按照github上的提示安装。



2.预训练模型下载和ucf-101数据集下载

模型地址： https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4. 里面有很多Kinetics上训练好的模型，选择r3d50_K_200ep.pth（使用resnet-50在Kineeics-700上训练好的模型）进行预训练。

数据集地址：https://www.crcv.ucf.edu/data/UCF101.php .



Master branch:

3.数据集预处理

预处理之前要按照原repo的readme中来放置数据和相关文件。（我是新建了一个data文件夹，在里面分别建立ucf_videos，results文件夹来放置数据集和训练好的模型，预训练的模型放在models文件夹中，同时还有ucf101.json索引文件）

（1）.avi视频文件转换为.jpg图像文件

```shell
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

例如在我的服务器上为：

```shell
python -m util_scripts.generate_video_jpgs /home/sunzheng/Video_Classification/ResNet3D/data/UCF-101/ /home/sunzheng/Video_Classification/ResNet3D/data/ucf101_videos/jpg/ ucf101
```

（2）使用.jpg文件生成数据集索引文件

```shell
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

例如在我的服务器上为：

```shell
python -m util_scripts.ucf101_json /home/sunzheng/Video_Classification/ResNet3D/data/ucfTrainTestlist /home/sunzheng/Video_Classification/ResNet3D/data/ucf101_videos/jpg/ /home/sunzheng/Video_Classification/ResNet3D/data/
```



4.使用Kinetics上预训练的模型进行微调训练UCF-101

```shell
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```



成功训练！





