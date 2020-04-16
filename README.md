# Video_Classification_ResNet3D_Pytorch
Using ResNet3D to train on Kinetics form scratch or fine-tune on UCF-101(or others) with Kinetics pretrained model.



此repo中的代码和模型来源于（https://github.com/kenshohara/3D-ResNets-PyTorch） ，没有原创性。





### 一、利用ResNet3D-50在Kinetics训练好的模型，进行UCF-101的微调

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

预处理之前要按照原repo的readme中来放置数据和相关文件。（我是新建了一个data文件夹，在里面分别建立ucf101_videos（放置数据集），results文件夹（放置微调过程中训练好的模型，models文件夹（放置Kinetics上训练好的模型），同时还有ucf101.json索引文件）

（1）.avi视频文件转换为.jpg图像文件

```shell
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

例如在我的服务器上为：

```shell
python -m util_scripts.generate_video_jpgs /home/sunzheng/Video_Classification/data/UCF-101/ /home/sunzheng/Video_Classification/data/ucf101_videos/jpg/ ucf101
```

（2）使用.jpg文件生成数据集索引文件

```shell
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

例如在我的服务器上为：

```shell
python -m util_scripts.ucf101_json /home/sunzheng/Video_Classification/data/ucfTrainTestlist /home/sunzheng/Video_Classification/data/ucf101_videos/jpg/ /home/sunzheng/Video_Classification/data/
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



成功训练，训练200个epoch，结果为：（源代码竟然没有计算一个平均精度来衡量！）

![result_ucf](/Users/momo/Documents/Video_Classification_ResNet3D_Pytorch/result_ucf.png)

手动算了一下最终测试集的平均精度，结果为81.57%，和文章中的89.3%有差距。





### 二、利用ResNet3D-50在Kinetics训练好的模型，进行HMDB-51数据集的微调

1.数据集和训练好的模型下载：

网址：https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ ，包括数据集文件和训练集测试集划分文件；模型下载和上述一样。



Master branch:

2.数据集预处理

预处理之前要按照原repo的readme中来放置数据和相关文件。（我是新建了一个data_hmdb文件夹，在里面分别建立hmdb51_videos（放置抽帧的图片），results文件夹（放置微调过程中保存的模型），models文件夹（放置Kinetics预训练的模型），同时还有ucf101.json索引文件）

（1）.avi视频文件转换为.jpg图像文件

```shell
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path hmdb51
```

例如在我的服务器上为：

```shell
python -m util_scripts.generate_video_jpgs /home/sunzheng/Video_Classification/data_hmdb/hmdb51_org/ /home/sunzheng/Video_Classification/data_hmdb/hmdb51_videos/jpg/ hmdb51
```

（2）使用.jpg文件生成数据集索引文件

```shell
python -m util_scripts.hmdb51_json annotation_dir_path jpg_video_dir_path dst_json_path
```

例如在我的服务器上为：

```shell
python -m util_scripts.hmdb51_json /home/sunzheng/Video_Classification/data_hmdb/testTrainMulti_7030_splits /home/sunzheng/Video_Classification/data_hmdb/hmdb51_videos/jpg/ /home/sunzheng/Video_Classification/data_hmdb/
```



3.使用Kinetics上预训练的模型进行微调训练HMDB-51

```shell
python main.py --root_path ~/data --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results --dataset hmdb51 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json
--result_path results --dataset hmdb51 --n_classes 101 --n_pretrain_classes 700
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```



成功训练，训练200个epoch，结果为：

![result_hmdb](/Users/momo/Documents/Video_Classification_ResNet3D_Pytorch/result_hmdb.png)

手动算了一下最终测试集的平均精度，结果为53.19%，和文章中的61.0%有差距。





### 三、利用ResNet3D-50在Kinetics训练好的模型，进行打架行为识别数据集的微调

1.环境配置和微调UCF-101数据集的时候是一致的



2.数据集下载以及使用说明

（1）数据集下载：https://pan.baidu.com/s/1kwf_oWME5BUOVhtFUj9e5g 提取码：jszq

数据集使用文档：https://pan.baidu.com/s/1jo0UKzbb8ZZ_XzycMS3UMA 提取码：rgt0



3.数据集预处理

