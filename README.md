# Video_Classification_ResNet3D_Pytorch
Using ResNet3D to train on Kinetics form scratch or fine-tune on UCF-101(or others) with Kinetics pretrained model.

此repo中前两部分的使用Kinetics上训练好的模型来微调UCF-101和HMDB-51的代码和模型来源于（https://github.com/kenshohara/3D-ResNets-PyTorch ） ，没有原创性；第三部分打架行为识别部分的前面数据预处理是根据打架数据集写的，后续的训练代码是ResNet3D中的。



### 一、利用ResNet3D-50以及R(2+1)D-50在Kinetics训练好的模型，进行UCF-101的微调

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
python -m util_scripts.ucf101_json /home/sunzheng/Video_Classification/data/ucfTrainTestlist/ /home/sunzheng/Video_Classification/data/ucf101_videos/jpg/ /home/sunzheng/Video_Classification/data/
```



4.使用ResNet3D在Kinetics上预训练的模型进行微调训练UCF-101

（1）微调训练模型

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



（2）评测训练好的模型

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --resume_path results/save_200.pth \
--model_depth 50 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data/ucf101_01.json /home/sunzheng/Video_Classification/data/results/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-1 accuracy
top-1 accuracy: 0.8977002379064235
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-3 accuracy
top-3 accuracy: 0.9777954004758128
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-5 accuracy
top-5 accuracy: 0.9875759978852763
```



5.使用R(2+1)D在Kinetics上预训练的模型进行微调训练UCF-101

（1）微调训练模型

```shell
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results_finetune_r2p1d --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/r2p1d50_K_200ep.pth --ft_begin_module fc \
--model resnet2p1d --model_depth 50 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results_finetune_r2p1d --dataset ucf101 --resume_path results_finetune_r2p1d/save_200.pth \
--model_depth 50 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data/ucf101_01.json /home/sunzheng/Video_Classification/data/results_finetune_r2p1d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-1 accuracy
top-1 accuracy: 0.9217552207242928
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-3 accuracy
top-3 accuracy: 0.9867829764736982
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-5 accuracy
top-5 accuracy: 0.9936558287073751
```





### 二、利用ResNet3D-50以及R(2+1)D-50在Kinetics训练好的模型，进行HMDB-51数据集的微调

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



3.使用ResNet3D在Kinetics上预训练的模型进行微调训练HMDB-51

（1）微调训练

```shell
python main.py --root_path ~/data --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results --dataset hmdb51 --n_classes 51 --n_pretrain_classes 700 \
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json
--result_path results --dataset hmdb51 --n_classes 51 --n_pretrain_classes 700
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc
--model resnet --model_depth 50 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results --dataset hmdb51 --resume_path results/save_200.pth \
--model_depth 50 --n_classes 51 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_hmdb/hmdb51_1.json /home/sunzheng/Video_Classification/data_hmdb/results/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-1 accuracy
top-1 accuracy: 0.5784313725490197
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-3 accuracy
top-3 accuracy: 0.7803921568627451
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-5 accuracy
top-5 accuracy: 0.8581699346405228
```



4.使用R(2+1)D在Kinetics上预训练的模型进行微调训练HMDB-51

（1）微调训练

```shell
python main.py --root_path ~/data --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results --dataset hmdb51 --n_classes 51 --n_pretrain_classes 700 \
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc \
--model resnet2p1d --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json
--result_path results_finetune_r2p1d --dataset hmdb51 --n_classes 51 --n_pretrain_classes 700
--pretrain_path models/r2p1d50_K_200ep.pth --ft_begin_module fc
--model resnet2p1d --model_depth 50 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_finetune_r2p1d --dataset hmdb51 --resume_path results_finetune_r2p1d/save_200.pth \
--model_depth 50 --n_classes 51 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_hmdb/hmdb51_1.json /home/sunzheng/Video_Classification/data_hmdb/results_finetune_r2p1d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-1 accuracy
top-1 accuracy: 0.6581699346405229
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-3 accuracy
top-3 accuracy: 0.8522875816993464
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-5 accuracy
top-5 accuracy: 0.9117647058823529
```





### 三、利用ResNet3D-50在Kinetics训练好的模型，进行打架行为识别数据集的微调

**这部分相比于前两部分，代码解读会较多**

1.环境配置和微调UCF-101以及HMDB-51数据集的时候是一致的



2.数据集下载以及使用说明

（1）数据集下载：https://pan.baidu.com/s/1kwf_oWME5BUOVhtFUj9e5g 提取码：jszq

数据集使用文档：https://pan.baidu.com/s/1jo0UKzbb8ZZ_XzycMS3UMA 提取码：rgt0



3.数据集预处理

抽帧：将视频文件转换为图像文件，由于该数据集的视频文件是在各个公开数据集（包括Kinetics，UCF-101，HMDB-51等）中抽取的，所以视频格式包括.mkv，.mp4，.avi等。

3.1.将.avi，.mkv，.mp4等视频文件转换为.jpg图像文件

在抽帧过程中发现打架数据集中有很多.mkv(或者.mp4.webm格式)文件无法获取视频的总帧数，进而报错。解决方法：将非.avi格式转换为.avi格式，可以成功获取总帧数。

转换命令：

```shell
ffmpeg test.mkv test.avi
```

获取视频信息代码get_video_infomation.py：

```python
import sys
import random
import os

def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)

if __name__ == '__main__':
    #file_path = '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight/angleview_p01p02_fight_a1.avi' #  获取某一视频文件的信息
    #file_path = '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight/IMPfjjPQLnU.mkv'
    file_path = '/home/sunzheng/test.avi'
    video_info = get_video_info(file_path)
    print(video_info)
    total_frames = int(video_info['nb_frames'])
    #total_duration = float(video_info['duration'])
    print('总帧数：' + str(total_frames))
    #print('总时长：' + str(total_duration))

'''
    file_path = '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight'
    file_all = os.listdir(file_path)
    for video in file_all:
        video_path = file_path + '/' + video
        print(video_path)
        video_info = get_video_info(video_path)
        print(video_info)
        total_frames = int(video_info['nb_frames'])
        print('总帧数：' + str(total_frames))
'''
```

将视频格式转换之后，可以成功运行脚本来获取视频信息。

所以下面的抽帧过程为：（1）先将数据集中所有的非.avi文件都转换为.avi格式change_style.py；（2）调用generate_video_jpgs_dj.py来抽帧。

（1）change_style.py

```python
# 此脚本用于将视频目录下的所有非.mp4或者非.avi格式的视频转换为.avi格式的视频，用于抽帧

import argparse
import subprocess
import tqdm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_path', default=None, type=str, help='Directory path of original videos')
    parser.add_argument(
        '--dst_path', default=None, type=str, help='Directory path of .avi videos')
    args = parser.parse_args()

    file_all = os.listdir(args.source_path)
    for video in tqdm.tqdm(file_all):
        name = video.split('.')[0] #  获取视频名称
        cmd = 'ffmpeg -i ' + args.source_path + video + ' ' + args.dst_path + name + '.avi' #  调用命令行执行转换格式的命令，命令为：ffmpeg -i /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/train/fight/xxx.mp4 /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/train/fight/xxx.avi
        subprocess.call(cmd, shell=True)
```



分别执行指令：

训练集打架文件夹视频转换：

```shell
python change_style.py --source_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/train/fight/ --dst_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/train/fight/
```

训练集非打架文件夹视频转换：

```shell
python change_style.py --source_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/train/non-fight/ --dst_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/train/non-fight/
```

测试集打架文件夹视频转换：

```shell
python change_style.py --source_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight/ --dst_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/val/fight/
```

测试集非打架文件夹视频转换：

```shell
python change_style.py --source_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/non-fight/ --dst_path /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/val/non-fight/
```



（2）generate_video_jpgs_dj.py抽帧

代码：

```python
# 此脚本用于对打架数据集进行抽帧

import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix: #  后缀不是avi，函数直接返回
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split() #  增加一个参数获取视频的总帧数
    ffprobe_cmd.append(str(video_file_path))
    #print(ffprobe_cmd) #  所有包含视频文件的指令列表，如['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-of', 'default=noprint_wrappers=1:nokey=1', '-show_entries', 'stream=width,height,avg_frame_rate,duration', '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/non-fight/-0IErS_cisg.mp4']

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    print(res) # 显示视频文件的基本信息，如['340', '256', '10/1', '12.000000'],分别是空间尺寸，帧率，时长
    if len(res) < 4:
        return

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])
    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])
    
    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    #print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    #print('\n')


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    #print(class_dir_path) #  视频文件类别路径，如/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/train/fight
    #print(class_dir_path.name) # 列出视频文件的所有类别名称，如fight,non-fight
    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True) #  在图像文件所在目录里生成相应的类别文件夹
    #print(dst_class_path) #  图像文件所在目录,如/home/sunzheng/Video_Classification/data_dj/dj_videos/jpg/train/fight

    for video_file_path in sorted(class_dir_path.iterdir()):
        #print(class_dir_path.iterdir())
        #print(video_file_path) #  输出视频文件所在路径，如/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/train/fight/UPaStKbekxc_000386_000396.mp4
        video_process(video_file_path, dst_class_path, ext, fps, size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        'dataset',
        default='',
        type=str,
        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet | dj)') # 包括打架数据集
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['kinetics', 'mit', 'activitynet']:
        ext = '.mp4'
    else:
        ext = '.avi'
        
    if args.dataset == 'activitynet':
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps, args.size)
                                 for video_file_path in video_file_paths)
    else:
        class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
        #print(class_dir_paths) #  输出各个视频类别的路径列表，比如[PosixPath('/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight'), PosixPath('/home/su
nzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/non-fight')]
        test_set_video_path = args.dir_path / 'test'
        #print(test_set_video_path) # /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/test，加了个test类别
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)
        #print(class_dir_paths) # 视频类别文件夹中没有test类别，输出和上面一样

        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in class_dir_paths) # class_dir_path是某一个类别的视频文件路径，如PosixPath('/home/sunzheng/Video_Classification/data_dj/Fight-da
taset-2020/videos/val/fight')
```



执行指令：

```shell
python -m util_scripts.generate_video_jpgs_dj avi_video_dir_path jpg_video_dir_path dj
```

例如在我的服务器上为（由于打架数据集已经划分好了训练集和测试集，这里要对训练集和测试集分别进行抽帧）：

训练集抽帧：

```shell
python -m util_scripts.generate_video_jpgs_dj /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/train/ /home/sunzheng/Video_Classification/data_dj/dj_videos/jpg/train/ dj
```

测试集抽帧：

```shell
python -m util_scripts.generate_video_jpgs_dj /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/val/ /home/sunzheng/Video_Classification/data_dj/dj_videos/jpg/val/ dj
```



3.2.利用.jpg文件生成数据集索引.json文件

由于打架数据集已经划分好训练集和测试集，和UCF-101以及HMDB-51不一样，这里采用如下方法生成打架数据集的.json文件（生成.json文件的原因是要适应已有代码的数据读取方式）：

（1）利用打架数据集中的数据，直接生成类似UCF-101数据集中的trainlist01.txt和testlist01.txt；

（2）将划分好的打架数据集中的训练集和测试集放到一起（这一步也是为了和UCF-101数据集的形式保持一致）；

（1）利用打架数据集的数据直接生成打架数据集的划分文档trainlist01.txt和testlist01.txt

make_index.py:

```python
import argparse
import tqdm
import os

def make_index_txt(class_path, txt_path, train):
    cate = os.listdir(class_path) # ['fight', 'non-fight']
    if train == 1:
        i = 0 # trainlist01.txt类别数
        with open(txt_path, 'w') as f:
            for is_fight in cate:
                i = i+1
                video_list = os.listdir(class_path + is_fight)
                for video in video_list:
                    f.write(is_fight + '/' + video + ' ' + str(i) + '\n')
    elif train == 0:
        with open(txt_path, 'w') as f:
            for is_fight in cate:
                video_list = os.listdir(class_path + is_fight)
                for video in video_list:
                    f.write(is_fight + '/' + video + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_train', default=None, type=int, help='make train(or test)list01.txt')
    parser.add_argument('--dst_path', default='/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/', type=str, help='path to generate txt file')
    args = parser.parse_args()

    if args.is_train == 1:
        class_path = '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/train/'
        txt_path = args.dst_path + 'trainlist01.txt'
    elif args.is_train == 0:
        class_path = '/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi/videos/val/'
        txt_path = args.dst_path + 'testlist01.txt'
    make_index_txt(class_path, txt_path, args.is_train)
```

分别运行下面两个命令来生成trainlist01.txt和testlist01.txt

```shell
python make_index.py --is_train 1 # 生成trainlist01.txt
```

```shell
python make_index.py --is_train 0 # 生成testlist01.txt
```

再将这两个划分文件分为复制为：trainlist02.txt和testlist02.txt，trainlist03.txt和testlist03.txt，为了生成ucf101_01.json，ucf101_02.json，ucf101_03.json。实际上后续训练的时候只需要一个即可。



（2）将打架数据集中的训练集和测试集放到一起

这里选择将jpg文件夹中抽好帧的图像，重新复制一份到jpg_mix文件夹中，在该文件夹中将fight和non-fight类别的训练集和测试集放到一起。



以上都完成之后，采用UCF-101生成.json的脚本命令来生成打架数据集的.json文件：

```shell
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

例如在我的服务器上为：

```shell
python -m util_scripts.ucf101_json /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020-avi /home/sunzheng/Video_Classification/data_dj/dj_videos/jpg_mix/ /home/sunzheng/Video_Classification/data_dj/
```

生成ucf101_01.json，ucf101_02.json，ucf101_03.json。

.json文件的内容如下：

```python
......"vpCp4Jqd3R0": {"subset": "validation", "annotations": {"label": "non-fight", "segment": [1, 158]}}......
```

包含了所有的视频名称，测试集还是训练集，标签以及帧数。



4.使用Kinetics上预训练的模型进行微调训练打架数据集（和微调UCF-101时的指令保持一致）

```shell
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_dj/ --video_path dj_videos/jpg_mix --annotation_path ucf101_01.json
--result_path results --dataset ucf101 --n_classes 2 --n_pretrain_classes 700
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc
--model resnet --model_depth 50 --batch_size 32 --n_threads 4 --checkpoint 5
```

训练200个epoch，batch_size为32，其余参数和代码中默认一致。



5.评测训练好的模型

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_dj --video_path dj_videos/jpg_mix --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --resume_path results/save_200.pth \
--model_depth 50 --n_classes 2 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

运行之后，opt的参数如下：

![opt_val](/Users/momo/Documents/Video_Classification_ResNet3D_Pytorch/opt_val.png)



这一步是在生成results/val.json文件，val.json文件内容如下：

```python
......"v_YoYo_g07_c03": [{"label": "PlayingFlute", "score": 0.3044358193874359}, {"label": "WallPushups", "score": 0.08625337481498718}, {"label": "PlayingPiano", "score": 0.07247895002365112}, {"label": "JugglingBalls", "score": 0.07046709209680557}, {"label": "Haircut", "score": 0.05863256752490997}]......
```

文件中是测试集中各个视频的名称以及前五项类别的概率。



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_dj/ucf101_01.json /home/sunzheng/Video_Classification/data_dj/results/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1准确率：

```python
load ground truth
number of ground truth: 304
load result
number of result: 304
calculate top-1 accuracy
top-1 accuracy: 0.8223684210526315
```





6.调用模型识别单个视频是否为打架视频

这一步和5中不一样，5中是调用了模型去测试所有的测试集数据，有时候需要调用模型去识别单个视频，如展示模型的性能的时候，和repo （https://github.com/fourierer/Learn_GhostNet_pytorch） 中一样，重点部分在于预处理部分。

main.py脚本主要包括如下函数：

（1）get_opt()函数，获得opt；

（2）get_normalize_method()函数，正则化方法；

（3）get_train_utils()函数，预处理以及加载训练集数据；

（4）get_val_utils()函数，预处理以及加载验证集数据；

（5）get_inference_utils()函数，5中评测模型涉及到的函数，只不过是测了验证集中所有的视频，需要使用这个函数来测试单个视频；

（3）-（5）中的函数非常类似，实际上是将数据的预处理和加载分为了两个阶段：

1）预处理阶段：get_xxx_data（如get_train_data,get_val_data,get_inference_data）将数据预处理，类似于图像分类中的ImageFolder函数。很明显这个函数是作者自己写的为了特定的视频数据读取，而不是直接使用ImageFolder函数；

2）加载阶段：使用torch.utils.data.DataLoader将数据按照batch_size打包；

（6）save_checkpoint()函数，存储模型为tar包；

（7）main_worker()函数，训练代码。这个训练代码和图像分类训练代码非常类似：

1）先设置分布式训练方式，如单机多卡，多机多卡；

2）数据加载，加载训练集数据和验证集数据；

3）使用加载的训练集，验证集数据进行训练；

4）如果是5中的模型评测，则不需要步骤3）,4），直接加载推理（inference）数据集进行评测；

下面给出调用模型来识别单个视频的脚本(eval_pytorch_model_video.py)（利用main.py函数中的inference部分的代码）：

```python
import torch
import torchvision
from torchvision import transforms
from model import generate_model_resnet
from PIL import Image
import time
import os
import math
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose

device = torch.device('cuda')

'''
def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


# 数据预处理

spatial_transform = transforms.Compose([
    transforms.Resize(112),  # 改变图像大小，作为112*112的正方形
    #transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形>转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.4345, 0.4051, 0.3775],
                         std=[0.2768, 0.2713, 0.2737])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])
'''
inference_crop = 'center'
mean = [0.4345, 0.4051, 0.3775]
std = [0.2768, 0.2713, 0.2737]
no_mean_norm = False
no_std_norm = False
sample_size = 112
value_scale = 1
input_type = 'rgb'
sample_t_stride = 1
sample_duration = 16
inference_stride = 16

#normalize = get_normalize_method(mean, std, no_mean_norm, no_std_norm)
normalize = Normalize(mean, std)
spatial_transform = [Resize(sample_size)]
if inference_crop == 'center':
    spatial_transform.append(CenterCrop(sample_size))
if input_type == 'flow':
    spatial_transform.append(PickFirstChannels(n=2))
spatial_transform.append(ToTensor())
spatial_transform.extend([ScaleValue(value_scale), normalize])
spatial_transform = Compose(spatial_transform)

temporal_transform = []
if sample_t_stride > 1:
    temporal_transform.append(TemporalSubsampling(sample_t_stride))
temporal_transform.append(SlidingWindow(sample_duration, inference_stride))
temporal_transform = TemporalCompose(temporal_transform)



# 加载模型
#print('load model begin!')
model = generate_model_resnet(1) # 生成resnet模型
#model = torch.load('./save_200.pth')
checkpoint = torch.load('./save_200.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
#print(model)
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
#print('load model done!')


count = 0
# 测试单个视频
# fight
#img_path = './fight/angleview_p08p09_fight_a1/'
#img_path = './fight/EuRuxPSjgn8/'
#img_path = './fight/WGxSNBg_tl0/'
img_path = './fight/IQHsCcud-zE/'

# non-fight
#img_path = './non-fight/zzMeCV_eK3c_000002_000012/'
#img_path = './non-fight/zYYxrt0602w_000211_000221/'
#img_path = './non-fight/Zp-S9G02idU_000029_000039/'
#img_path = './non-fight/ZpbuVxloNpQ_000011_000021/'
#img_path = './non-fight/zFFfWSpwlok/'
#img_path = './non-fight/ZcI5Ht9e4QU/'


for img_name in os.listdir(img_path):
    count += 1
    #print(img_name)
    img = Image.open(img_path + img_name)
    img = spatial_transform(img)
   # print(type(img))
    #print(img.shape)
    img = img.unsqueeze(0)
    img = img.unsqueeze(4)
    #print(img.shape)
    if count==1:
        video = img
    else:
        video = torch.cat([video, img], dim=4)

#video = torch.Tensor(1, 3, 224, 224, 30) #如果只是测试时间，直接初始化一个Tensor即可
#video = temporal_transform(video)
#print(type(video))
video = video.permute(0, 1, 4, 2, 3)
#print(video.shape)


time_start = time.time()
video_ = video.to(device)
outputs = model(video_)
time_end = time.time()
time_c = time_end - time_start
_, predicted = torch.max(outputs,1)
threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
#print('model output:' + str(outputs))
if predicted==torch.tensor([0], device='cuda:0'):
    print('this video maybe: fight')
else:
    print('this video maybe: non-fight')
print('prob of fight:' + str(threshold))
print('time cost:', time_c, 's')
'''
# 批量测试数据集中的样本
N = 0 # 当前视频索引
acc_N = 0 # 分类正确的视频数
video_path = '/home/sunzheng/Video_Classification/data_dj/dj_videos/jpg/val/fight/'
for videos in os.listdir(video_path):
    #print(video)
    N += 1
    img_path = video_path + videos + '/'
    print(img_path)
    count = 0 # 当前视频的图片索引
    for img_name in os.listdir(img_path):
        count += 1
        #print(img_name)
        img = Image.open(img_path + img_name)
        img = spatial_transform(img)
        #print(type(img))
        #print(img.shape)
        img = img.unsqueeze(0)
        img = img.unsqueeze(4)
        #print(img.shape)
        if count==1:
            video = img
        else:
            video = torch.cat([video, img], dim=4)

    #video = torch.Tensor(1, 3, 224, 224, 30) #如果只是测试时间，直接初始化一个Tensor即可
    #video = temporal_transform(video)
    #print(type(video))
    video = video.permute(0, 1, 4, 2, 3)
    #print(video.shape)

    video = video.to(device)
    time_start = time.time()
    #video_= video.to(device)
    outputs = model(video)
    time_end = time.time()
    time_c = time_end - time_start
    outputs=outputs.cpu()
    _, predicted = torch.max(outputs,1)
    threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
    if threshold<0.5:
        acc_N += 1
    print('the video number:' + str(N))
    print('model output:' + str(outputs))
    print('this video maybe:' + str(predicted))
    print('prob of fight:' + str(threshold))
    print('time cost:', time_c, 's')
print('acc_N:' + str(acc_N))
print('total acc:' + str(acc_N/N))
'''
```

