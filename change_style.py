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


