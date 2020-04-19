import ffmpeg
import numpy
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
    total_duration = float(video_info['duration'])
    print('总帧数：' + str(total_frames))
    print('总时长：' + str(total_duration))

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



