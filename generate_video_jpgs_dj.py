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
        #print(class_dir_paths) #  输出各个视频类别的路径列表，比如[PosixPath('/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight'), PosixPath('/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/non-fight')]
        test_set_video_path = args.dir_path / 'test'
        #print(test_set_video_path) # /home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/test，加了个test类别
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)
        #print(class_dir_paths) # 视频类别文件夹中没有test类别，输出和上面一样

        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in class_dir_paths) # class_dir_path是某一个类别的视频文件路径，如PosixPath('/home/sunzheng/Video_Classification/data_dj/Fight-dataset-2020/videos/val/fight')


