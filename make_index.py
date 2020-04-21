# 此脚本用于生成打架数据集汇的trainlist01.txt和testlist01.txt

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







