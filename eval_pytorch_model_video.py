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








