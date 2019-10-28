# -*- coding: utf-8 -*-
import wave
import os
import numpy as np
from struct import unpack
import pyaudio
from endpointDetection import EndPointDetect
# from mfcc import getMFCC
import scipy.io.wavfile as wav
from python_speech_features import *

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--maxnum", help="max digit",type = int, default = 9)
parser.add_argument("-c", "--count", help="examples for each digit",type = int, default = 5)


parser.add_argument("-f", "--framerate", help="max digit",type = int, default = 16000)
parser.add_argument("-ch", "--channels", help="采样频率 8000 or 16000",type = int, default = 1)
parser.add_argument("-s", "--sampwidth", help="max digit",type = int, default = 2)
parser.add_argument("-chunk", "--CHUNK", help="录音的块大小",type = int, default = 1024)
parser.add_argument("-rate", "--RATE", help="采样频率 8000 or 16000",type = int, default = 16000)
parser.add_argument("-resec", "--RECORD_SECONDS", help="录音时长 单位 秒(s)",type = float, default = 2.5)
parser.add_argument("-d", "--datapath", help="dir of data",type = str, default = "./ProcessedData/")

args = parser.parse_args()


maxnum = args.maxnum # max digit
count = args.count # examples for each digit
# 存储成 wav 文件的参数
framerate = args.framerate  # 采样频率 8000 or 16000
channels = args.channels       # 声道数
sampwidth = args.sampwidth      # 采样字节 1 or 2
datapath = args.datapath

# 实时录音的参数
CHUNK = args.CHUNK        # 录音的块大小
RATE = args.RATE         # 采样频率 8000 or 16000
RECORD_SECONDS = args.RECORD_SECONDS # 录音时长 单位 秒(s)


test_count = count - 1
# 读取已经用 HTK 计算好的 MFCC 特征

def extract_MFCC(file):

    fs, audio = wav.read(file)
    wav_feature = mfcc(audio, samplerate=fs,numcep=13, winlen=0.025, winstep=0.01,nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature_mfcc = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    # print(feature_mfcc)
    # print(feature_mfcc.shape)
    return feature_mfcc

def getMFCC():
    MFCC = []
    for i in range(maxnum+1) :
        MFCC_rows = []
        for j in range(count) :
            file_name = datapath + str(i) + "-" + str(j + 1) + ".wav"
            feature = extract_MFCC(file_name)
            MFCC_rows.append(feature)
        MFCC.append(MFCC_rows)
    return MFCC

# 取出其中的模板命令的 MFCC 特征 
def getMFCCModels(MFCC) :
    MFCC_models = []
    for i in range(len(MFCC)) :
        MFCC_models.append(MFCC[i][0])
    return MFCC_models

# 取出其中的待分类语音的 MFCC 特征
def getMFCCUndetermined(MFCC) :
    MFCC_undetermined = []
    for i in range(len(MFCC)) :
        for j in range(1, len(MFCC[i])) :
            MFCC_undetermined.append(MFCC[i][j])
    return MFCC_undetermined

# DTW 算法...
def dtw(M1, M2) :
    # 初始化数组 大小为 M1 * M2
    M1_len = len(M1)
    M2_len = len(M2)
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]
    
    # 初始化 dis 数组
    dis = []
    for i in range(M1_len) :
        dis_row = []
        for j in range(M2_len) :
            dis_row.append(distance(M1[i], M2[j]))
        dis.append(dis_row)
    
    # 初始化 cost 的第 0 行和第 0 列
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len) :
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len) :
        cost[0][j] = cost[0][j - 1] + dis[0][j]
    
    # 开始动态规划
    for i in range(1, M1_len) :
        for j in range(1, M2_len) :
            cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1, \
                            cost[i- 1][j - 1] + dis[i][j] * 2, \
                            cost[i][j - 1] + dis[i][j] * 1)
    return cost[M1_len - 1][M2_len - 1]

# 两个维数相等的向量之间的距离
def distance(x1, x2) :
    sum = 0
    for i in range(len(x1)) :
        sum = sum + abs(x1[i] - x2[i])
    return sum

# 将语音文件存储成 wav 格式
def save_wave_file(filename, data):
    '''save the date to the wavfile'''
    wf = wave.open(filename,'wb')
    wf.setnchannels(channels)   # 声道
    wf.setsampwidth(sampwidth)  # 采样字节 1 or 2
    wf.setframerate(framerate)  # 采样频率 8000 or 16000
    wf.writeframes(b"".join(data))
    wf.close()



# 存储所有语音文件的 MFCC 特征
# 读取已经用 HTK 计算好的 MFCC 特征
MFCC = getMFCC()

# 取出其中的模板命令的 MFCC 特征 
MFCC_models = getMFCCModels(MFCC)

# 取出其中的待分类语音的 MFCC 特征
MFCC_undetermined = getMFCCUndetermined(MFCC)

# 开始匹配
n = 0
for i in range(len(MFCC_undetermined)) :
    flag = 0
    
    min_dis = dtw(MFCC_undetermined[i], MFCC_models[0])
    for j in range(1, len(MFCC_models)) :
        # print(np.array(MFCC_undetermined[i]).shape)
        dis = dtw(MFCC_undetermined[i], MFCC_models[j])
        if dis < min_dis :
            min_dis = dis
            flag = j
    
    if i + 1 <= (flag + 1) * test_count and i + 1 >= flag * test_count :
        n = n + 1
    
    # print(str(i) + "\t" + str(flag) + "\n")
    print("%d-%d,pred:%d,true:%d"%(int(i/test_count),i%test_count,flag,int(i/test_count)))
print("acc:%f"%(n/(test_count*(maxnum+1))))
# 录音
pa = pyaudio.PyAudio()
stream = pa.open(format = pyaudio.paInt16, channels = 1, \
                   rate = framerate ,    input = True, \
                   frames_per_buffer = CHUNK)

print("start recording realtime...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    
print("recording finished!")

# 存储刚录制的语音文件
save_wave_file("./recordedVoice_before.wav", frames)

# 对刚录制的语音进行端点检测
f = wave.open("./recordedVoice_before.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes) 
wave_data = np.fromstring(str_data, dtype = np.short)
f.close()
end_point_detect = EndPointDetect(wave_data)

# 存储端点检测后的语音文件
N = end_point_detect.wave_data_detected
m = 0
print(N)
while m < len(N) :
    save_wave_file("./recordedVoice_after.wav", wave_data[N[m] * 256 : N[m+1] * 256])
    m = m + 2


MFCC_recorded = extract_MFCC("./recordedVoice_before.wav")

# 进行匹配
flag = 0
min_dis = dtw(MFCC_recorded, MFCC_models[0])
for j in range(1, len(MFCC_models)) :
    dis = dtw(MFCC_recorded, MFCC_models[j])
    if dis < min_dis :
        min_dis = dis
        flag = j
print( "\t" + str(flag) + "\n")