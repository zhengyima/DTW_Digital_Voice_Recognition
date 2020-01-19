# DTW_Digital_Voice_Recognition
基于DTW与MFCC特征提取进行数字0-9的语音识别，DTW，MFCC，语音识别，端点检测，中英数据，Digital Voice Recognition。


## Preinstallation

以下命令主要使用Anaconda安装环境，当然也可以更换为pip。

```
 conda create -n dtw -c anaconda python=3.6 numpy tqdm pyaudio scipy #也可以使用pip
 conda activate dtw
 pip install python_speech_features
```


## Launch the script
```
  git clone https://github.com/zhengyima/DTW_Digital_Voice_Recognition.git DTW_DVR
  cd DTW_DVR
  mkdir ProcessedData # 创建端点处理后数据目录
  python endpointDetection_RecordedVoice.py # 端点检测，默认使用英文数据
  python VoiceRecog.py 
  
```
