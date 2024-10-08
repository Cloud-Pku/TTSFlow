from aoss_client.client import Client 
from tqdm import tqdm
import json

client = Client(conf_path="~/aoss.conf")
dir_list = []

# 读取目录列表
for name in client.list("s3://mmdata/audio/WenetSpeech/train/podcast/"):
    dir_list.append("s3://mmdata/audio/WenetSpeech/train/podcast/" + name)

for name in client.list("s3://mmdata/audio/WenetSpeech/train/youtube/"):
    dir_list.append("s3://mmdata/audio/WenetSpeech/train/youtube/" + name)

# 打开文件准备写入
with open('wav_list.json', 'w') as file:
    # 写入 JSON 文件的开头
    file.write('[\n')
    
    num = 0
    for dir in tqdm(dir_list):
        for name in client.list(dir):
            wav_path = dir + name
            # print(num)
            num += 1
            
            # 将文件路径写入 JSON 文件
            if num > 1:
                file.write(',\n')
            file.write(json.dumps(wav_path))
    
    # 写入 JSON 文件的结尾
    file.write('\n]')