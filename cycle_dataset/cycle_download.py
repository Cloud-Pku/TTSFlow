from aoss_client.client import Client
from torch.optim import radam 
from tqdm import tqdm
import json
import random
import shutil
import time
import os

file_num = 100
keep_num = 5

def keep_one_file(directory_path):
    # 列出目录中的所有文件
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    keep_filename = random.sample(files, keep_num)

    for filename in files:
        if filename not in keep_filename:
            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete file {file_path}. Reason: {e}")

client = Client(conf_path="~/aoss.conf")


with open('/mnt/afs/chenyun/TTSFlow/cycle_dataset/wav_list.json', 'r', encoding='utf-8') as file:
    wav_list = json.load(file)

print(len(wav_list))


while True:
    try:
        keep_one_file("/mnt/afs/chenyun/TTSFlow/cycle_dataset/tmp_dataset/")
        # os.makedirs("/mnt/afs/chenyun/TTSFlow/cycle_dataset/tmp_dataset/", exist_ok=True)
        tmp_wav = random.sample(wav_list, file_num-keep_num)
        for path in tmp_wav:
            value = client.get(path)
            file_name = path.split('/')[-1]
            with open(f"/mnt/afs/chenyun/TTSFlow/cycle_dataset/tmp_dataset/{file_name}", 'wb') as file:
                # 将 bytes 数据写入文件
                file.write(value)
        time.sleep(20)
    except Exception as e:
        print(e)


# value = client.get("s3://videodata/audio/vfhq/zw_1lF9qv_Q.webm")
# with open("zw_1lF9qv_Q.webm", 'wb') as file:
#     # 将 bytes 数据写入文件
#     file.write(value)

