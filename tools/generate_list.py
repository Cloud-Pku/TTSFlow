import json

# 读取JSON文件
with open('/mnt/afs/chenyun/TTSFlow/dataset/ref.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据并生成所需的列表格式
output_lines = []
for item in data:
    ref_wav_path = item['ref_wav_path']
    up_name = item['up_name']
    prompt_text = item['prompt_text']
    output_line = f"/mnt/afs/chenyun/TTSFlow/ref_wav/{ref_wav_path}|filterd_audio|ZH|{prompt_text}"
    output_lines.append(output_line)

# 将结果写入文件
with open('/mnt/afs/chenyun/TTSFlow/dataset/ref.list', 'w', encoding='utf-8') as file:
    for line in output_lines:
        file.write(line + '\n')

