# def read_first_n_lines(file_path, n):
#     with open(file_path, 'r') as file:
#         lines = []
#         for _ in range(n):
#             line = file.readline()
#             if not line:
#                 break
#             lines.append(line.strip())
#     return lines

# # 示例
# file_path = '/mnt/afs/lijiayi1/code/TTS/GPT-SoVits/data/wenetspeech_slicer_opt.list'
# n = 5
# first_n_lines = read_first_n_lines(file_path, n)
# print(first_n_lines)


import subprocess

def read_last_n_lines(file_path, n):
    result = subprocess.run(['tail', '-n', str(n), file_path], capture_output=True, text=True)
    return result.stdout.strip().split('\n')

# 示例
file_path = '/mnt/afs/lijiayi1/code/TTS/GPT-SoVits/data/wenetspeech_slicer_opt.list'
n = 5
last_n_lines = read_last_n_lines(file_path, n)
print(last_n_lines)