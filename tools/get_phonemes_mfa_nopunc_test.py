import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import traceback
import torch
from text.cleaner_duration_nopunc import clean_text
from tqdm import tqdm


def process(data, device):
    """
    处理输入数据，获取音素序列及BERT特征。

    Args:
        data (list): 输入数据列表，每个元素为[wav_name, text, language]
        save_dir (str): BERT特征保存目录
        tokenizer (BertTokenizer): BERT分词器对象
        bert_model (BertModel): BERT模型对象
        device (str): 设备类型（'cuda'或'cpu'）

    Returns:
        list: 处理结果列表，每个元素为[wav_name, 音素序列, word2ph, norm_text]
    """
    res = []
    for full_name, text, lan in tqdm(data):
        try:
            name = os.path.basename(full_name)
            # 清理文本并获取音素序列、单词到音素的映射以及规范化后的文本
            pinyin, phones, word2ph, norm_text = clean_text(
                text.replace("%", "-").replace("￥", ",").replace("^", ","), lan
            )               
            phones = " ".join(phones)
            if pinyin:
                pinyin = " ".join(pinyin)
            else:
                pinyin = None
            # pinyin, phones, norm_text = "", "", ""
            res.append([full_name, name, pinyin, phones, norm_text])
        except:
            print(full_name, name, text, traceback.format_exc())

    return res

def get_phonemes_mfa_nopunc(input_txt_path: str, 
                 save_path: str, 
                 is_half: bool=False,
                 RANK: int=0,
                 WORLD_SIZE: int=1,
                 **kwargs) -> None:
    """
    从输入文本文件中获取音素序列和BERT特征。

    Args:
        input_txt_path (str): 输入文本文件路径
        save_path (str): 保存结果的路径
        bert_pretrained_dir (str, optional): BERT预训练模型路径. Defaults to 'pretrained_models/chinese-roberta-wwm-ext-large'.
        is_half (bool, optional): 是否使用半精度（FP16）模式. Defaults to False.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    todo = []
    with open(input_txt_path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
        data_size_per_rank = len(lines) // WORLD_SIZE + 1
        data_list = lines[RANK * data_size_per_rank : (RANK + 1) * data_size_per_rank]
        for line in data_list:
            try:
                wav_name, spk_name, language, text = line.split("|")
                todo.append([wav_name, text, language.lower()])
            except:
                print(line, traceback.format_exc())

    res = process(todo, device)
    
    mfa_path = f'{save_path}'
    os.makedirs(mfa_path, exist_ok=True)
    
    for full_name, name, pinyin, phones, norm_text in tqdm(res):
        # if not os.path.exists(f"{mfa_path}/{name}.txt"):
        #     with open(f"{mfa_path}/{name}.txt", "w", encoding="utf8") as f:
        #         f.write(norm_text + "\n")
        name = name.split('.')[0]
        os.makedirs(f"{mfa_path}/{name}", exist_ok=True)
        if not os.path.exists(f"{mfa_path}/{name}/{name}.lab"):
            with open(f"{mfa_path}/{name}/{name}.lab", "w", encoding="utf8") as f:
                if pinyin:
                    f.write(pinyin + "\n")
                else:
                    f.write(norm_text + "\n")
        if not os.path.exists(f"{mfa_path}/{name}/{name}.wav"):
            # source = full_name
            # if os.path.exists(source):
            #     link = f"{mfa_path}/{name}.wav"
            #     command = ["ln", "-s", source, link]
            #     subprocess.run(command, check=True)
            # else:
            # if full_name.startswith("s3://"):
            with open(f"{mfa_path}/{name}/{name}.wav", "w", encoding="utf8") as f:
                f.write(full_name)
            # else:
            #     source = full_name
            #     link = f"{mfa_path}/{name}.wav"
            #     command = ["cp", source, link]
            #     subprocess.run(command, check=True)
            

    # opt = []
    # for full_name, name, pinyin, phones, norm_text in res:
    #     opt.append("%s\t%s" % (name, pinyin))

    # with open(f"{save_path}/text2pinyin_mfa_{RANK}.txt", "w", encoding="utf8") as f:
    #     f.write("\n".join(opt) + "\n")
        
    # opt = []
    # for full_name, name, pinyin, phones, norm_text in res:
    #     opt.append("%s\t%s\t%s\t%s" % (name, pinyin, phones, norm_text))
        
    # with open(f"{save_path}/text2phonemes_mfa_{RANK}.txt", "w", encoding="utf8") as f:
    #     f.write("\n".join(opt) + "\n")

    print("文本转音素mfa已完成！")