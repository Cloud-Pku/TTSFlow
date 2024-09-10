from text import chinese_mfa_nopunc, japanese, cleaned_text_to_sequence, symbols, english
from text.chinese_mfa_nopunc import TextTokenizer
from text.symbol_table import SymbolTable
import os
import LangSegment

current_file_path = os.path.dirname(__file__)

language_module_map = {"zh": chinese_mfa_nopunc, "ja": japanese, "en": english, "multi": chinese_mfa_nopunc}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

def make_lab(tt, wav):
    id = ".".join(wav.split('/')[-1].split('.')[:-1])
    folder = '/'.join(wav.split('/')[:-1])
    # Create lab files
    with open(f'{folder}/{id}.txt', 'r') as f:
        txt = f.read()

        with open(f'{folder}/{id}.lab', 'w') as f:
            f.write(' '.join(tt.tokenize(txt)))

def clean_text(text, language, infer=False):
    if(language not in language_module_map):
        language="en"
        text=" "
        pinyin=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    if language == "zh":
        language_module = language_module_map[language]
        norm_text = language_module.text_normalize(text)
        tt = TextTokenizer()
        pinyin = tt.tokenize(norm_text)
        phones = tt.tokenize_lty(pinyin)
        word2ph = None
    elif language == "en" or language == "ja":
        language_module = language_module_map[language]
        norm_text = language_module.text_normalize(text)
        phones = language_module.g2p(norm_text)
        pinyin = norm_text.split(" ")
        word2ph = None
    else:
        pinyin, phones, norm_text_all = [], [], ""
        LangSegment.setfilters(["zh","en"])
        langlist, textlist = [], []
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
        # print(textlist)
        # print(langlist)

        for i in range(len(textlist)):
            lang = langlist[i]
            if lang == "zh":
                language_module = language_module_map[lang]
                norm_text = language_module.text_normalize(textlist[i])
                tt = TextTokenizer()
                pinyin_one = tt.tokenize(norm_text)
                pinyin.extend(pinyin_one)
                phones.extend(tt.tokenize_lty(pinyin_one))
                word2ph = None
                norm_text_all += norm_text
            else:
                language_module = language_module_map[lang]
                norm_text = (language_module.text_normalize(textlist[i]))
                phones.extend(language_module.g2p(norm_text.strip()))
                pinyin.extend(norm_text.strip().split(" "))
                word2ph = None
                if infer:
                    norm_text_all += " ".join(language_module.g2p(norm_text.strip()))
                else:
                    norm_text_all += norm_text
        norm_text = norm_text_all
        
    # lty = tt.tokenize_lty(phones)

    # unique_tokens = SymbolTable.from_file(os.path.join(current_file_path,"unique_text_tokens.k2symbols")).symbols
    # for ph in phones:
    #     assert ph in unique_tokens
    return pinyin, phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
