import os
import argparse
from get_phonemes_mfa_nopunc_test import get_phonemes_mfa_nopunc


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

def main(data_dir="../../data/", log_dir="logs/", name="dolly"):
    params = {
        "input_txt_path": os.path.join(data_dir),
        "save_path": f"{log_dir}/{name}",
        "input_wav_path": os.path.join(data_dir),
        "RANK": RANK,
        "WORLD_SIZE": WORLD_SIZE,
    }
    # get_phonemes(**params)
    
    # # mfa
    get_phonemes_mfa_nopunc(**params)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="/mnt/afs/chenyun/TTSFlow/dataset/ref.list", help="Directory to save data")
    parser.add_argument("--log_dir", type=str, default="/mnt/afs/chenyun/TTSFlow/dataset", help="Directory to save logs")
    parser.add_argument("--name", type=str, default="ref_wav", help="Name of the logs")
    
    args = parser.parse_args()

    main(args.data_dir, args.log_dir, args.name,)
    