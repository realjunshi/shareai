from pathlib import Path
import os

model_name = "train_max.npy"

DATA_ROOT = Path(os.environ.get('DATA', './data'))
ckpt_path = os.path.join(DATA_ROOT, model_name)

if __name__ == '__main__':
    # print(DATA_ROOT)
    print(ckpt_path)