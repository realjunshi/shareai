import math
import gc
import time
from pathlib import Path

import colossalai
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from titans.utils import barrier_context

import torch
import os

from model import LanguageModel
from model import generate
from data import SongDataset

DATA_ROOT = Path(os.environ.get('DATA', './data'))

torch.backends.cudnn.benchmark = True

torch.manual_seed(1337)
batch_size = 64

max_iters = 5000
eval_inteval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

best_val_loss = 1e9
# ---------------------

out_dir = "./"
model_name = 'ckpt.pt'
ckpt_path = os.path.join(out_dir, model_name)
print(ckpt_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--init_from', default='scratch', help='train from scratch or not')
    args = parser.parse_args()

    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    if args.init_from == 'scratch':
        # 从头初始化一个模型
        print("从头初始化一个模型：")
        model = LanguageModel()
    elif args.init_from == 'resume':
        print(f"从 {out_dir} 加载已经训练好的模型")
        model = LanguageModel()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))


    with barrier_context():
        train_dataset = SongDataset('train_max.npy')
        train_loader = get_dataloader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=gpc.config.BATCH_SIZE,
            pin_memory=True,
        )
    val_dataset = SongDataset('val_max.npy')
    val_loader = get_dataloader(
        dataset=val_dataset,
        add_sampler=False,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )
    # 误差函数
    criterion = torch.nn.CrossEntropyLoss()
    # 随机梯度下降
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
    )

    for epoch in range(gpc.config.NUM_EPOCHS):
        engine.train()

        train_loss = 0
        val_loss = 0
        # ================train begin===================
        train_dl = tqdm(train_dataloader)
        for source, target in train_dl:
            source = source.cuda()
            target = target.cuda()

            engine.zero_grad()
            output = engine(source)

            B, T, C = output.shape
            output = output.view(B * T, C)
            target = target.view(B * T)
            train_loss = engine.criterion(output, target)

            engine.backward(train_loss)
            engine.step()

            if train_loss < best_val_loss:
                torch.save(model.state_dict(), ckpt_path)

        lr_scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

        # ================train end===================

        # ================val begin===================
        engine.eval()
        val_losses = 0
        # index = 0
        for x_val, y_val in val_loader:
            # index += 1
            x_val = x_val.cuda()
            y_val = y_val.cuda()

            output = engine(x_val)
            B, T, C = output.shape
            output = output.view(B * T, C)
            y_val = y_val.view(B * T)
            val_loss = engine.criterion(output, y_val)

        gc.collect()
        torch.cuda.empty_cache()

            # val_losses = val_loss.item()
        # out = val_losses / index

        # ================val end===================
        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, val loss: {val_loss:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
            ranks=[0])

if __name__ == '__main__':
    main()
    # print(generate(model, "还不如不见"), 10)
