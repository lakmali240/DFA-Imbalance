from argparse import ArgumentParser

from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection, JaccardIndex, F1Score, Dice
from network.sam_network import PromptSAM, PromptSAMLateFusion
import torch


def get_metrics(cfg):
    num_classes = cfg.dataset.num_classes + 1 # Note that we have an extra class
    # if cfg.dataset.ignored_classes_metric is not None:
    #     ignore_index = [0, cfg.dataset.ignored_classes_metric]
    # else:
    ignore_index = 0
    metrics = MetricCollection({
        "IOU_Jaccard_Bal": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass'),
        "IOU_Jaccard": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass',
                                        average="micro"),
        "F1": F1Score(num_classes=num_classes, ignore_index=ignore_index, task='multiclass', average="micro"),
        "Dice": Dice(num_classes=num_classes, ignore_index=ignore_index, average="micro"),
        "Dice_Bal": Dice(num_classes=num_classes, ignore_index=ignore_index, average="macro"),
    })
    return metrics

def get_model(cfg):
    if cfg.model.extra_encoder is not None:
        print("Using %s as an extra encoder" % cfg.model.extra_encoder)
        neck = True if cfg.model.extra_type == 'plus' else False
        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)
        else:
            raise NotImplementedError
    else:
        extra_encoder = None
    if cfg.model.extra_type in ['plus']:
        MODEL = PromptSAM
    elif cfg.model.extra_type in ['fusion']:
        MODEL = PromptSAMLateFusion
    else:
        raise NotImplementedError

    model = MODEL(
        model_type = cfg.model.type,
        checkpoint = cfg.model.checkpoint,
        prompt_dim = cfg.model.prompt_dim,
        num_classes = cfg.dataset.num_classes,
        extra_encoder = extra_encoder,
        freeze_image_encoder = cfg.model.freeze.image_encoder,
        freeze_prompt_encoder = cfg.model.freeze.prompt_encoder,
        freeze_mask_decoder = cfg.model.freeze.mask_decoder,
        mask_HW = cfg.dataset.image_hw,
        feature_input = cfg.dataset.feature_input,
        prompt_decoder = cfg.model.prompt_decoder,
        dense_prompt_decoder=cfg.model.dense_prompt_decoder,
        no_sam=cfg.model.no_sam if "no_sam" in cfg.model else None
    )
    return model


parser = ArgumentParser()
parser.add_argument("--config", default='configs.BCSS', type=str, help="config file path (default: None)")
parser.add_argument('--devices', type=lambda s: [int(item) for item in s.split(',')], default=[0])
parser.add_argument('--project', type=str, default="c17")
parser.add_argument('--name', type=str, default="test_sam_prompt")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

module = __import__(args.config, globals(), locals(), ['cfg'])
cfg = module.cfg

cfg["project"] = args.project
cfg["devices"] = args.devices
cfg["name"] = args.name
cfg["seed"] = args.seed

seed_everything(cfg["seed"])
print(cfg)
# main(cfg)

metrics_calculator = get_metrics(cfg=cfg)

sam_model = get_model(cfg)
ckpt = torch.load(
    'model.ckpt', map_location='cuda:0'
)

updated_state_dict = {k[6:]: v for k, v in ckpt['state_dict'].items() if k[6:] in sam_model.state_dict()}
sam_model.load_state_dict(updated_state_dict)
sam_model.eval()

import cv2 as cv
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    def __init__(self):
        dataset = 'BCSS'
        mode = 'test'
        with open(f'../datasets/{dataset}/{mode}_files.txt', 'r') as f:
            self.img_paths = f.read().splitlines()

        self.dataset = dataset
        self.transform = A.Compose(
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform]
            + [ToTensorV2()], p=1)

        import pandas as pd
        import numpy as np

        df = pd.read_csv('/mnt/Xsky/szy/Former/SAMPath/dataset_cfg/BCSS_cv.csv', header=0)
        df = df[df['fold'] < 0]
        self.img_paths = np.asarray(df.iloc[:, 0])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        index = index % len(self)
        # img_path = '../' + self.img_paths[index]
        img_path = f'/mnt/Xsky/szy/Former/datasets/merged_dataset/img/{self.img_paths[index]}'

        image = cv.imread(img_path + '.jpg')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(img_path.replace('img', 'mask') + '.png', cv.IMREAD_GRAYSCALE)

        ret = self.transform(image=image, mask=mask)
        image, mask = ret["image"], ret["mask"]

        return image, mask.long()


from mmengine.config import Config

cfg = Config.fromfile('../config/BCSS.py')

from torch.utils.data import DataLoader

test_dataset = ImageMaskDataset()
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.data.batch_size_per_gpu,
    shuffle=False,
    num_workers=cfg.data.num_workers,
    drop_last=False
)

device = 'cuda:0'
metrics_calculator = metrics_calculator.to(device)
import sys

from torchmetrics import MetricCollection, JaccardIndex, F1Score, ClasswiseWrapper

ignore_index = 0
num_classes = 6
epoch_iterator = tqdm.tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                           dynamic_ncols=True)
epoch = 0
sam_model.to(device)

for data_iter_step, (images, true_masks) in enumerate(epoch_iterator):
    epoch_iterator.set_description(
        "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

    images = images.to(device)
    true_masks = true_masks.to(device)

    ignored_masks = torch.eq(true_masks, 0).long()

    pred_masks = sam_model(images)[0]
    pred_masks = torch.stack(pred_masks, dim=0)

    pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
    pred_masks = pred_masks * (1 - ignored_masks)

    metrics_calculator.update(pred_masks, true_masks)

print(metrics_calculator.compute())