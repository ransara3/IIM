"""Short CPU fine-tuning of IIM (VGG16_FPN) on JHU, for when no pretrained IIM checkpoint is available.

Backbone starts from torchvision's ImageNet VGG16-BN weights. Saves state dicts that test.py can load.
"""
import argparse
import os
import time

import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader

import misc.transforms as own_transforms
from datasets import basedataset
from datasets.setting.JHU import cfg_data
from misc.utils import adjust_learning_rate
from model.locator import Crowd_locator

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='VGG16_FPN')
parser.add_argument('--minutes', type=float, default=60, help='training time budget')
parser.add_argument('--phase1_minutes', type=float, default=40, help='time spent training only backbone + head')
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--crop_h', type=int, default=256)
parser.add_argument('--crop_w', type=int, default=512)
parser.add_argument('--lr1', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-5)
parser.add_argument('--save_every', type=int, default=50, help='iterations between checkpoints')
parser.add_argument('--out', default='./exp/cpu_JHU_VGG16_FPN/latest.pth')
parser.add_argument('--max_iters', type=int, default=0, help='stop early after N iterations (0 = use time budget)')
args = parser.parse_args()

torch.manual_seed(3035)
torch.set_num_threads(os.cpu_count() or 4)

crop = (args.crop_h, args.crop_w)
main_transform = own_transforms.Compose([
    own_transforms.ScaleByRateWithMin([0.8, 1.2], crop[1], crop[0]),
    own_transforms.RandomCrop(crop),
    own_transforms.RandomHorizontallyFlip(),
])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*cfg_data.MEAN_STD),
])
mask_transform = standard_transforms.Compose([standard_transforms.ToTensor()])

train_set = basedataset.Dataset(
    'JHU', 'train',
    main_transform=main_transform, img_transform=img_transform, mask_transform=mask_transform,
    list_file=[{'data_path': cfg_data.DATA_PATH, 'imgId_txt': cfg_data.TRAIN_LST, 'box_gt_txt': []}],
)
# The processed JHU folder stores masks in 'mask/', not the 'mask_30_60/' the dataset class assumes.
train_set.mask_path = [p.replace('/mask_30_60/', '/mask/') for p in train_set.mask_path]

loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

net = Crowd_locator(args.net, '0', pretrained=True)
optimizer = optim.Adam([
    {'params': net.Extractor.parameters(), 'lr': args.lr1, 'weight_decay': 1e-5},
    {'params': net.Binar.parameters(), 'lr': args.lr2},
])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
start = time.time()
deadline = start + args.minutes * 60
phase1_end = start + args.phase1_minutes * 60
step = 0
net.train()
done = False
while not done:
    for img, gt_map in loader:
        optimizer.zero_grad()
        if time.time() < phase1_end:
            # Phase 1: backbone + head only, since the threshold module is the costly part on CPU.
            _, pre_map = net.Extractor(img)
            head_loss = torch.nn.functional.mse_loss(pre_map, gt_map)
            binar_loss = torch.zeros(())
            head_loss.backward()
        else:
            net(img, gt_map)
            head_loss, binar_loss = net.loss
            (head_loss + binar_loss).backward()
        optimizer.step()
        step += 1
        elapsed = time.time() - start
        print(f'[it {step}][head {head_loss.item():.5f}][binar {binar_loss.item():.4f}]'
              f'[gt_max {gt_map.max().item():.1f}][{elapsed / step:.1f}s/it]', flush=True)
        if step % args.save_every == 0:
            torch.save(net.state_dict(), args.out)
        if time.time() > deadline or (args.max_iters and step >= args.max_iters):
            done = True
            break

torch.save(net.state_dict(), args.out)
print('saved', args.out, 'after', step, 'iterations')
