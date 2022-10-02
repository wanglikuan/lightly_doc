# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
#from lightly.loss import SwaVLoss
from swav_loss_ob_q import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes

from cifar_utils import CIFAR10Instance
import os

class SwaV(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=10) #512

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV(backbone)

device = "cuda" #if torch.cuda.is_available() else "cpu"
model.to(device)

# # we ignore object detection annotations by setting target_transform to return 0
# pascal_voc = torchvision.datasets.VOCDetection(
#     "datasets/pascal_voc", download=True, target_transform=lambda t: 0
# )
# dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# # or create a dataset from a folder containing images or videos:
# # dataset = LightlyDataset("path/to/folder")

#cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
cifar10 = CIFAR10Instance("datasets/cifar10", train=True, download=True, transform=None)
dataset = LightlyDataset.from_torch_dataset(cifar10)
dataset.transform = torchvision.transforms.ToTensor()
dataloader = torch.utils.data.DataLoader(
    dataset,        # use the same dataset as before
    batch_size=1,   # we can use batch size 1 for inference
    shuffle=False,  # don't shuffle your data during inference
)

Save_path = "D:\\0915-FedQ\\lightly-master\\Saved_model\\swav_client0"
resume_path = '%s/ep%s.t7' % (Save_path, 200)
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['net'])
model.to(device)
model.eval()

criterion = SwaVLoss(temperature=0.5, sinkhorn_epsilon=0.05)

with torch.no_grad():
    for idx, batch, label, fnames in enumerate(dataloader):
        multi_crop_features = [model(x.to(device)) for x in batch]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss, theq = criterion(high_resolution, low_resolution)
        print('label: ', label, '    q: ', theq)
        if idx == 99 :
            break

print('end get q')


# lr = 0.05

# criterion = SwaVLoss(temperature=0.5, sinkhorn_epsilon=0.05)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr) #0.001

# # path = os.path.abspath('.')
# # Save_path = os.path.join(path, "/Saved_model/swav_client0")
# Save_path = "D:\\0915-FedQ\\lightly-master\\Saved_model\\swav_client3"

# print("Starting Training")
# for epoch in range(201):
#     total_loss = 0
#     if epoch == 40:
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr * 0.1
#         print("change lr to {}".format( lr * 0.1 ))
#     elif epoch == 100:
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr * 0.01
#         print("change lr to {}".format( lr * 0.01 ))

#     # if epoch % 40 == 0:
#     #     if epoch >= 40:
#     #         for param_group in optimizer.param_groups:
#     #             param_group["lr"] = lr * (0.1 ** (epoch//40))  #i.e. 40,80,120,160,200
#     #         print("change lr to {}".format( lr * (0.1 ** (epoch//40)) ))

#     for batch, _, _ in dataloader:
#         #print(f"epoch: {epoch:>02}, 1 step in dataloader")
#         model.prototypes.normalize()
#         multi_crop_features = [model(x.to(device)) for x in batch]
#         high_resolution = multi_crop_features[:2]
#         low_resolution = multi_crop_features[2:]
#         loss = criterion(high_resolution, low_resolution)
#         total_loss += loss.detach()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     if epoch % 100 == 0:
#         print('Saving...')
#         state = {'net': model.state_dict(),
#                  'epoch': epoch,
#                  'opt': optimizer.state_dict(),}
#         if not os.path.isdir(Save_path):
#             os.mkdir(Save_path)
#         torch.save(state, '%s/ep%s.t7' % (Save_path, epoch))
#     avg_loss = total_loss / len(dataloader)
#     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
