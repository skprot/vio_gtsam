import torch
from torch import nn
import torch.nn.functional as nnfunc
from .unet_blocks import UNetConvBlock, UNetUpBlock


class Net(nn.Module):
    def __init__(self, weights='', in_channels=3, n_classes=1, depth=5, wf=4, padding=True, batch_norm=True,
                 up_mode='upconv'):
        super(Net, self).__init__()
        torch.set_num_threads(4)
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.activ = nn.Sigmoid()
        if len(weights) != 0:
            checkpoint = torch.load(map_location='gpu', f=weights)
            self.load_state_dict(checkpoint['state_dict'])
            self = nn.DataParallel(self)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = nnfunc.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.activ(self.last(x))

    def eval_predict(self, img):
        self.eval()
        with torch.no_grad():
            res = self.forward(img)
            return (res.cpu()).squeeze().numpy()
