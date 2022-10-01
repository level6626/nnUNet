#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import torch
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss


class TopKLoss(RobustCrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target, mask = False):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        # pdb.set_trace()
        if (mask):
            topk_mask = torch.zeros(num_voxels)
            raw_shape = res.shape
            res, topk_index = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
            topk_mask[topk_index] = 1
            topk_mask = topk_mask.reshape(raw_shape)
            return (res.mean(), topk_mask.numpy())
        else:
            res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
            return res.mean()
