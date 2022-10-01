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


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import DC_and_topk_loss

import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

from os.path import exists, join
from os import listdir, makedirs
import numpy as np

class nnUNetTrainerV2_Loss_DiceTopK10(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 600
        self.loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
                                     {'k': 10})
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l, topk_mask = self.loss(output, target)
                ## save topk mask
                ## pdb.set_trace()
                basedir = "/root/autodl-tmp/TopK/ep500.1"
                makedirs(basedir, exist_ok=True)
                key1 = data_dict["keys"][0]
                key2 = data_dict["keys"][1]
                if (key1.find("001") != -1 or key2.find("001") != -1):
                    cnt1 = 0
                    cnt2 = 0
                    for f in listdir(basedir):
                        if (f.find(key1) != -1):
                            cnt1 += 1
                        if (f.find(key2) != -1):
                            cnt2 += 1
                    cnt1 /= 2
                    cnt2 /= 2
                    key1_fn = join(basedir, "%s.%d.npy" % (key1, cnt1))
                    key2_fn = join(basedir, "%s.%d.npy" % (key2, cnt2))
                    key1_target = join(basedir, "%s.%d.target.npy" % (key1, cnt1))
                    key2_target = join(basedir, "%s.%d.target.npy" % (key2, cnt2))
                    np.save(key1_fn, topk_mask[0, ...])
                    np.save(key2_fn, topk_mask[1, ...])
                    np.save(key1_target, target[0][0].detach().cpu().numpy())
                    np.save(key2_target, target[0][1].detach().cpu().numpy())


                

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
