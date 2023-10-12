import torch
from torchvision.transforms import transforms, functional
from torchvision.utils import save_image

import copy

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

class Synthesizer:

    def __init__(self):
        pass

    def make_backdoor_batch(self, batch, test=True, attack=True):
        """for test, no training"""
        attack_portion = len(batch)
        backdoored_batch = copy.deepcopy(batch).cuda()
        self.apply_backdoor(backdoored_batch, attack_portion)

        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)

        return

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented



class PatternSynthesizer(Synthesizer):
    pattern_tensor = {
		'default': torch.tensor([
			[1., 0., 1.],
			[-10., 1., -10.],
			[-10., -10., 0.],
			[-10., 1., -10.],
			[1., 0., 1.]]),
		'tri1_2x2': torch.tensor([
			[0., 1.],
			[1., 0.]
		]),
		'tri1_3x3': torch.tensor([
			# [0., 1., 0.],
			# [1., 0., 1.],
			# [0., 1., 0.]
            # for F3BA
            [ 1.0701,  2.1036, -1.0729],                        
            [-1.1868,  0.6721,  0.5508],                         
            [ 2.1612, -1.3786,  0.5577],
		]),
		'tri1_5x5': torch.tensor([
			[0., 1., 0., 1., 0.],
			[1., 0., 1., 0., 1.],
			[0., 1., 0., 1., 0.],
			[1., 0., 1., 0., 1.],
			[0., 1., 0., 1., 0.],
		]),
        'tri2_1x4_bg': torch.tensor([
            [1., 0., 1., 0., -10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, 1., 0., 1., 0.],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [1., 0., 1., 0., -10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, 1., 0., 1., 0.]
	    ]),
	    'tri2_1x4': torch.tensor([
            [-10.,-10,-10.,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, -10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,1., 0., 1., 0.,-10.,-10,1., 0., 1., 0.,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,1., 0., 1., 0.,-10,-10, 1., 0., 1., 0.]
	    ]),
	    'tri2_2x4': torch.tensor([
            [-10.,-10,-10.,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10, -10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,0., 1., 0., 1.,-10.,-10,0., 1., 0., 1.,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,1., 0., 1., 0.,-10.,-10,1., 0., 1., 0.,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,],
            [-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,-10.,-10,0., 1., 0., 1.,-10.,-10,0., 1., 0., 1.],
            [-10.,-10,-10.,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,1., 0., 1., 0.,-10,-10, 1., 0., 1., 0.]
	    ]),

	}
    # pattern_tensor: torch.Tensor
    "Just some random 2D pattern."

    x_top = 0
    "X coordinate to put the backdoor into."
    y_top = 0
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, trigger_name: str):
        super().__init__()
        self.make_pattern(self.pattern_tensor[trigger_name], self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        input_shape = [3, 32, 32]
        full_image = torch.zeros(input_shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot > input_shape[1] or \
                y_bot > input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).cuda()
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
        self.pattern = normalize(full_image).cuda()

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        batch[:attack_portion] = (1 - mask) * \
                                        batch[:attack_portion] + \
                                        mask * pattern

        return

    def get_pattern(self):
        return self.pattern, self.mask
