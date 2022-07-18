import torch
from torch.utils.data.dataloader import default_collate

from evaluation.swap_visualization_delta_evaluator import SwapVisualizationDeltaEvaluator
from evaluation.swap_visualization_interested_evaluator import eval_list
from util.facescape_bs import FaceScapeBlendshape


class SwapVisualizationDeltaInterestedEvaluator(SwapVisualizationDeltaEvaluator):
    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.eval_iter = iter(eval_list)

    def gather_images(self, dataset):
        all_images = []
        num_images_to_gather = max(self.opt.swap_num_columns, self.opt.num_gpus)
        exhausted = False
        underlying_dataset = dataset.underlying_dataset
        while len(all_images) < num_images_to_gather:
            while True:
                try:
                    id_name, exp_name = next(self.eval_iter)
                    break
                except StopIteration:
                    self.eval_iter = iter(eval_list)
            data = default_collate(
                [underlying_dataset[underlying_dataset.get_index_by_id_name_exp_name(id_name, exp_name)]])
            label = [FaceScapeBlendshape.get_bs_concat(data['exp_name_A'][:1])]
            if self.opt.concat_age:
                label.append(self.facescape_age.get_age_concat(data['id_name_A'][:1]))
            label = torch.cat(label, dim=1).to(data['real_A'].device)
            all_images.append((data["real_A"][:1], label))
            if len(all_images) >= num_images_to_gather:
                break
        if len(all_images) == 0:
            return None, True
        return all_images, exhausted
