from torch.utils.data.dataloader import default_collate

from evaluation.swap_visualization_evaluator import SwapVisualizationEvaluator

eval_list = [
    ('490', '10_dimpler'), ('360', '2_smile'), ('20', '19_brow_raiser'), ('440', '5_jaw_left'),
    ('350', '9_mouth_right'), ('370', '12_lip_puckerer'), ('50', '15_lip_roll'), ('290', '18_eye_closed'),
    ('350', '8_mouth_left'), ('300', '3_mouth_stretch'), ('420', '19_brow_raiser'), ('320', '20_brow_lower'),
    ('430', '4_anger'), ('440', '17_cheek_blowing'), ('450', '19_brow_raiser'), ('510', '6_jaw_right'),
]


class SwapVisualizationInterestedEvaluator(SwapVisualizationEvaluator):
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
            all_images.append(data["real_A"][:1])
            if len(all_images) >= num_images_to_gather:
                break
        if len(all_images) == 0:
            return None, None, True
        return all_images, exhausted
