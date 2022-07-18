from .tmux_launcher import Options, TmuxLauncher

blender_path = 'blender'
ffmpeg_path = 'ffmpeg'


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="./predef/sample_dataset/",  # just a small sample dataset
            dataset_mode="seg_dpmap_both_pair",
            input_nc=2,
            num_gpus=2, batch_size=16,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="none",
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
            optimizer='swapping_autoencoder_recon_cond',
            netTExp='SpatialOnlyExp',
            lambda_PatchGAN=0,
            lambda_patch_R1=0,
            seg_type='dt',
            lpips_channel_index=1,
            netTExp_num_layers=4,
            remove_small=False,
            netD='StyleGAN2MultiHead',
            model='swapping_autoencoder_delta_multihead_seg',
            concat_age=True,
            netTAge='SpatialGlobalAge',
            netTAge_num_layers=4,
            netTExp_feature_dim=512,
            netTAge_feature_dim=512,
            netE_num_downsampling_sp=5,
            global_code_ch=64,
            netG_scale_capacity=0.25,
            netD_scale_capacity=0.25,
            netE_scale_capacity=0.25,
            lambda_LPIPS=0,
            lambda_LPIPS_mix=0,
            lambda_LPIPS_cyc=0,
            lambda_FM=20,
            seg_mean=1,
            seg_std=10,
            dir_A_name='mask_simp_v6',
            netE_nc_max=512,
            lambda_L1_seg=50,
        ),

        opts = [
            opt.specify(
                name="SEMM",
            ),
        ]
        return opts

    def train_options(self):
        common_options = self.options()
        res = [opt.specify(
            continue_train=True,
            evaluation_metrics=("swap_visualization_delta_interested,"
                                "swap_visualization_delta_age_only_interested"),
            evaluation_freq=50000,
        ) for opt in common_options]

        return res

    def test_options_fid(self):
        return []

    def test_options(self):
        common_options = self.options()

        res = [
            opt.tag("swap_expression").specify(
                num_gpus=1,
                batch_size=1,
                evaluation_metrics=("swap_visualization_delta_interested"
                                    ",swap_visualization_delta_exp_only_interested"
                                    ",swap_visualization_delta_age_only_interested"),
            ) for opt in common_options
        ]
        return res

    def exp_age_edit_options(self):
        common_options = self.options()

        res = [
            opt.tag("exp_age_edit").specify(
                num_gpus=1,
                batch_size=1,
                evaluation_metrics="scan_image_editing",
                test_exp=True,
                test_age=True,
                root_path='./demo/output',
                blender_path=blender_path,
            ) for opt in common_options
        ]
        return res

    def bs_anime_options(self):
        common_options = self.options()

        res = [
            opt.tag('exp_inter').specify(
                num_gpus=1,
                batch_size=1,
                evaluation_metrics="scan_image_bs_anime",
                file_paths='./demo/output/18',
                blender_path=blender_path,
                ffmpeg_path=ffmpeg_path,
            ) for opt in common_options
        ]
        return res

    def exp_inter_options(self):
        common_options = self.options()

        res = [
            opt.tag('exp_inter').specify(
                num_gpus=1,
                batch_size=1,
                evaluation_metrics="scan_image_exp_inter",
                file_paths='./demo/output/18',
                target_exps='19_brow_raiser',
                blender_path=blender_path,
                ffmpeg_path=ffmpeg_path,
            ) for opt in common_options
        ]
        return res

    def age_progression_options(self):
        common_options = self.options()

        res = [
            opt.tag('exp_inter').specify(
                num_gpus=1,
                batch_size=1,
                evaluation_metrics="scan_image_age_progression",
                file_paths='./demo/output/18',
                n_frame=30,
                blender_path=blender_path,
                ffmpeg_path=ffmpeg_path,
            ) for opt in common_options
        ]
        return res
