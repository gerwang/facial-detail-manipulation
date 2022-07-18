import os


class BlenderRender:
    def __init__(self, blender_path='blender',
                 ffmpeg_path='ffmpeg',
                 env_path='./blender_scripts/HDR_112_River_Road_2_Env.hdr'):
        self.blender_path = blender_path
        self.env_path = env_path
        self.ffmpeg_path = ffmpeg_path

    def render_recon(self, file_path):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/current.obj '
               f'-d {file_path}/recon/recon.png '
               f'-p {file_path}/params.npz '
               f'-b {file_path}/undistort_img.png '
               f'-s {file_path}/recon/render_recon '
               f'-em {self.env_path} '
               f'-sm '
               f'-e c '
               f'-wp ')
        os.system(cmd)

    def render_exp_edit(self, file_path, exp_name):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/key_exp/{exp_name}.obj '
               f'-d {file_path}/exp_edit/{exp_name}.png '
               f'-p {file_path}/params.npz '
               f'-b {file_path}/undistort_img.png '
               f'-s {file_path}/exp_edit/render_{exp_name} '
               f'-em {self.env_path} '
               f'-sm '
               f'-e c '
               f'-wp '
               f'-nb ')
        os.system(cmd)

    def render_age_edit(self, file_path, target_age):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/current.obj '
               f'-d {file_path}/age_edit/age_{target_age:.2f}.png '
               f'-s {file_path}/age_edit/render_age_{target_age:.2f} '
               f'-em {self.env_path} '
               f'-sm '
               f'-e c ')
        os.system(cmd)

    def render_exp_inter(self, file_path, target_exp_name, n_frame):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/current.obj '
               f'-m2 {file_path}/key_exp/{target_exp_name}.obj '
               f'-d {file_path}/exp_inter_{target_exp_name}/frame_{{}}.png '
               f'-dd '
               f'-p {file_path}/params.npz '
               f'-b {file_path}/undistort_img.png '
               f'-s {file_path}/exp_inter_{target_exp_name}/render_frame '
               f'-sm '
               f'-nb '
               f'-wp '
               f'-e c '
               f'-em {self.env_path} '
               f'-f {n_frame} ')
        os.system(cmd)
        cmd = (f'{self.ffmpeg_path} '
               f'-start_number 0 '
               f'-r 15 '
               f'-i {file_path}/exp_inter_{target_exp_name}/render_frame_%d.jpg '
               f'-pix_fmt yuv420p '
               f'-y {file_path}/exp_inter_{target_exp_name}/render_frame.mp4 ')
        os.system(cmd)

    def render_age_progression(self, file_path, n_frame):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/current.obj '
               f'-d {file_path}/age_progression/frame_{{}}.png '
               f'-dd '
               f'-f {n_frame} '
               f'-s {file_path}/age_progression/render_frame '
               f'-sm '
               f'-e c '
               f'-em {self.env_path} ')
        os.system(cmd)
        cmd = (f'{self.ffmpeg_path} '
               f'-start_number 0 '
               f'-r 15 '
               f'-i {file_path}/age_progression/render_frame_%d.jpg '
               f'-pix_fmt yuv420p '
               f'-y {file_path}/age_progression/render_frame.mp4 ')
        os.system(cmd)

    def render_bs_anime(self, file_path, clip_name, bs_clip_path, bs_clip_start, bs_clip_end):
        cmd = (f'{self.blender_path} --background --factory-startup --python '
               f'./blender_scripts/camera_pers.py -- '
               f'-m {file_path}/bs/0.obj '
               f'--is_bs '
               f'-d {file_path}/bs_anime_{clip_name}/{{}}.png '
               f'-dd '
               f'-s {file_path}/bs_anime_{clip_name}/render_frame '
               f'-bs {bs_clip_path} '
               f'-sm '
               f'-e c '
               f'-em {self.env_path} '
               f'--bs_clip_start {bs_clip_start} '
               f'--bs_clip_end {bs_clip_end} ')
        os.system(cmd)
        cmd = (f'{self.ffmpeg_path} '
               f'-start_number 0 '
               f'-r 30 '
               f'-i {file_path}/bs_anime_{clip_name}/render_frame_%d.jpg '
               f'-pix_fmt yuv420p '
               f'-y {file_path}/bs_anime_{clip_name}/render_frame.mp4 ')
        os.system(cmd)
