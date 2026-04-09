import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # Data and Experiment
        self.parser.add_argument('--exp', default='anime_transfer', help='name of experiment')
        self.parser.add_argument('--input_path', default='data/data_anime', help='path to input data')
        self.parser.add_argument('--output_path', default='checkpoints', help='path to output saves')
        self.parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
        self.parser.add_argument('--image_size', type=int, default=512, help='size of image')
        self.parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
        self.parser.add_argument('--contrast1', type=str, default='A', help='contrast selection for model')
        self.parser.add_argument('--contrast2', type=str, default='B', help='contrast selection for model')

        # Diffusion Parameters
        self.parser.add_argument('--num_timesteps', type=int, default=4)
        self.parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
        self.parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
        self.parser.add_argument('--use_geometric', action='store_true', default=False)
        self.parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
        
        # Generator (NCSN++) Architecture
        self.parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denosing model')
        self.parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
        self.parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
        self.parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
        self.parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
        self.parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
        self.parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
        self.parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
        self.parser.add_argument('--fir', action='store_false', default=True, help='FIR')
        self.parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
        self.parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
        self.parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
        self.parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
        self.parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
        self.parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
        self.parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
        self.parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
        self.parser.add_argument('--not_use_tanh', action='store_true', default=False)
        self.parser.add_argument('--nz', type=int, default=100)
        self.parser.add_argument('--z_emb_dim', type=int, default=256)
        self.parser.add_argument('--t_emb_dim', type=int, default=256)

        # Optimization Parameters (Common)
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        return args

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--resume', action='store_true', default=False)
        self.parser.add_argument('--num_epoch', type=int, default=200)
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
        self.parser.add_argument('--no_lr_decay', action='store_true', default=False)
        self.parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
        self.parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
        self.parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
        self.parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')
        self.parser.add_argument('--save_content', action='store_true', default=False)
        self.parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
        self.parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
        self.parser.add_argument('--lambda_l1_loss', type=float, default=0.5, help='weightening of l1 loss part of diffusion ans cycle models')
        
        # DDP
        self.parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
        self.parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
        self.parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
        self.parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
        self.parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
        self.parser.add_argument('--port_num', type=str, default='6021', help='port selection for code')

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--compute_fid', action='store_true', default=False, help='whether or not compute FID')
        self.parser.add_argument('--epoch_id', type=int, default=1000)
        self.parser.add_argument('--which_epoch', type=int, default=50)
        self.parser.add_argument('--gpu_chose', type=int, default=0)
        self.parser.add_argument('--source', type=str, default='T2', help='source contrast')
