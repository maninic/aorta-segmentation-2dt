import argparse
import os

import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        parser.add_argument('--data_path', type=str, help='Images path')
        parser.add_argument('--model_path', type=str, help='model path where you can find the created dictionaries for training validation and testing')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--patch_size', default=(64, 64), help='Size of the crop or pad for the image')
        parser.add_argument('--sd', type=int, default=3, help='# spatial dimension')
        parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        parser.add_argument('--channels', default=(4, 8, 16, 32, 64, 128), help='channels')
        parser.add_argument('--strides', default=(2, 2, 2, 2, 2), help='strides for each layer')
        parser.add_argument('--res_unit', type=int, default=2, help='# of residual unit')
        parser.add_argument('--Tresample', type=int, default=64, help='final temporal dimension (third dimension in our case)')
        parser.add_argument('--sp_res', type=int, default=(1.54, 1.54), help='final spatial resolution (2D)')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout to apply to the model')
        #check this out
        parser.add_argument('--gpu_ids', default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

        self.initialized = True

        return parser
    
    def gather_options(self):
        # initialize parser with basic options
        #if not self.initialized:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, dir):
        opt = self.gather_options()
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(dir, 'options.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # set gpu ids
#        str_ids = list(opt.gpu_ids)
#        str_ids.remove(',')
#        opt.gpu_ids = []
#        for str_id in str_ids:
#            id = int(str_id)
#            if id >= 0:
#                opt.gpu_ids.append(id)
#        if len(opt.gpu_ids) > 0:
#            torch.cuda.set_device(opt.gpu_ids[0])


        self.opt = opt
        return self.opt
