import argparse
import os

class DictOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        parser.add_argument('--data_path', type=str, help='Data path, can contain subfolder for the different sites')
        parser.add_argument('--datasets', nargs = "+", help='List of subfolder (different sites) in the data folder to include in the training separate them with a space')
        parser.add_argument('--out_path', type=str, help='output path where to save the dictionaries')
        parser.add_argument('--filter_data', type=int, default=0, help='Do you want to filter the data to use in your model? 0=no and 1=yes', required=True)
        parser.add_argument('--discriminator', type=str, default='DatasetInfo.xlsx', help='csv file containing the information to split the data')
        parser.add_argument('--criteria_name', type=str, help='criteria to use to split. Usually one of: PatientSex, PatientAge, BAV. In general one of the column in the discriminator csv file')
        parser.add_argument('--criteria_value', nargs = "+", help='criteria value to use to split. Examples: M for PatientSex or 1 for BAV. For age goups give the minimum and maximum of the age to include separated by a space')
        parser.add_argument('--train_split', type=float, default = 0.8, help='split value for the training set, between 0 and 1')
        parser.add_argument('--val_split', type=float, default = 0.1, help='split value for the validation set, between 0 and 1')
        parser.add_argument('--modelID', type=str, default = 'newmodel', help='give a name for the model identifier, it will create a new folder to not overwrite previous models')
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

    def print_options(self,dir):
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
        file_name = os.path.join(dir, 'dict_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()

        self.opt = opt
        return self.opt
