from Options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=200, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        #
        parser.add_argument('--weight_celoss', type=float, default=0.7, help='weight of the CE loss')
        parser.add_argument('--threshold', type=float, default=0.5, help='threshold for the probability map')
        parser.add_argument('--n_epoch', type=int, default=100, help='number of epoch')
        parser.add_argument('--writer', type=bool, default=True, help='do you want to save the writer file?')
        parser.add_argument('--val_nepoch', type=int, default=1, help='validation interval in epoch')
        parser.add_argument('--save_final', type=bool, default=True, help='save the final model even if not the best')
        parser.add_argument('--eval_metrics', default=['DICE', 'hausdorff', 'ASSD', 'sens', 'prec', 'acc', 'mr'], nargs = "+", help='List of metrics to be used for evaluation, default include DICE, hausdorff, ASSD, send, prec, acc, mr')
        parser.add_argument('--save_predictions', type=bool, default=False, help='save the prediction of the model')
        
        self.isTrain = True
        return parser
