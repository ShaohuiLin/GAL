import argparse
import os 
import pdb
parser = argparse.ArgumentParser(description='Generatvie Adversarial Learning')

## Warm-up 
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help='Dataset to train')
parser.add_argument(
    '--data_dir',
    type=str,
    default='data/',
    help='The directory where the input data is stored.')
parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--teacher_dir',
    type=str,
    default='pretrained/',
    help='The directory where the teacher model saved.')
parser.add_argument(
    '--reset',
    action='store_true',
    help='Reset the directory?')
parser.add_argument(
    '--resume', 
    type=str, 
    default=None,
    help='Load the model from the specified checkpoint.')
parser.add_argument(
    '--refine', 
    type=str, 
    default=None,
    help='Path to the model to be fine tuned.')
    
## Training
parser.add_argument(
    '--student_model',
    type=str,
    default='resnet_56_sparse',
    help='The model of student.')
parser.add_argument(
    '--teacher_model',
    type=str,
    default='resnet_56',
    help='The model of teacher.')
parser.add_argument(
    '--num_epochs', 
    type=int,
    default=100,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer.')
parser.add_argument(
    '--lr',
    type=float,
    default=1e-2
)
parser.add_argument(
    '--lr_decay_step',
    type=int,
    default=30
)
parser.add_argument(
    '--mask_step',
    type=int,
    default=200,
    help='The frequency of mask to update'
)
parser.add_argument(
    '--weight_decay', 
    type=float,
    default=2e-4,
    help='The weight decay of loss.')
parser.add_argument(
    '--miu', 
    type=float,
    default=1,
    help='The miu of data loss.')
parser.add_argument(
    '--lambda',
    dest='sparse_lambda',
    type=float,
    default=0.6,
    help='The sparse lambda for l1 loss')
parser.add_argument(
    '--random',
    action='store_true',
    help='Random weight initialize for finetune')
parser.add_argument(
    '--pruned',
    action='store_true',
    help='Load pruned model')
parser.add_argument(
    '--thre',
    type=float,
    default=0.0,
    help='Thred of mask to be pruned')
parser.add_argument(
    '--keep_grad',
    action='store_true',
    help='Keep gradients of mask for finetune')

## Status
parser.add_argument(
    '--print_freq', 
    type=int,
    default=200,
    help='The frequency to print loss.')
parser.add_argument(
    '--test_only', 
    action='store_true',
    help='Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

