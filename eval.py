from __future__ import print_function


import argparse
import torch # type: ignore
import os
import pandas as pd # type: ignore
from utils.utils import *
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *


# Generic settings
parser = argparse.ArgumentParser(description='Configurations for WSI Evaluation')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--results_dir', type=str, default=None, help='(default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None, help='experiment code to load trained models')
parser.add_argument('--seed', type=int, default=2021,help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='start fold')
parser.add_argument('--k_end', type=int, default=5, help='end fold')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--splits_dir', type=str, default=None,help='manually specify the set of splits to use')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['bladder-brs'],help='task to train')
parser.add_argument('--model_type', type=str, choices=['max_pool', 'mean_pool', 'max_pool_ins', 'mean_pool_ins', 'gabmil', 'clam', 'transmil', 'madmil', 'rrtmil', 'simlp', 'wikg', 'x_cos', 'dsmil', 'acmil'], default='gabmil', help='type of model')
parser.add_argument('--feat_type', type=str, choices=['uni', 'gigapath'], default='uni', help='type of features to use')

# GABMIL specific options
parser.add_argument('--use_local', action='store_true', default=False, help='no global information')
parser.add_argument('--use_grid', action='store_true', default=False, help='enable grid information')
parser.add_argument('--use_block', action='store_true', default=False, help='enable block information')
parser.add_argument('--win_size_b', type=int, default=1, help='block window size')
parser.add_argument('--win_size_g', type=int, default=1, help='grid window size')

### CLAM specific options
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm')
parser.add_argument('--subtyping', action='store_true', default=True)
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--B', type=int, default=8)

### MAD-MIL specific options
parser.add_argument('--n', type= int, default= 2, help='number of heads in the attention network')

### ACMIL specific options  
parser.add_argument("--n_token", type=int, default=5, help="number of attention branches.")
parser.add_argument("--n_masked_patch", type=int, default=10, help="top-K instances are be randomly masked in STKIM.")
parser.add_argument("--mask_drop", type=float, default=0.6, help="maksing ratio in the STKIM")

args = parser.parse_args()

args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.task = "bladder-brs"
args.splits_dir = "./splits/bladder-brs"
args.csv_path = './dataset_csv/bladder-brs.csv'
args.results_dir = './results/UNI/MULTI_10x/'
args.eval_dir = './eval_results/UNI/MULTI_10x/'
args.data_root_dir = "/home/20215294/Data/CHIM/CHIM_ostu_10x/"
args.feat_type = 'uni'
sub_feat_dir = 'feat_uni'
args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)


assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)


settings = {
            'local': args.use_local,
            'grid': args.use_grid,
            'block': args.use_block,
            'win_size_b': args.win_size_b,
            'win_size_g': args.win_size_g,
            'model_type': args.model_type,
            'n': args.n,
            'n_token': args.n_token,
            'n_masked_patch': args.n_masked_patch,
            'mask_drop': args.mask_drop,
            'sub_feat_dir': sub_feat_dir,}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

   
if args.task == 'bladder-brs': 
    args.k = 5
    args.k_end = 5   
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='BRS',
                            label_dict = {'BRS1':0, 'BRS2':0, 'BRS3':1},
                            patient_strat=False,
                            ignore=[])             
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_acc_class = []
    all_bacc = []
    all_f1 = []
    all_prc = []
    all_recall = []
    all_pr_area = []
    all_loss = []
    all_kappa = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        model, patient_results, auc, acc, acc_class, bacc, f1, prc, recall, pr_area, loss, kappa, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(acc)
        all_acc_class.append(acc_class)
        all_bacc.append(bacc)
        all_f1.append(f1)
        all_prc.append(prc)
        all_recall.append(recall)
        all_pr_area.append(pr_area)
        all_loss.append(loss)
        all_kappa.append(kappa)

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_bacc': all_bacc, 'test_f1': all_f1, 'test_prc': all_prc, 'test_recall': all_recall, 'test_pr_area': all_pr_area, 'test_loss': all_loss, 'test_kappa': all_kappa})

    save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    
    "Compute the average and std of the metrics"
    test_auc_ave= np.mean(all_auc)
    test_acc_ave= np.mean(all_acc)
    test_acc_class_ave= np.mean(all_acc_class, axis=0)
    test_bacc_ave= np.mean(all_bacc)
    test_f1_ave= np.mean(all_f1)
    test_prc_ave= np.mean(all_prc)
    test_recall_ave= np.mean(all_recall)
    test_pr_area_ave= np.mean(all_pr_area)
    test_loss_ave= np.mean(all_loss)
    test_kappa_ave= np.mean(all_kappa)

    test_auc_std= np.std(all_auc)
    test_acc_std= np.std(all_acc)
    test_acc_class_std= np.std(all_acc_class, axis=0)
    test_bacc_std= np.std(all_bacc)
    test_f1_std= np.std(all_f1)
    test_prc_std= np.std(all_prc)
    test_recall_std= np.std(all_recall)
    test_pr_area_std= np.std(all_pr_area)
    test_loss_std= np.std(all_loss)
    test_kappa_std= np.std(all_kappa)
 
    print('\n\nTest:\n pr_area ± std: {0:.2f} ± {1:.2f}, loss ± std: {2:.2f} ± {3:.2f}, bacc ± std: {4:.2f} ± {5:.2f}, f1 ± std: {6:.2f} ± {7:.2f}, kappa ± std: {8:.2f} ± {9:.2f}\n\n'.
          format(test_pr_area_ave, test_pr_area_std, test_loss_ave, test_loss_std, test_bacc_ave, test_bacc_std, test_f1_ave, test_f1_std, test_kappa_ave, test_kappa_std))    