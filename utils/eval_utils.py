import numpy as np

import torch
import pandas as pd
import torch.nn.functional as F
from models.model_pool import Max_Pool, Mean_Pool
from models.model_pool_instance import Max_Pool_Ins, Mean_Pool_Ins
from models.model_gabmil import GABMIL
from models.model_x import X_Cos
from models.model_transmil import TransMIL
from models.model_clam import CLAM_SB
from models.model_madmil import MADMIL
from models.model_rrtmil import RRTMIL
from models.model_wikg import WiKG
from models.model_dsmil import DSMIL
from models.model_acmil import ACMIL
from utils.utils import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score,  auc, precision_recall_curve, f1_score, cohen_kappa_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true':[],'y_pred':[], 'y_prob':[]}

    def log(self, Y_prob, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
        self.data_all['y_prob'].append(Y_prob)

    def get_accuracy(self):
        correct = sum([x["correct"] for x in self.data])
        count = sum([x["count"] for x in self.data])
        accuracy = correct / count
        return accuracy

    def get_balanced_accuracy(self):
        acc = []
        for i in range(self.n_classes):
            acc.append(self.get_summary(i))
        balanced_accuracy = np.mean(acc)
        return balanced_accuracy
    
    def get_f1(self):
        thresholds = np.arange(0, 1.01, 0.01)
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_prob = np.asarray(self.data_all['y_prob']).reshape(-1,)

        best_f1 = 0
        best_thresh = 0.5
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        print(f'Best threshold: {best_thresh}, Best F1: {best_f1}')
        return best_f1
    
    def get_prc(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        prc = precision_score(y_true, y_pred, average='macro')
        return prc

    def get_recall(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        recall = recall_score(y_true, y_pred, average='macro')
        return recall

    def cohen_kappa(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        kappa = cohen_kappa_score(y_true, y_pred)
        return kappa
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        if count == 0:
            acc = None
        else:
            acc = float(correct) / count
        return acc
        
def initiate_model(args, ckpt_path):
    print('\nInit Model...', end=' ')
    model_dict = {"n_classes": args.n_classes, "feat_type": args.feat_type}    
    if args.model_type == 'max_pool':
        model = Max_Pool(**model_dict)
    elif args.model_type == 'mean_pool':
        model = Mean_Pool(**model_dict)
    elif args.model_type == 'max_pool_ins':
        model = Max_Pool_Ins(**model_dict)
    elif args.model_type == 'mean_pool_ins':
        model = Mean_Pool_Ins(**model_dict)              
    elif args.model_type == 'gabmil':
        model_dict.update({"use_local": args.use_local, "use_block": args.use_block, "use_grid": args.use_grid, "win_size_b": args.win_size_b, "win_size_g": args.win_size_g})
        model = GABMIL(**model_dict)  
    elif args.model_type == 'x_cos':
        model = X_Cos(**model_dict)
    elif args.model_type == 'clam':
        model_dict.update({"subtyping": True, 'k_sample': args.B})  
        from topk.svm import SmoothTop1SVM # type: ignore
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        instance_loss_fn = instance_loss_fn.cuda()     
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif args.model_type == 'transmil':
        model = TransMIL(**model_dict)
    elif args.model_type == 'madmil':
        model_dict.update({"n": args.n})
        model = MADMIL(**model_dict)  
    elif args.model_type == 'rrtmil':
        model = RRTMIL(epeg_k=15,crmsa_k=3, **model_dict)
    elif args.model_type == 'wikg':
        model = WiKG(**model_dict)
    elif args.model_type == 'dsmil':
        model = DSMIL(**model_dict)
    elif args.model_type == 'acmil':
        model_dict.update({"n_token": args.n_token, "n_masked_patch": args.n_masked_patch, "mask_drop": args.mask_drop})
        model = ACMIL(**model_dict)
        
    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    # load the model    
    model.load_state_dict(ckpt_clean, strict=True)
    print('Load checkpoint from {}'.format(ckpt_path))

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(args.device)

    patient_results, auc, pr_area, loss, df, test_logger = summary(model, loader, args, loss_fn)

    acc = test_logger.get_accuracy()
    f1 = test_logger.get_f1()
    prc = test_logger.get_prc()
    recall = test_logger.get_recall()
    kappa = test_logger.cohen_kappa()
    bacc = test_logger.get_balanced_accuracy()

    # class-wise accuracy
    acc_class = []
    for i in range(args.n_classes):
        acc = test_logger.get_summary(i)
        acc_class.append(acc)

    return model, patient_results, auc, acc, acc_class, bacc, f1, prc, recall, pr_area, loss, kappa, df

def summary(model, loader, args, loss_fn):
    device = args.device
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_error = 0.
    test_loss = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    with torch.no_grad():
        for batch_idx, (data, label, coord, clinic) in enumerate(loader):
            data, label, coord, clinic = data.to(device, non_blocking=True), label.to(device, non_blocking=True), coord.to(device, non_blocking=True), clinic.to(device, non_blocking=True)   
            slide_id = slide_ids.iloc[batch_idx]
            
            if model.__class__.__name__== 'ACMIL':
                _, slide_preds, _ = model(data,use_attention_mask=False)
            
                Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]   
                Y_prob = F.softmax(slide_preds, dim = 1) 
                logits = slide_preds
            
            elif model.__class__.__name__== 'X_Cos':
                logits, _, _ = model(data)
                slide_preds = logits[-1]
                
                Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]
                Y_prob = F.softmax(slide_preds, dim = 1) 
                logits = slide_preds
        
            else:    
                logits, Y_prob, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)
            
            loss = loss_fn(logits, label)

            test_loss += loss.item() 

            acc_logger.log(Y_prob[0][1].cpu().numpy(), Y_hat, label)
            probs = Y_prob.cpu().numpy()
            
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

    del data
    test_loss /= len(loader)
    test_error /= len(loader)

    if args.n_classes == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])

        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        pr_area = auc(recall, precision)
    else:
        # Compute ROC AUC for multi-class
        auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        # Compute Precision-Recall AUC for each class and average
        pr_area = 0
        for i in range(args.n_classes):
            precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
            pr_area += auc(recall, precision)
        pr_area /= args.n_classes

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})

    df = pd.DataFrame(results_dict)
    
    return patient_results, auc_score, pr_area, test_loss, df, acc_logger
