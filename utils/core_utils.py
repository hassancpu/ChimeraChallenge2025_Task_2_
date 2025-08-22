
import numpy as np
import torch
from utils.utils import *
import os
import torch.nn.functional as F
from models.model_pool import Max_Pool, Mean_Pool
from models.model_pool_instance import Max_Pool_Ins, Mean_Pool_Ins
from models.model_gabmil import GABMIL
from models.model_abmil import ABMIL
from models.model_transmil import TransMIL
from models.model_clam import CLAM_SB
from models.model_madmil import MADMIL
from models.model_rrtmil import RRTMIL
from models.model_wikg import WiKG
from models.model_x import X_Cos
from models.model_dsmil import DSMIL
from models.model_acmil import ACMIL
from sklearn.metrics import roc_auc_score, f1_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true':[],'y_pred':[]}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        f1 = f1_score(y_true,y_pred,average='macro')
        return f1


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter # type: ignore
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    print('Done!')
    
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
    elif args.model_type == 'abmil':
        model = ABMIL(**model_dict)  
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
    model.relocate()
    print('Done!')
        
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True,weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience= 10, stop_epoch= 15, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type == 'clam':
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
        elif args.model_type == 'acmil':    
            train_loop_acmil(epoch, model, train_loader, optimizer, args.n_classes, args.n_token, writer, loss_fn)
        elif args.model_type == 'x_cos':    
            train_loop_x(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.model_type)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            
        stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        
    _, val_error, val_loss, val_auc, val_logger= summary(model, val_loader, args.n_classes, loss_fn)
    print('\n\nVal error: {:.4f}, Val loss: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_loss, val_auc))
    val_f1 = val_logger.get_f1()
    
    results_dict, test_error, test_loss, test_auc, acc_logger = summary(model, test_loader, args.n_classes, loss_fn)
    print('Test error: {:.4f}, Test loss: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_loss, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
            
    test_f1 = acc_logger.get_f1()
    
    if writer:
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/val_loss', val_loss, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.add_scalar('final/test_loss', test_loss, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
        
    return results_dict, test_auc, val_auc, test_f1, val_f1, test_loss, val_loss

# Training loop for GABMIL, Mean_Pool, Max_Pool, Mean_Pool_Ins, Max_Pool_Ins, MADMIL, RRTMIL
def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for _, (data, label, coord, clinic) in enumerate(loader):
        data, label, coord, clinic = data.to(device), label.to(device), coord.to(device), clinic.to(device)
        # forward pass
        logits, _, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


# Training loop for CLAM
def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for _, (data, label, _, clinic) in enumerate(loader):
        data, label, clinic = data.to(device), label.to(device), clinic.to(device)
        logits, _, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True, clinic=clinic)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


# Training loop for ACMIL
def train_loop_acmil(epoch, model, loader, optimizer, n_classes, n_token= 1, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_loss_slide= 0.
    train_loss_sub= 0.
    train_loss_diff= 0.
    
    train_error = 0.
    print('\n')
    for _, (data, label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        sub_preds, slide_preds, attn = model(data, use_attention_mask=True)

        if n_token > 1:
            loss0 = loss_fn(sub_preds, label.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.)

        loss1 = loss_fn(slide_preds, label)

        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)
        for i in range(int(n_token)):
            for j in range(i + 1, n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)

        loss = diff_loss + loss0 + loss1
        
        Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        train_loss_slide += loss1.item()
        train_loss_sub += loss0.item()
        train_loss_diff += diff_loss.item()

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_loss_slide /= len(loader)
    train_loss_sub /= len(loader)
    train_loss_diff /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_loss_slide: {:.4f}, train_loss_sub: {:.4f}, train_loss_diff: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_loss_slide, train_loss_sub, train_loss_diff, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


# Training loop for X
def train_loop_x(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, model_type = 'x_cos'):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_loss_slide= 0.
    train_loss_sub= 0.

    train_error = 0.
    print('\n')
    for _, (data, label, coord, clinic) in enumerate(loader):
        data, label, coord, clinic = data.to(device), label.to(device), coord.to(device), clinic.to(device)
        
        logits, _, Y_hat, attn, _ = model(data, coords= coord, clinic= clinic)

        if model_type == 'x_cos':
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)
            for i in range(int(n_classes)):
                for j in range(i + 1, n_classes):
                    diff_loss += torch.cosine_similarity(attn[:, i, :], attn[:, j, :], dim=-1).mean() / (
                                n_classes * (n_classes - 1) / 2)
                    
            loss0 = diff_loss
    
        # slide prediction loss
        slide_preds = logits
        loss1 = loss_fn(slide_preds, label)

        loss = loss0 + loss1
        
        Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        train_loss_slide += loss1.item()
        train_loss_sub += loss0.item()

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_loss_slide /= len(loader)
    train_loss_sub /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_loss_slide: {:.4f}, train_loss_sub: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_loss_slide, train_loss_sub, train_error))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)



# Validation loop for all models
# Note: The validation loop is the same for all models  
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))


    with torch.no_grad():
        for batch_idx, (data, label, coord, clinic) in enumerate(loader):
            data, label, coord, clinic = data.to(device, non_blocking=True), label.to(device, non_blocking=True), coord.to(device, non_blocking=True), clinic.to(device, non_blocking=True)

            if model.__class__.__name__== 'ACMIL':
                _, slide_preds, _ = model(data,use_attention_mask=False)
            
                Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]   
                Y_prob = F.softmax(slide_preds, dim = 1) 
                logits = slide_preds
            elif model.__class__.__name__== 'X_Cos':
                logits, Y_prob, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)                
            else:    
                logits, Y_prob, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            acc_logger.log(Y_hat, label)
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur))) 
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False



# Summary for all models
# Note: The summary loop is the same for all models
def summary(model, loader, n_classes, loss_fn):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

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
                logits, Y_prob, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)
            else:    
                logits, Y_prob, Y_hat, _, _ = model(data, coords= coord, clinic= clinic)
            
            loss = loss_fn(logits, label)

            test_loss += loss.item()                
            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error
        
    test_loss /= len(loader)
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    return patient_results, test_error, test_loss, auc, acc_logger
