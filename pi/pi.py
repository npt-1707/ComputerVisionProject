from pi.dataset import get_cifar10, get_cifar100, get_svhn
from pi.model import PiModel

import torch, logging, os, numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torch.optim.lr_scheduler import LambdaLR

def get_dataloader(args):
    get_dataset = {
        "cifar10": get_cifar10,
        "cifar100": get_cifar100,
        "svhn": get_svhn
    }
    assert args.dataset in get_dataset.keys(), "Dataset must be in {}".format(
        get_dataset.keys())
    train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset = get_dataset[
        args.dataset](args)

    labeled_sampler = RandomSampler(train_labeled_dataset,
                                    replacement=True,
                                    num_samples=len(train_unlabeled_dataset) //
                                    args.uratio)
    unlabeled_sampler = RandomSampler(train_unlabeled_dataset)
    labeled_batch_sampler = BatchSampler(labeled_sampler,
                                         args.batch_size//2,
                                         drop_last=True)
    unlabeled_batch_sampler = BatchSampler(unlabeled_sampler,
                                           args.batch_size//2,
                                           drop_last=True)
    labeled_loader = DataLoader(train_labeled_dataset,
                                batch_sampler=labeled_batch_sampler,
                                num_workers=args.num_workers)
    unlabeled_loader = DataLoader(train_unlabeled_dataset,
                                  batch_sampler=unlabeled_batch_sampler,
                                  num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    return labeled_loader, unlabeled_loader, valid_loader, test_loader
    
def cal_consistency_weight(epoch, args):
    """Sets the weights for the consistency loss"""
    if epoch > args.ramp_up_length:
        return args.wu
    else:
        return ramp_up(epoch, args.ramp_up_length) * args.wu

def ramp_up(current_epoch, ramp_up_length):
    t = np.clip(current_epoch / ramp_up_length, 0.0, 1.0)
    return np.exp(-5 * (1 - t) ** 2)

def ramp_down(current_epoch, total_epoch, ramp_down_length):
    t = np.clip((current_epoch - (total_epoch - ramp_down_length)) / ramp_down_length, 0.0, 1.0)
    return np.exp(-12.5 * t ** 2)

def adjust_optimizer(optimizer, epoch, args):
    lr_rate = 1.0
    b1_rate = 1.0
    if epoch < args.ramp_up_length:
        lr_rate = ramp_up(epoch, args.ramp_up_length)
    if epoch >= args.epoch - args.ramp_down_length:
        lr_rate = ramp_down(epoch, args.epochs, args.ramp_down_length)
        b1_rate = 1 - ramp_down(epoch, args.epochs, args.ramp_down_length)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_rate
        param_group['betas'] = (0.9 - 0.4 * b1_rate, 0.999)
class Pi:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PiModel(args.num_classes).to(self.device)
        self.criteria = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(),args.lr,
                                    betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)
        self.epoch = 0
        self.args = args
        
        self.train_labeled_loader, self.train_unlabeled, self.valid_loader, self.test_loader = get_dataloader(args)
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        logging.info("Loading checkpoint")
        checkpoints = [
            x for x in os.listdir(self.save_path)
            if f"{self.args.dataset}_{self.args.num_labels}_checkpoint" in x
        ]
        logging.info(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        if len(checkpoints) > 0:
            checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = checkpoints[-1]
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint))
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_acc = checkpoint["best_acc"]
            self.best_model = checkpoint["model"]
            self.epoch = checkpoint["epoch"]
            self.train_loss = checkpoint["train_loss"]
            self.valid_loss = checkpoint["valid_loss"]
        
    def train(self):
        logging.info("Start training" if self.epoch ==
                     0 else "Continue training")
        
        self.train_loss = []
        self.valid_loss = []
        
        labeled_iter = iter(self.train_labeled_dataloader)
        unlabeled_iter = iter(self.train_unlabeled_dataloader)
        for epoch in range(self.epoch, self.args.epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")
            adjust_optimizer(self.optimizer, epoch, self.args)
            logging.info(
                f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            total_loss = []
            label_loss = []
            unlabel_loss = []
            len_iter = len(unlabeled_iter)
            for batch_idx in range(len_iter):
                weight_cl = cal_consistency_weight(epoch*len_iter+batch_idx, end_ep=(self.args.epochs//2)*len_iter, end_w=1.0)
                try:
                    l_img, label = next(labeled_iter)
                except:
                    labeled_iter = iter(self.train_labeled_dataloader)
                    l_img, label = next(labeled_iter)
                u_1_img, u_2_img = next(unlabeled_iter) 
                l_img, label, u_w_img, u_s_img = l_img.to(
                    self.device), label.to(self.device), u_w_img.to(
                        self.device), u_s_img.to(self.device)
                l_logit, u_1_logit, u_2_logit = self.model(l_img), self.model(u_1_img), self.model(u_2_img)
                l_loss = F.cross_entropy(l_logit, label, reduction="mean")
                u_loss = F.mse_loss(u_1_logit, u_2_logit, reduction="mean")
                
                loss = l_loss + weight_cl * u_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                total_loss.append(loss.item())
                label_loss.append(l_loss.item())
                unlabel_loss.append(u_loss.item())
                
                if batch_idx % int(self.args.eval_steps / 5) == 0:
                    logging.info(
                        f"\tBatch: {batch_idx}/{self.args.eval_steps} - Loss: {loss.item():.6f} - Label Loss: {l_loss.item():.6f} - Pseudo Loss: {u_loss.item():.6f}"
                    )
                    
                self.optimizer.step()
                logging.info(f"Train loss: {sum(total_loss) / len(total_loss)}")
                self.train_loss.append(sum(total_loss) / len(total_loss))
                self.valid_loss.append(self.validate())
                
            if (epoch + 1) % 50 == 0:
                save = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "best_model": self.best_model,
                    "train_loss": self.train_loss,
                    "valid_loss": self.valid_loss,
                }
                torch.save(
                    save,
                    "{}/{}_{}_checkpoint_{}".format(self.save_path,
                                                    self.args.dataset,
                                                    self.args.num_labels,
                                                    epoch + 1))
            if epoch == self.args.epochs - 1:
                self.test()
                
        logging.info('Training completed.')
        print("Training completed") if self.args.debug else None
    
    def validate(self):
        logging.info("Validating")
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for img, label in self.valid_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                logit = self.model(img)
                _, pred = torch.max(logit, dim=1)
                loss = self.criterion(logit, label)
                total_loss += loss.item()
                correct += (pred == torch.max(label, dim=1)[1]).sum().item()
                total += len(label)
        acc = correct / total
        total_loss /= len(self.test_dataloader)
        logging.info(f"Accuracy: {acc:.6f} - Loss: {total_loss:.6f}")
        print(f"Validating: Accuracy: {acc:.6f} - Loss: {total_loss:.6f}"
              ) if self.args.debug else None
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = self.model.state_dict()
        return total_loss, acc
    
    def test(self):
        logging.info("Testing")
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for img, label in self.test_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                logit = self.model.ema(img)
                _, pred = torch.max(logit, dim=1)
                loss = self.criterion(logit, label)
                total_loss += loss.item()
                correct += (pred == torch.max(label, dim=1)[1]).sum().item()
                total += len(label)
        acc = correct / total
        total_loss /= len(self.test_dataloader)
        logging.info(f"Accuracy: {acc:.6f} - Loss: {total_loss:.6f}")
        print(f"Testing: Accuracy: {acc:.6f} - Loss: {total_loss:.6f}"
              ) if self.args.debug else None
        