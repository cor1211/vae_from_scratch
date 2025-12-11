import torch
import os
import sys
from tqdm import tqdm
from model import VAE

class Trainer():
    def __init__(self, model, optimizer, train_loader, valid_loader, device, config, writer, run_name, resume_path):
        # config
        self.config = config
        self.train_cfg = self.config['train']
        self.valid_cfg = self.config['valid']
        self.model_cfg = self.config['model']

        self.device = device
        self.writer = writer
        self.run_name = run_name
        self.resume_path = resume_path

        # model 
        self.model = model
        self.optimizer = optimizer

        # dataloader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)
        
        self.start_epoch = 1
        self.best_psnr = 0.0
        self.checkpoint_path = os.path.join('checkpoints', self.run_name)
        
        self.total_epochs = self.train_cfg['epoch']
        self.end_epoch = self.total_epochs + self.start_epoch

        if self.resume_path:
            self._load_checkpoint(self.resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.end_epoch = self.start_epoch + self.total_epochs
            self.best_psnr = checkpoint['best_psnr']
        
        except Exception as e:
            print(f'Error loading checkpoint {e}. Check your resume path')
            sys.exit(1)

    
    def _save_checkpoint(self, epoch: int, is_best: bool):
        checkpoint_data = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        last_save_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint_data, last_save_path)
        print(f"Epoch {epoch}: New best model saved to {last_save_path}")


        if is_best:
            best_save_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint_data, best_save_path)
            print(f"Epoch {epoch}: New best model saved to {best_save_path}")

    def _train_epoch(self, epoch: int):
        self.model.train()
        tqdm_train_loader = tqdm(self.train_loader, desc=f'Epoch [{epoch}/{self.end_epoch-1}] Train')

        for iter, input in enumerate(tqdm_train_loader):
            input = input.to(self.device)
            out = self.model(input)
            