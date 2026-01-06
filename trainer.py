import torch
import os
import sys
from tqdm import tqdm
from model import VAE
from utils import KL_divergence, L2_loss
from torch.nn import MSELoss
from ignite.metrics import PSNR, SSIM, Loss
from ignite.engine import Engine
from torchvision.utils import make_grid


def denorm(tensor:torch.Tensor) -> torch.Tensor:
    """
    Denorm (B, C, H, W) Tensor from [-1, 1] to [0, 1]
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


class Trainer():
    def __init__(self, model, optimizer, train_loader, valid_loader, device, config, writer, run_name, resume_path):
        # config
        self.config = config
        self.train_cfg = self.config['train']
        self.val_step = self.train_cfg.get('val_step', 1000)
        self.total_epochs = self.train_cfg.get('total_epochs', 100)
        
        # model 
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer

        # Kl_loss coef
        self.lambda_kl = self.train_cfg.get('lambda_kl', 1)

        # DataLoader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)

        self.total_steps = self.total_epochs * self.num_iter_train

        # Log, checkpoints
        self.writer = writer
        self.run_name = run_name
        self.resume_path = resume_path
        self.checkpoint_path = os.path.join('checkpoints', self.run_name)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.current_step = 0
        self.best_psnr = 0.0

        # Losses
        self.criterion_L2 = MSELoss()
        self.loss_l2 = 0.0
        self.loss_kl = 0.0

        if self.resume_path:
            self._load_checkpoint(self.resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_step = checkpoint['step']
            self.best_psnr = checkpoint['best_psnr']

            print(f"Resumed from step {self.current_step}. Best PSNR so far: {self.best_psnr}")

        except Exception as e:
            print(f'Error loading checkpoint {e}. Check your resume path')
            sys.exit(1)

    
    def _save_checkpoint(self, step: int, is_best: bool):
        checkpoint_data = {
            'step': step, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        last_save_path = os.path.join(self.checkpoint_path, 'last.pth')
        torch.save(checkpoint_data, last_save_path)

        if is_best:
            best_save_path = os.path.join(self.checkpoint_path, 'best.pth')
            torch.save(checkpoint_data, best_save_path)
            print(f"Step {step}: New best model saved to {best_save_path}")
    

    def _validate_step(self, current_step):
        """
        Process an validation by val_step
        """
        def eval_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                img = batch
                img = img.to(self.device)
                _, _, img_gen = self.model(img) # Forward
            
            return img, img_gen

        #-----------------Compute the losses of training phrase------------
        loss_l2_avg = self.loss_l2 / self.val_step
        loss_kl_avg = self.loss_kl / self.val_step

        # Show LOSSES LOG of Training Phrase
        print(
f"""Step [{self.current_step}/{self.total_steps}]
{20 * '-'}
Average Train L2 Loss: {loss_l2_avg:.3f}
Average Train KL Loss: {loss_kl_avg:.3f}
{20 * '-'}"""
)

        print('Start Validating...')

        #--------------Compute metrics of valid set-------------------
        evaluator = Engine(eval_step)
        Loss(self.criterion_L2, output_transform=lambda x: (x[0], x[1])).attach(evaluator, 'l2')
        PSNR(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(evaluator, 'psnr')
        SSIM(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(evaluator, 'ssim')
        evaluator.run(tqdm(self.valid_loader, desc="Validating", leave=False))
        l2_avg = evaluator.state.metrics['l2']
        psnr_avg = evaluator.state.metrics['psnr']
        ssim_avg = evaluator.state.metrics['ssim']
        
        # Show METRICS LOG of Valid Phrase
        print(
f"""{20 * '-'}
L2: {l2_avg:.3f}
PSNR: {psnr_avg:.3f} dB
SSIM: {ssim_avg:.3f}
{20 * '-'}""")

        # Log at tensorboard
        self.writer.add_scalar(tag = 'L2 Loss/Train_Step', scalar_value = loss_l2_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'KL Loss/Train_Step', scalar_value = loss_kl_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'Metrics/L2', scalar_value = l2_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'Metrics/PSNR', scalar_value = psnr_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'Metrics/SSIM', scalar_value = ssim_avg, global_step = current_step)

        # Log images
        real, fake = evaluator.state.output
        n_imgs = min(8, fake.size(0))        
        self.writer.add_image('Images/Fake', make_grid((denorm(fake))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/Real', make_grid((denorm(real))[:n_imgs], nrow=4), current_step)

        #----------SAVE Checkpoints-----------
        is_best_psnr = psnr_avg > self.best_psnr # is_best ckp ?
        if is_best_psnr:
            self.best_psnr = psnr_avg
            print(f'New best PSNR: {self.best_psnr:.3f}')
        else:
            print(f'Not best PSNR. Only save at latest checkpoint')

        self._save_checkpoint(current_step, is_best_psnr)

        # Reset The losses of train phrase
        self.loss_kl = 0.0
        self.loss_l2 = 0.0


    def run(self):
        if not self.resume_path:
            print(f"""--------------------
                \nStarting new run: {self.run_name}
                """)
        else:
            print(f"""------------------
                  \nResuming run '{self.run_name}' from step {self.current_step}.
                """)
        

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.total_steps, initial=self.current_step, desc='Training')

        #-------------------MAIN-----------------
        while self.current_step < self.total_steps:
            try:
                img = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                img = next(train_iter)
            
            img = img.to(self.device)

            
            #--------------Training--------------
            self.model.train()
            mean, logvar , img_gen = self.model(img) # Forward
            self.optimizer.zero_grad() # Clear old gradients

            # Compute losses
            lossMSE = self.criterion_L2(img_gen, img)
            lossKL = KL_divergence(mean, logvar)
            loss_total = lossMSE + self.lambda_kl*lossKL

            # Update weights
            loss_total.backward() # Compute gradients
            self.optimizer.step() # Update weights
            

            #-----------Logging and Update-----------
            self.loss_kl += lossKL.item()
            self.loss_l2 += lossMSE.item()

            pbar.set_postfix({
                'L2': f'{lossMSE.item():.3f}',
                'KL': f'{lossKL.item():.3f}'
            })
            self.current_step+=1
            pbar.update(1)

            #------------Validate-------------
            if self.current_step % self.val_step == 0:
                self._validate_step(self.current_step)

        self.writer.close()
        print('Training finished')
            