from model import Generator_3 as Generator
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from torch.utils.tensorboard import SummaryWriter

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy
import utils

# use demo data for simplicity
# make your own validation set as needed
validation_pt = pickle.load(open('assets/demo.pkl', "rb"))

class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, val_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.val_loader = val_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        

        # Build the model and tensorboard.
        self.build_model()

        self.train_writer = SummaryWriter(os.path.join(self.log_dir, "Train"))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, "Validation"))

            
    def build_model(self):        
        self.G = Generator(self.hparams)
        
        self.Interp = InterpLnr(self.hparams)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        
        self.G.to(self.device)
        self.Interp.to(self.device)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)
        
        
    def save_checkpoint(self, model, optimizer, filepath, iteration):
        print("Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath))
        fp = os.path.join(filepath.split('/')[0],filepath.split('/')[1])
        if os.path.isdir(fp) == False:
            os.mkdir(fp)
        torch.save({'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, filepath)

    def load_checkpoint(self, checkpoint_path, model, optimizer):
        if os.path.isfile(checkpoint_path):
            print("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(self.device))
            model.load_state_dict(checkpoint_dict['state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            iteration = checkpoint_dict['iteration']
            print("Loaded checkpoint '{}' from iteration {}" .format(
                checkpoint_path, iteration))
        else:
            iteration = 0
        return model, optimizer, iteration
        
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
#=====================================================================================================================
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        val_loader = self.val_loader
        
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        val_iter = iter(val_loader)
        # Start training from scratch or resume training.
        checkpoint_path = os.path.join(self.log_dir,'drum_loop_split.pt')
# '/home/pc2752/share/sve/logs/pytorch/autovc_vctk/autovc.pt'
        self.G, self.g_optimizer, self.iteration = self.load_checkpoint(checkpoint_path=checkpoint_path, model=self.G, optimizer=self.g_optimizer)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
        
        # Print logs in specified order
        count = 0 

        recon_loss = 0
        content_loss = 0

        val_recon_loss = 0
        val_content_loss = 0

        epoch = int(self.iteration/self.log_step)
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for it in range(self.iteration, self.num_iters):
            count+=1

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, len_org = next(data_iter)

            # len_org = torch.from_numpy(np.repeat(x_real_org.shape[-1], x_real_org.shape[0])).type(torch.FloatTensor)
            # import pdb;pdb.set_trace()
            x_real_org = x_real_org.to(self.device).transpose(1,2)
            len_org = len_org.to(self.device)
            
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_intrp = self.Interp(x_real_org, len_org)

            x_identic, code_real = self.G(x_intrp, x_real_org)

            g_loss_id = F.mse_loss(x_real_org, x_identic) 

            # _, code_reconst = self.G(x_identic, x_identic)

            # g_loss_cd = F.l1_loss(code_real, code_reconst)
           
            # Backward and optimize.
            g_loss = g_loss_id

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            recon_loss+=g_loss_id.item()
            # content_loss+=g_loss_cd.item()

            self.train_writer.add_scalar('Reconstruction',g_loss_id.item(), it+1)
            # self.train_writer.add_scalar('Content',g_loss_cd.item(), it+1)
            # self.train_writer.add_scalar('Total_Loss',g_loss.item(), it+1)
            

            # =================================================================================== #
            #                               Validation                                            #
            # =================================================================================== #

            self.G = self.G.eval()

            try:
                x_real_org, len_org = next(val_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, len_org = next(val_iter)

            # len_org = torch.from_numpy(np.repeat(x_real_org.shape[-1], x_real_org.shape[0])).type(torch.FloatTensor)
            x_real_org = x_real_org.to(self.device).transpose(1,2)
            len_org = len_org.to(self.device)

            x_intrp = self.Interp(x_real_org, len_org)

            x_identic, code_real = self.G(x_intrp, x_real_org)
            g_loss_id = F.mse_loss(x_real_org, x_identic) 

            # _, code_reconst = self.G(x_identic, x_identic)

            # g_loss_cd = F.l1_loss(code_real, code_reconst)
           
            # Backward and optimize.
            g_loss_val = g_loss_id 



            # Logging.
            val_recon_loss+=g_loss_id.item()
            # val_content_loss+=g_loss_cd.item()

            self.val_writer.add_scalar('Reconstruction',g_loss_id.item(), it+1)
            # self.val_writer.add_scalar('Content',g_loss_cd.item(), it+1)
            # self.val_writer.add_scalar('Total_Loss',g_loss_val.item(), it+1)

            utils.progress(count, self.log_step, "Loss: {} Val: {}".format(g_loss.item(),g_loss_val.item()))

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (it+1) % self.log_step == 0:
                et = time.time() - start_time
                count = 0
                epoch+=1
                loss = {}
                loss['Recon Loss'] = recon_loss/self.log_step
                # loss['Content Loss'] = content_loss/self.log_step

                loss['Val Recon Loss'] = val_recon_loss/self.log_step
                # loss['Val Content Loss'] = val_content_loss/self.log_step

                et = str(datetime.timedelta(seconds=et))[:-7]
                self.save_checkpoint(model=self.G, optimizer=self.g_optimizer,\
                    filepath=checkpoint_path, iteration=it+1)

                log = "Elapsed [{}], Iteration [{}/{}]".format(et, it, self.num_iters)

                for tag in loss.keys():
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

                recon_loss = 0
                postnet_recon_loss = 0
                content_loss = 0

                val_recon_loss = 0
                val_postnet_recon_loss = 0
                val_content_loss = 0         
