#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse, json
from tensorboardX import SummaryWriter
import scipy.stats

from utils.torch_utils import *
from utils.utils import *
from utils.models import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class NewsDataset(Dataset):
    def __init__(self, sequences, charmap):
        
        self.charmap = charmap    
        self.raw_sequences = sequences
        self.sequeneces_num = [[self.charmap[char_] for char_ in sequence] for sequence in sequences]
        self.sequences = np.eye(len(self.charmap))[self.sequeneces_num]
        
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return item
    
    def __len__(self):
        return len(self.sequences)


class WGAN_LangGP():
    def __init__(self, args):
        self.input_rate = args.input_rate
        self.output_rate = args.output_rate

        self.lr = args.lr
        self.G_hidden = args.G_hidden
        self.D_hidden = args.D_hidden
        self.noise_hidden = args.noise_hidden
        self.batch_size = args.batch_size
        
        self.gumbel = args.gumbel == 1
        self.n_epochs = args.num_epochs
        self.seq_len = args.seq_len
        self.d_steps = args.d_steps
        self.lamda = args.lamda 
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.run_name = args.run_name
        self.time = args.time if args.time != "" else int(time.time()) 

        self.retrain = args.retrain == 1
        self.iteration = args.iteration
        self.load_dir = args.load_dir
        
        self.load_data(self.train_dir, self.val_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def get_init_(self):
        
        self.checkpoint_dir = './checkpoint/{}/{}/'.format(self.run_name, self.time)
        self.model_path = os.path.join(self.checkpoint_dir, "model_path")
        self.sample_dir = os.path.join(self.checkpoint_dir, "samples")

        if not os.path.exists(self.model_path): 
            os.makedirs(self.model_path)
        if not os.path.exists(self.sample_dir): 
            os.makedirs(self.sample_dir)

        self.write = SummaryWriter(self.checkpoint_dir)
        
        self.log_param()

    def log_param(self):
        log_file_json = os.path.join(self.checkpoint_dir, "log.json")
        log_dict = dict()

        log_dict["input_rate"] = self.input_rate
        log_dict["output_rate"] = self.output_rate

        log_dict["lr"] = self.lr
        log_dict["G_hidden"] = self.G_hidden
        log_dict["D_hidden"] = self.D_hidden
        log_dict["noise_hidden"] = self.noise_hidden
        log_dict["n_epochs"] = self.n_epochs
        log_dict["batch_size"] = self.batch_size
        log_dict["seq_len"] = self.seq_len
        log_dict["d_steps"] = self.d_steps
        log_dict["train_dir"] = self.train_dir
        log_dict["val_dir"] = self.val_dir
        log_dict["lamda"] = self.lamda
        log_dict["checkpoint_dir"] = self.checkpoint_dir
        log_dict["sample_dir"] = self.sample_dir
        log_dict["charmap"] = self.charmap
        log_dict["inv_charmap"] = self.inv_charmap
        log_dict["gumbel"] = self.gumbel

        log_dict["retrain"] = self.retrain
        log_dict["iteration"] = self.iteration
        log_dict["load_dir"] = self.load_dir

        with open(log_file_json, 'w') as log_file:
            json.dump(log_dict, log_file, indent=4 ) 


    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.G_hidden, self.input_rate, self.output_rate, self.gumbel)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.D_hidden, self.input_rate, self.output_rate)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def load_data(self, train_datadir, val_datadir, max_examples=1e6):
        self.contents, self.charmap, self.inv_charmap = utils.language_helpers.load_dataset_2(data_dir=train_datadir)
        self.data = np.array(self.contents)[:,0].tolist()
        self.labels = np.array(self.contents)[:,1].tolist()

        self.val_contents, _, _ = utils.language_helpers.load_dataset_2(data_dir=val_datadir)
        self.val_data = np.array(self.val_contents)[:,0].tolist()
        self.val_labels = np.array(self.val_contents)[:,1].tolist()

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.model_path, "G_weights_{}.pth".format(epoch)))
        torch.save(self.D.state_dict(), os.path.join(self.model_path, "D_weights_{}.pth".format(epoch)))

    def load_model(self, model_dir, iteration=None):
        list_G = glob.glob(os.path.join(model_dir, "G*.pth"))
        list_D = glob.glob(os.path.join(model_dir,  "D*.pth"))
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        if iteration is None:
            print("Loading most recently saved...")
            G_file = max(list_G, key=os.path.getctime)
            D_file = max(list_D, key=os.path.getctime)
        else:
            G_file = "G_weights_{}.pth".format(iteration)
            D_file = "D_weights_{}.pth".format(iteration)

            directory = "./checkpoint"
            G_file = os.path.join(directory, G_file)
            D_file = os.path.join(directory, D_file)

            print("G_file: {}".format(G_file))
            print("D_file: {}".format(D_file))

        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))


    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.contiguous().view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        #gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return self.lamda * ((gradients_norm - 1) ** 2).mean()

    def disc_train_iteration(self, real_data):
        self.D_optimizer.zero_grad()

        fake_data = self.sample_generator(self.batch_size)
        d_fake_pred = self.D(fake_data)
        d_fake_err = d_fake_pred.mean()

        d_real_pred = self.D(real_data)
        d_real_err = d_real_pred.mean()

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)

        d_err = d_fake_err - d_real_err + gradient_penalty
        d_err.backward()
        self.D_optimizer.step()

        return d_fake_err.data, d_real_err.data, gradient_penalty.data

    def sample_generator(self, num_sample):
        z_input = Variable(torch.randn(num_sample, 128))
        if self.use_cuda: 
            z_input = z_input.cuda()
        generated_data = self.G(z_input)
        return generated_data


    def gen_train_iteration(self):
        self.G.zero_grad()
        z_input = to_var(torch.randn(self.batch_size, 128))
        g_fake_data = self.G(z_input)
        dg_fake_pred = self.D(g_fake_data)
        g_err = -torch.mean(dg_fake_pred)
        g_err.backward()
        self.G_optimizer.step()
        return g_err


    def train_model(self):
        self.get_init_()

        if self.retrain:
            self.load_model(self.load_dir, self.iteration)
            losses_f = open(self.checkpoint_dir + "losses_retrain.txt", 'a+')
        else:
            losses_f = open(self.checkpoint_dir + "losses.txt", 'a+')

        # dataloader
        train_dataset = NewsDataset(self.data, self.charmap)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        counter = 0
        import time
        for epoch in range(self.n_epochs):
            
            n_batches = int(len(self.data)/self.batch_size)
            print('In epoch {}, n_batches is {}'.format(epoch, n_batches))

            start_time = time.time()
            for batch_data in train_dataloader:
                real_data = to_var(batch_data.type(torch.FloatTensor))

                d_fake_err, d_real_err, gradient_penalty = self.disc_train_iteration(real_data)

                # Append things for logging
                d_fake_np, d_real_np, gp_np = d_fake_err.cpu().numpy(), d_real_err.cpu().numpy(), gradient_penalty.cpu().numpy()
                
                d_loss = d_fake_np - d_real_np + gp_np
                w_dist = d_real_np - d_fake_np
            
                if counter % self.d_steps == 0:
                    g_err = self.gen_train_iteration()
                    g_loss = (g_err.data).cpu().numpy()

                if counter % 10 == 9:
                    summary_str = 'Iteration [{}] - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                        .format(counter, ((d_fake_err - d_real_err + gradient_penalty)).cpu().numpy(),
                        (d_fake_err).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                    print(summary_str)
                    losses_f.write(summary_str)
                    
                self.write.add_scalar('G_losses', g_loss, counter)
                self.write.add_scalar('D_losses', d_loss, counter)
                self.write.add_scalar('W_dist', w_dist, counter)
                self.write.add_scalar('grad_penalties', gp_np, counter)
                self.write.add_scalar('d_fake_losses', d_fake_np, counter)
                self.write.add_scalar('d_real_losses', d_real_np, counter)
                counter += 1

            # save model and generate samples
            self.save_model(epoch+1)

            # evaluate model
            mmd_tra, mmd_val, act_entropy_tra, act_entropy_val = self.evaluate_model(20000, epoch)
            self.write.add_scalar('mmd_tra', mmd_tra, epoch)
            self.write.add_scalar('mmd_val', mmd_val, epoch)
            self.write.add_scalar('act_entropy_tra', act_entropy_tra, epoch)
            self.write.add_scalar('act_entropy_val', act_entropy_val, epoch)
            print("mmd_tra:{:.3f}, mmd_val:{:.3f}, act_entropy_tra:{:.3f}, act_entropy_val:{:.3f}".format(mmd_tra, mmd_val, act_entropy_tra, act_entropy_val))
            
            end_time = time.time()
            print("-->>> epoch: {}; Time: {:.6f}".format(epoch, end_time-start_time))


    def evaluate_model(self, num=10000, epoch=1):
        from post_evaluate.mmd.util_evaluate import mmd_2
        from post_evaluate.activity.util import Activity_predict
        n_batch = int(num/512)
        
        fake_seqs = []
        for i in range(n_batch):
            sequences = self.sample_generator(512).to("cpu").tolist()
            sequences = [decode_one_seq_2(seq, self.inv_charmap) for seq in sequences]
            fake_seqs += sequences
        
        indexes = list(set(np.random.randint(0, len(self.data), num).tolist()))
        real_seqs = np.array(self.data)[indexes].tolist()
        
        args_dict={    
                    "mer": 3,
                    "embedding": "spectrum", 
                    "max_length": 250,
                    "batch_size": 128,
                    
                    "mode": "count",
                    "normalize": True,
                    "kernel": "linear",
                    "return_pvalue": False
                  }
        
        mmd_value_tra = mmd_2(args_dict, real_seqs, fake_seqs)[0]
        mmd_value_val = mmd_2(args_dict, self.val_data, fake_seqs)[0]
        
        # activity prediction by DeepSTARR
        real_seqs_act = Activity_predict(real_seqs)
        val_seqs_act = Activity_predict(self.val_data)
        fake_seqs_act = Activity_predict(fake_seqs)

        def Activity_Probability(seq_activity):
            frag_num = 30
            activity_range_count = [0]*(frag_num+1)
            
            sub_spaces = np.linspace(-5, 5, frag_num)
            
            for activity in seq_activity:
                if activity < -5.0:
                    activity_range_count[0] += 1
                    continue
                elif activity > 5.0:
                    activity_range_count[frag_num] += 1
                    continue

                for index in range(len(sub_spaces)-1):
                    if activity >= sub_spaces[index] and activity < sub_spaces[index+1]:
                        activity_range_count[index+1] += 1
                        break
            activity_pro = np.array(activity_range_count) / sum(activity_range_count)
            return activity_pro

        def Entropy(real_probability, fake_probability):
            M = (real_probability + fake_probability) / 2
            entropy = 0.5*scipy.stats.entropy(real_probability, M, base=2) + 0.5*scipy.stats.entropy(fake_probability, M, base=2)
            return entropy

        real_probability = Activity_Probability(real_seqs_act[0])
        val_probability = Activity_Probability(val_seqs_act[0])
        fake_probability = Activity_Probability(fake_seqs_act[0])

        entropy_tra = Entropy(real_probability, fake_probability)
        entropy_val = Entropy(val_probability, fake_probability)

        with open(os.path.join(self.sample_dir, "sampled_{}.txt".format(epoch)), 'w+') as f:
            contents = list(zip(fake_seqs, fake_seqs_act[0]))
            for item in contents:
                content = "{}\t{}\n".format(item[0], item[1])
                f.write(content)
                f.flush()

        return mmd_value_tra, mmd_value_val, entropy_tra, entropy_val
        

    def sample(self, epoch):
        z = to_var(torch.randn(self.batch_size, 128))
        self.G.eval()
        torch_seqs = self.G(z)
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap)+"\n" for seq in seqs]
        with open(os.path.join(self.sample_dir, "sampled_{}.txt".format(epoch)), 'w+') as f:
            f.writelines(decoded_seqs)
        self.G.train()


def main():
    parser = argparse.ArgumentParser(description='WGAN.')
    parser.add_argument("--input_rate",     default=0.7, type=float)
    parser.add_argument("--output_rate",    default=0.3, type=float)

    parser.add_argument("--lr",             default=0.00005, type=float)
    parser.add_argument("--batch_size",     default=64, type=int)
    parser.add_argument("--noise_hidden",   default=128, type=int)
    parser.add_argument("--num_epochs",     default=250, type=int)
    parser.add_argument("--seq_len",        default=249, type=int)
    parser.add_argument("--train_dir",      default="./data/real_Sequence_train.txt", type=str)
    parser.add_argument("--val_dir",        default="./data/real_Sequence_val.txt", type=str)
    parser.add_argument("--run_name",       default= "Vanilla-GAN", type=str)
    parser.add_argument("--G_hidden",       default= 512, type=int)
    parser.add_argument("--D_hidden",       default= 256, type=int)
    parser.add_argument("--d_steps",        default=5, type=int)
    parser.add_argument("--lamda",          default=8, type=int)
    parser.add_argument("--gumbel",         default=0, type=int)
# df52d8d
    parser.add_argument("--retrain",        default=0, type=int)
    parser.add_argument("--load_dir",       default="", help="Load pretrained GAN checkpoints")
    parser.add_argument("--time",           default="", type=str)
    parser.add_argument("--iteration",      default=0, type=int)
    parser.add_argument("--preds_cutoff",   default=0.7, type=float)
    args = parser.parse_args()

    model = WGAN_LangGP(args)
    model.train_model()

if __name__ == '__main__':
    main()


