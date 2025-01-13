import pandas as pd
import torch
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from rembg import remove
from PIL import Image
import numpy as np
import io
import torchvision.transforms as transforms
class Update_method(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        
        self.device = next(self.model.module.parameters()).device
        self.args = args
        self.num_sample = self.args.num_sim
        self.embedding = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[:-1])))
        self.style_layer1 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[:-5])))
        self.entire_encode1 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[-5:-1])))
        self.style_layer2 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[:-4])))
        self.entire_encode2 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[-4:-1])))
        self.style_layer3 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[:-3])))
        self.entire_encode3 = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[-3:-1])))
        self.anchors, self.pseudo_labels = self.generate_anchor()
        self.eps = args.alpha_cap
        self.num_sample = args.num_sim
        
    def generate_anchor(self):
        anchors = []
        num_classes = self.model.module.fc.out_features
        pseudo_labels = []
        for class_idx in tqdm.tqdm(range(num_classes)):
            target_vector = torch.zeros((1, num_classes)).to(self.device)
            target_vector[0, class_idx] = 1.0
            pseudo_labels.append(target_vector)
            
            input_embedding = torch.randn((1, self.model.module.fc.in_features)).to(self.device)
            input_embedding.requires_grad = True
            
            optimizer = torch.optim.Adam([input_embedding], lr=0.0001)
            
            for _ in range(self.args.epoch_anchors):
                optimizer.zero_grad()
                output = self.model.module.fc(input_embedding)
                loss = -torch.nn.functional.log_softmax(output, dim=1)[0, class_idx]
                loss.backward()
                optimizer.step()
            anchors.append(input_embedding.detach().clone())
            # self.log_anchor(target_vector, anchors[class_idx])
        return anchors, pseudo_labels
    
    def get_embedding(self, x):
        return self.embedding(x)
    
    def get_style(self, x):
        mu = x.mean(dim = [2, 3], keepdim = False)
        var = x.var(dim = [2, 3], keepdim = False)
        var = (var + 1e-6).sqrt()
        # mu, var = mu.detach(), var.detach()
        return mu, var
    
    def sqrtvar(self, x):
        t = (x.var(dim = 0, keepdim = True) + 1e-6).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t
    
    def  _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * 0.1
        return mu + epsilon * std 

    def compute_ulb_grads(self, ulb_embed_batch, pseudo_label_batch):
        var_emb = ulb_embed_batch.requires_grad_(True).to(self.device)
        output = self.model.module.fc(var_emb) 
        loss = F.cross_entropy(output, pseudo_label_batch, reduction='mean')  
        grads = torch.autograd.grad(loss, var_emb, retain_graph=True)[0]  
        del loss, var_emb, output
        return grads

    def calculate_optimum_alpha(self, eps, lb_embedding_batch, ulb_embedding_batch, ulb_grads_batch):
        z = lb_embedding_batch - ulb_embedding_batch  # Shape: [batch_size, embedding_dim]
        z_norm = z.norm(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        ulb_grads_norm = ulb_grads_batch.norm(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        alpha = (eps * z_norm / (ulb_grads_norm + 1e-8)).repeat(1, z.size(1)) * ulb_grads_batch / (z + 1e-8)

        return alpha
    
    def mix_feature(self, ulb_embed_batch, anchor_batch, alpha_batch):
        # ulb_embed_batch: tensor of shape [batch_size, embedding_dim]
        # anchor_batch: tensor of shape [batch_size, embedding_dim]
        # alpha_batch: tensor of shape [batch_size, embedding_dim]
        
        # Mixing the features using batch-wise tensor operations
        mixed_features = (1 - alpha_batch) * ulb_embed_batch + alpha_batch * anchor_batch
        # Shape of mixed_features: [batch_size, embedding_dim]
        
        return mixed_features
    
    def auxiliary_sample(self, x, eps, anchors, num_sample):
        B = x.shape[0]
        
    def filter_sample(self, x, eps, anchors, num_sample):
        # x: tensor of shape [batch_size, channels, height, width]
        # eps: perturbation parameter
        # anchors: list of anchor embeddings, each of shape [1, embedding_dim]
        # num_sample: number of simulations (threshold)

        B = x.shape[0]  # Batch size
        num_classes = self.model.module.fc.out_features
    
        # Get embeddings from the model's embedding layer
        if self.args.model == "vitbase_timm":
            ulbs_embed = self.embedding(x)[:, 0, :]  # Shape: [batch_size, embedding_dim]
        else:
            ulbs_embed = self.embedding(x).detach()  # Shape: [batch_size, embedding_dim]

        # Get the predictions for the original batch of samples
        output = self.model(x).detach()
        prob_output = output.softmax(1)
        predictions = torch.argmax(output, dim=1)
    #-----------------------------------------REMOVE BACKGROUND---------------------------------    
        # mean = torch.tensor([0.1307, 0.1307, 0.0]).view(-1, 1, 1).to(x.device)
        # std = torch.tensor([0.3081, 0.3081, 0.3081]).view(-1, 1, 1).to(x.device)
        # processed_images = []
        # for i in range(B):
        #     tensor_image = x[i]*std + mean #(C, H, W)
        #     transform_to_pil = transforms.ToPILImage()
        #     pil_image = transform_to_pil(tensor_image)
        #     images_bytes = io.BytesIO()
        #     pil_image.save(images_bytes, format = "PNG")
        #     images_bytes = images_bytes.getvalue()
        #     output_data = remove(images_bytes)
        #     output_image = Image.open(io.BytesIO(output_data))
        #     output_np = np.array(output_image) #(H, W, 4)
        #     rgb, alpha = output_np[:, :, :3], output_np[:, :, 3]
        #     background_mask = alpha < 50
        #     rgb[background_mask] = [255, 255, 255]
        #     output_tensor = torch.from_numpy(rgb).permute(2, 0, 1) #(3, H, W)
        #     processed_images.append(output_tensor)
        # x_new = torch.stack(processed_images) #(B, 3, H, W)
        # x_new = x_new.to(torch.float32).to("cuda:0")
        # mean = torch.tensor([0.1307, 0.1307, 0.0]).view(1, -1, 1, 1).to(x_new.device)
        # std = torch.tensor([0.3081, 0.3081, 0.3081]).view(1, -1, 1, 1).to(x_new.device)
        # # import pdb; pdb.set_trace()
        # x_new = (x_new - mean)/std

        # with torch.no_grad():
        #     output_new = self.model(x_new).detach()

        # prob_new = output_new.softmax(1)
        # # import pdb; pdb.set_trace()
        # plpd_bg = torch.gather(prob_output, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(prob_new, dim = 1, index = predictions.reshape(-1, 1))
        # plpd_bg = plpd_bg.reshape(-1).cpu()
        # all_below_threshold2 = torch.where(plpd_bg < eps)
    #-------------------------------------------------------------------------------------------
    #-----------------------------------------COLOR REMOVING---------------------------------
        # weights =  torch.tensor([0.2989, 0.5870, 0.1140]).view(1, -1, 1, 1).to(x.device)
        # x_gray = (x * weights).sum(dim = 1)
        # x_gray = x_gray.unsqueeze(1)
        # x_gray = x_gray.repeat(1, 3, 1, 1)
        # with torch.no_grad():
        #     output_gray = self.model(x_gray).detach()
        
        # prob_gray = output_gray.softmax(1)
        # plpd_gray = torch.gather(prob_output, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(prob_gray, dim = 1, index = predictions.reshape(-1, 1))
        # plpd_gray = plpd_gray.reshape(-1).cpu()
        # all_below_threshold3 = torch.where(plpd_gray < eps)
    #--------------------------------------------------------------------------------------
    #---------------------------------------FEATURE MIXING--------------------------------
        # # Update label counts for the original predictions
        # for i in range(B):
        #     labels_count[i][predictions[i]] += 1

        # # Iterate over each class to compute gradients and perform feature mixing for the whole batch
        # for j in range(num_classes):
        #     # Anchor and pseudo label for the current class
        #     anchor = anchors[j].repeat(B, 1)  # Shape: [batch_size, embedding_dim]
        #     label = self.pseudo_labels[j].repeat(B, 1)  # Shape: [batch_size, num_classes]
        #     # print("Check at line 110")
        #     # print(ulbs_embed.shape)
        #     # print(label.shape)
        #     ulbs_embed = ulbs_embed.view(ulbs_embed.size(0), -1)

        #     # Compute gradients for the whole batch
        #     grads = self.compute_ulb_grads(ulbs_embed, label)  # Shape: [batch_size, embedding_dim]

        #     # Calculate the optimal alpha for each sample in the batch
        #     alpha = self.calculate_optimum_alpha(eps, anchor, ulbs_embed, grads)  # Shape: [batch_size, embedding_dim]

        #     # Mix features for each sample in the batch
        #     feature_mix = self.mix_feature(ulbs_embed, anchor, alpha)  # Shape: [batch_size, embedding_dim]

        #     # Compute predictions for the mixed features
        #     mixed_output = self.model.module.fc(feature_mix)  # Shape: [batch_size, num_classes]
        #     pred = torch.argmax(mixed_output, dim=1)  # Shape: [batch_size]

        #     # Update label counts for each sample in the batch
        #     for i in range(B):
        #         labels_count[i][pred[i]] += 1

        # # Determine which samples are below the threshold for number of simulations
        # below_threshold = labels_count < num_sample
        # all_below_threshold = torch.all(below_threshold, dim=1)  # Shape: [batch_size]
        # import pdb; pdb.set_trace()
        # #--------------------------------------------------------------------------------------
        #---------------------------------------FEATURE MIXING VER 2--------------------------------
        # # Update label counts for the original predictions
        # remove = []
        # eps1 = 0.99
        # while (len(remove) < eps * B):
        #     labels_count = torch.zeros((B, num_classes))
        #     for i in range(B):
        #         labels_count[i][predictions[i]] += 1
        # # Iterate over each class to compute gradients and perform feature mixing for the whole batch
        #     for j in range(num_classes):
        #     # Anchor and pseudo label for the current class
        #         anchor = anchors[j].repeat(B, 1)  # Shape: [batch_size, embedding_dim]
        #         label = self.pseudo_labels[j].repeat(B, 1)  # Shape: [batch_size, num_classes]
        #     # print("Check at line 110")
        #     # print(ulbs_embed.shape)
        #     # print(label.shape)
        #         ulbs_embed = ulbs_embed.view(ulbs_embed.size(0), -1)

        #         # Compute gradients for the whole batch
        #         grads = self.compute_ulb_grads(ulbs_embed, label)  # Shape: [batch_size, embedding_dim]

        #         # Calculate the optimal alpha for each sample in the batch
        #         alpha = self.calculate_optimum_alpha(eps, anchor, ulbs_embed, grads)  # Shape: [batch_size, embedding_dim]

        #         # Mix features for each sample in the batch
        #         feature_mix = self.mix_feature(ulbs_embed, anchor, eps1)  # Shape: [batch_size, embedding_dim]

        #         # Compute predictions for the mixed features
        #         mixed_output = self.model.module.fc(feature_mix).detach()  # Shape: [batch_size, num_classes]
        #         pred = torch.argmax(mixed_output, dim=1)  # Shape: [batch_size]
                
        #     # Update label counts for each sample in the batch
        #         for i in range(B):
        #             labels_count[i][pred[i]] += 1
            
        # # Determine which samples are below the threshold for number of simulations
        #     below_threshold = labels_count < num_sample
        #     all_below_threshold = torch.all(below_threshold, dim=1)  # Shape: [batch_size]
        #     indices = torch.nonzero(~all_below_threshold, as_tuple=True)[0].tolist()
        #     for idxs in indices:
        #         if idxs not in remove:
        #             remove.append(idxs)
        #     eps1 = eps1 - 0.01
        # final_threshold = torch.ones(B, dtype = torch.bool)
        # final_threshold[remove] = False
        #--------------------------------------------------------------------------------------
        #---------------------------------------STYLE TRANSFERRING--------------------------------
        # with torch.no_grad():
        #     style_feature = self.style_layer1(x).detach()
        # dims = style_feature.shape[1]
        # mu, sig = self.get_style(style_feature)
        # list_mu = []
        # list_sig = []
        # generator = torch.distributions.MultivariateNormal(loc = torch.zeros(dims), covariance_matrix=torch.eye(dims))
        # for i in range(B):
        #     noise_mu = generator.sample()
        #     noise_sig = generator.sample()
        #     noise_mu = torch.reshape(noise_mu, (mu[i].shape[0], 1, 1)).to("cuda:0")
        #     noise_sig = torch.reshape(noise_sig, (sig[i].shape[0], 1, 1)).to("cuda:0")
        #     new_mu = noise_mu
        #     new_sig = noise_sig 
        #     # new_mu = torch.zeros(dims)
        #     # new_sig = noise_sig
        #     list_mu.append(new_mu)
        #     list_sig.append(new_sig)
        # final_mu = torch.vstack(list_mu).float()
        # final_sig = torch.vstack(list_sig).float()
        # final_mu = torch.reshape(final_mu, (style_feature.shape[0], style_feature.shape[1], 1, 1)).to("cuda")
        # final_sig = torch.reshape(final_sig, (style_feature.shape[0], style_feature.shape[1], 1, 1)).to("cuda")
        # norm = (style_feature - mu)/sig
        # x_trans = norm * final_sig + final_mu

        # with torch.no_grad():
        #     out1 = self.entire_encode1(x_trans).detach()
        # out1 = out1.view(out1.shape[0],-1)
        # with torch.no_grad():
        #     out = self.model.module.fc(out1).detach()
        # prob_trans = out.softmax(1)
        # plpd_trans = (torch.gather(prob_output, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(prob_trans, dim = 1, index = predictions.reshape(-1, 1)))    
        # plpd_trans = plpd_trans.reshape(-1).cpu()
        # all_below_threshold4 = torch.where(plpd_trans < eps)
        #---------------------------------------STYLE TRANSFERRING DSU--------------------------------
        with torch.no_grad():
            style_feature = self.style_layer1(x).detach()
        dims = style_feature.shape[1]
        mean, std = self.get_style(style_feature)
        
        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)
        # import pdb; pdb.set_trace()
        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std) 
        # import pdb; pdb.set_trace()
        norms = (style_feature - mean.reshape(style_feature.shape[0],style_feature.shape[1],1,1))/std.reshape(style_feature.shape[0],style_feature.shape[1],1,1)
        x_trans = norms * gamma.reshape(style_feature.shape[0],style_feature.shape[1],1,1) + beta.reshape(style_feature.shape[0],style_feature.shape[1],1,1)

        with torch.no_grad():
            out1 = self.entire_encode1(x_trans).detach()
        out1 = out1.view(out1.shape[0],-1)
        with torch.no_grad():
            out = self.model.module.fc(out1).detach()
        prob_trans = out.softmax(1)
        plpd_trans = (torch.gather(prob_output, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(prob_trans, dim = 1, index = predictions.reshape(-1, 1)))    
        plpd_trans = plpd_trans.reshape(-1).cpu()
        all_below_threshold4 = torch.where(plpd_trans < eps)
        #---------------------------------------STYLE TRANSFERRING CSU--------------------------------
        # al = 0.1
        # beta = torch.distributions.Beta(al, al)
        # with torch.no_grad():
        #     style_feature = self.style_layer1(x).detach()
        # B1, C1 = style_feature.size(0), style_feature.size(1)
        # mu, sig = self.get_style(style_feature)
        
        # norms = (style_feature - mu.reshape(style_feature.shape[0],style_feature.shape[1],1,1))/sig.reshape(style_feature.shape[0],style_feature.shape[1],1,1)

        # factor = beta.sample((B1, 1, 1, 1)).to(self.device)

        # mu_squeeze = torch.squeeze(mu)
        # mean_mu = torch.mean(mu_squeeze, dim = 0, keepdim=True)
        # correlation_mu = (mu_squeeze - mean_mu).T @ (mu_squeeze - mean_mu)/B1

        # sig_squeeze = torch.squeeze(sig)
        # mean_sig = torch.mean(sig_squeeze, dim = 0, keepdim=True)
        # correlation_sig = (sig_squeeze - mean_sig).T @ (sig_squeeze - mean_sig)/B1

        # with torch.no_grad():
        #     try:
        #         _, mu_eng_vector = torch.linalg.eigh(C1*correlation_mu+self.eps*torch.eye(C1, device=x.device))
        #     except:
        #         mu_eng_vector = torch.eye(C1, device=x.device)
        # # import pdb; pdb.set_trace()
        #     if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
        #         mu_eng_vector = torch.eye(C1, device=x.device)

        #     try:
        #         _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C1, device=x.device))
        #     except:
        #         sig_eng_vector = torch.eye(C1, device=x.device)
            
        #     if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
        #         sig_eng_vector = torch.eye(C1, device=x.device)

        # mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        # sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        # gaussian_mu = (torch.randn(B1, 1, C1, device=x.device) @ mu_corr_matrix)
        # gaussian_mu = torch.reshape(gaussian_mu, (B1, C1, 1, 1))

        # gaussian_sig = (torch.randn(B1, 1, C1, device=x.device) @ sig_corr_matrix)
        # gaussian_sig = torch.reshape(gaussian_sig, (B1, C1, 1, 1))    

        # mu_mix = mu + factor * gaussian_mu
        # sig_mix = sig + factor * gaussian_sig

        # x_trans = norms * sig_mix + mu_mix
        # with torch.no_grad():
        #     out1 = self.entire_encode1(x_trans).detach()
        # out1 = out1.view(out1.shape[0],-1)
        # with torch.no_grad():
        #     out = self.model.module.fc(out1).detach()
        # prob_trans = out.softmax(1)
        # plpd_trans = (torch.gather(prob_output, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(prob_trans, dim = 1, index = predictions.reshape(-1, 1)))    
        # plpd_trans = plpd_trans.reshape(-1).cpu()
        # all_below_threshold4 = torch.where(plpd_trans < eps)
        # import pdb; pdb.set_trace()
        return all_below_threshold4, plpd_trans

    def log_anchor(self, label, anchor):
        with open(self.args.save_path, "a") as f:
            f.write("Label: \n")
            f.write(str(label))
            f.write("Anchor: \n")
            f.write(str(anchor))