# --------------------------------------------------------
# Our codes are partly based on DeepInversion and SinGAN paper, we thank the authors
# --------------------------------------------------------
from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os
import cv2
import glob
import pdb
import torchvision.models as torch_models
#import model.net as net
from model.net import *
from imgaug import augmenters as iaa
import imgaug as ia
import PIL
from torchvision import transforms
import scipy
from scipy.ndimage.filters import gaussian_filter
import lpips


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2



def get_reference_images5(i_sample, folder_path, image_size, batch_size):
    image_bank = np.zeros((batch_size, 3, image_size, image_size))

    file_path = i_sample

    img = cv2.imread(file_path)
    img = np.float32(cv2.resize(img, (image_size, image_size))) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean)
    mean = np.reshape(mean, (3, 1, 1))
    std = np.array(std)
    std = np.reshape(std, (3, 1, 1))
    img = np.transpose(img, (2, 0, 1))
    img = img[[2, 1, 0], :, :]
    img = (img - mean) / std

    for i in range(batch_size):
        image_bank[i, :, :, :] = img

    return image_bank

def get_label_images(target_prefix, i_sample, image_size, batch_size):
    file_name = os.path.basename(i_sample)
    image_bank_ref = np.zeros((batch_size,image_size,image_size))
    image_bank_target = np.zeros((batch_size,image_size,image_size))
    
    img = cv2.imread('./labels/gt_' + file_name,0)
    img = np.float32(cv2.resize(img, (image_size, image_size)))
    img = np.where((img > 125), 1,0).astype('uint8')
    
    img2 = cv2.imread('./labels/target' + target_prefix + '_' + file_name,0)
    img2 = np.float32(cv2.resize(img2, (image_size, image_size)))
    img2 = np.where((img2 > 125), 1,0).astype('uint8')
    
    for i in range(batch_size):
            image_bank_ref[i, :, :] = img
            image_bank_target[i, :, :] = img2
            
    return image_bank_ref, image_bank_target

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


class MAGIC_Class(object):
    def __init__(self, bs=84,
                 net_teacher=None, net_D=None, path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 gpu_number=1,
                 vae=None,
                 vae_criterion=None):

        self.net_teacher = net_teacher
        self.net_D = net_D

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.start_noise = parameters["start_noise"]
            self.pre_w = parameters["pre_w"]
            self.save_prefix = parameters["save_prefix"]
            self.mode = parameters["mode"]
            self.target_prefix = parameters["target_prefix"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.store_best_images = False

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.save_every = 100
        self.criterion = criterion
        self.network_output_function = network_output_function
        self.gpunumber = gpu_number
        self.vae = vae
        self.vae_criterion = vae_criterion
        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.main_loss_multiplier = coefficients["main_loss_multiplier"]
        self.num_generations = 0
        self.final_data_path = final_data_path

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()


    def get_images(self, i_sample=None):

        print("get_images call")

        net_teacher = self.net_teacher
        save_every = self.save_every
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net_D = self.net_D

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        img_original = self.image_resolution

        data_type = torch.float
        inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device=device,dtype=data_type)

        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id == 0:
            skipfirst = False
        else:
            skipfirst = True

        starting_train_GAN = 5000
        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it == 0:
                iterations_per_layer = 3000
                reference_images = get_reference_images5(i_sample, folder_path='./input_images',
                                                         image_size=112, batch_size=self.bs)
                reference_images = torch.from_numpy(reference_images).to(device)
                reference_images = reference_images.type(torch.cuda.FloatTensor)
                
                
                reference_labels, input_labels = get_label_images(self.target_prefix, i_sample, image_size=112, batch_size=self.bs)
                reference_labels = torch.from_numpy(reference_labels).to(device)
                reference_labels = reference_labels.type(torch.cuda.FloatTensor)
                input_labels = torch.from_numpy(input_labels).to(device)
                input_labels = input_labels.type(torch.cuda.FloatTensor)

            else:

                iterations_per_layer = 2000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 18000
                reference_images = get_reference_images5(i_sample, folder_path='./input_images',
                                                         image_size=224, batch_size=self.bs)
                reference_images = torch.from_numpy(reference_images).to(device)
                reference_images = reference_images.type(torch.cuda.FloatTensor)
                
                reference_labels, input_labels = get_label_images(self.target_prefix, i_sample, image_size=224, batch_size=self.bs)
                reference_labels = torch.from_numpy(reference_labels).to(device)
                reference_labels = reference_labels.type(torch.cuda.FloatTensor)
                input_labels = torch.from_numpy(input_labels).to(device)
                input_labels = input_labels.type(torch.cuda.FloatTensor)


            if lr_it == 0 and skipfirst:
                continue

            if self.setting_id == 0:
                # multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
            elif self.setting_id == 1:
                # 2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
            elif self.setting_id == 2:
                # 20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps=1e-8)

            # optimizer D
            optimizerD = torch.optim.Adam(net_D.parameters(), lr=0.0005, betas=(0.5, 0.999))
            schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=0.1)
            optimizerVae = torch.optim.Adam(self.vae.parameters(), lr=0.0005, betas=(0.5, 0.999))
            schedulerVae = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerVae, milestones=[1600], gamma=0.1)

            
            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
            
            outputs_ref = net_teacher(reference_images)
            outputs_ref = self.network_output_function(outputs_ref)
            print(outputs_ref)
            targets = outputs_ref.max(1).indices
            print(targets)
            
            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res != 1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)
                outputs = self.network_output_function(outputs)
                
                # my adain loss
                feature_extractor = get_features(net_teacher)
                feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=range(self.gpunumber)).cuda()

                feature_reference = feature_extractor(reference_images)
                feature_target = feature_extractor(inputs_jit)

                interested_layers = ['conv0_0', 'conv1_2', 'conv2_3', 'conv3_5']
                interested_layer_weights = [1.0, 1.0, 1.0, 1.0]
                interested_weights = 1e2

                criterion_adain = AdaIN_LossCriterion(interested_layers, interested_weights).cuda()
                contentLoss = criterion_adain(feature_target, feature_reference, interested_layer_weights)

                # R_cross classification loss
                loss = criterion(outputs, targets).cuda()

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.l2_scale * loss_l2 + 5.0 * contentLoss
                    

                loss = self.main_loss_multiplier * loss + loss_aux

                    
                if iteration < 10000:
                    parsing_label_feat = self.vae(reference_images)
                    parsing_label_feat = torch.squeeze(parsing_label_feat,1)
                    loss_parsing_vae = self.vae_criterion(parsing_label_feat,reference_labels)

                    parsing_label_input = self.vae(inputs_jit.detach())
                    parsing_label_input = torch.squeeze(parsing_label_input,1)

                    
                    optimizerVae.zero_grad()
                    loss_parsing_vae.backward()
                    optimizerVae.step()
                    
                
                parsing_label_input = self.vae(inputs_jit)
                parsing_label_input = torch.squeeze(parsing_label_input,1)
                loss_parsing = self.vae_criterion(parsing_label_input,input_labels)
                
                #Patch_AE loss
                loss = loss + loss_parsing * 30.0
                
                if iteration > starting_train_GAN and iteration % 5 == 0:
                    # train discriminator
                    net_D.zero_grad()

                    output_dr = net_D(reference_images)
                    # D_real_map = output.detach()
                    errD_real = -output_dr.mean()  # -a
                    errD_real.backward(retain_graph=True)

                    output_df = net_D(inputs_jit.detach())
                    errD_fake = output_df.mean()
                    errD_fake.backward(retain_graph=True)

                    gradient_penalty = calc_gradient_penalty(net_D, reference_images, inputs_jit, 0.1,
                                                                       "cuda:0")
                    gradient_penalty.backward()

                    optimizerD.step()
                
                if iteration > starting_train_GAN:
                    output_g = net_D(inputs_jit)
                    # D_fake_map = output.detach()
                    errG = -output_g.mean()
                    if (self.mode == 'location'):
                        errG_coeff = 0.1
                    else:
                        errG_coeff = 0.05
                    loss = loss + errG_coeff * errG

                
                loss.backward()
                optimizer.step()


                if local_rank == 0:
                    if iteration % save_every == 0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("main criterion", criterion(outputs, targets).item())
                        print("perceptual", contentLoss.item())
                        print("l2 loss", loss_l2.item())
                        print("loss_parsing", loss_parsing.item())
                        print("loss_parsing_vae", loss_parsing_vae.item())
                        if iteration > starting_train_GAN:
                            print("D errD_real", errD_real.item())
                            print("D errD_fake", errD_fake.item())
                            print("G loss", errG.item())


                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                if iteration % save_every == 0 and (save_every > 0):
                    if local_rank == 0:
                        if iteration == iterations_per_layer or iteration % 500 == 0:
                            labe_ref = parsing_label_feat
                            labe_input = parsing_label_input
                            image = labe_ref.to("cpu").clone().detach()
                            image = image.numpy().squeeze()
                            image = image.clip(0, 1)
                            final_img_cv2 = np.uint8(255 * image)
                            cv2.imwrite('./generations_' + self.pre_w +'/'+self.save_prefix+'_labelref.jpg', final_img_cv2[0,:,:])
                            image = labe_input.to("cpu").clone().detach()
                            image = image.numpy().squeeze()
                            image = image.clip(0, 1)
                            final_img_cv2 = np.uint8(255 * image)
                            cv2.imwrite('./generations_' + self.pre_w +'/'+self.save_prefix+'_labelinput.jpg', final_img_cv2[0,:,:])
                            for i_img in range(self.bs):
                                final_img = inputs[i_img, :, :, :]
                                final_img = im_convert(final_img)
                                final_img_cv2 = np.uint8(255 * final_img)
                                final_img_cv2_bgr = final_img_cv2[:, :, [2, 1, 0]]

                                if not os.path.exists(
                                        './generations_' + self.pre_w):
                                    os.makedirs(
                                        './generations_' + self.pre_w)
                                i_name = i_sample.split('/')
                                save_image_name = './generations_' + self.pre_w + '/' + self.save_prefix + str(
                                    i_img) + 'th_' + i_name[-1]
                                cv2.imwrite(save_image_name, final_img_cv2_bgr)
                                if(iteration%1000 == 0 and iteration>9000 and iteration<=16000):
                                    cv2.imwrite('./generations_' + self.pre_w + '/' + self.save_prefix + str(
                                    i_img) + 'th_iteration_'+str(iteration)+ '_'+i_name[-1], final_img_cv2_bgr)


        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
