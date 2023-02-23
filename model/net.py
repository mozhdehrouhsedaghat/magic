"""Defines the neural network, losss function and metrics"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import pdb

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        
class ConvBlocknoac(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlocknoac,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel))
        
class ConvTransBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvTransBlock,self).__init__()
        self.add_module('conv',nn.ConvTranspose2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        
class myWDiscriminator(nn.Module):
    def __init__(self, nfc, min_nfc):
        super(myWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = 128#int(nfc)
        self.head = ConvBlock(3,N,3,0,1)
        self.body = nn.Sequential()
        for i in range(5-2):
            N = int(nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,min_nfc),max(N,min_nfc),3,0,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,min_nfc),1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x, x

class myWDiscriminator2(nn.Module):
    def __init__(self, nfc, min_nfc):
        super(myWDiscriminator2, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(nfc)
        self.head = ConvBlock(3,N*2,4,0,1)
        self.body = nn.Sequential()
        block = ConvBlock(N*2, N*4,4,0,2)
        self.body.add_module('block%1',block)
        block = ConvBlock(N*4, N*4,4,0,2)
        self.body.add_module('block%2',block) 
        self.lastlayer = ConvBlock(N*4, N*4,3,0,1)
        self.tail2 = nn.Conv2d(N*4,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        y2 = self.lastlayer(x)
        y2 = self.tail2(y2)
        return y2

    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        N = 64
        self.is_cuda = torch.cuda.is_available()
        self.head = ConvBlock(3,N,3,0,1)
        self.body = nn.Sequential()
        block = ConvBlock(N, N,3,0,1)
        self.body.add_module('block%1',block)
        block = ConvBlock(N, N,3,0,1)
        self.body.add_module('block%2',block)
        #block = ConvBlock(2*N, 2*N,3,0,1)
        #self.body.add_module('block%3',block) 
        #block = ConvTransBlock(2*N, 2*N,3,0,1)
        #self.body.add_module('block%4',block)
        block = ConvTransBlock(N, N,3,0,1)
        self.body.add_module('block%5',block)
        block = ConvTransBlock(N, N,3,0,1)
        self.body.add_module('block%6',block)
        block = ConvTransBlock(N, 1,3,0,1)
        self.body.add_module('block%7',block)
        
    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        softmax = nn.Softmax(dim = 1)
        return x
    
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        global printlayer_index
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1,output_padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # printlayer = [PrintLayer(name = str(printlayer_index))]
            # printlayer_index += 1
            # model = printlayer + down + [submodule] + up
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = printlayer + down + up
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
                # model = printlayer + down + [submodule] + printlayer + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule]  + up
                # model = printlayer + down + [submodule] + printlayer + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        model_output = self.model(x)
        wb,hb = model_output.size()[3],model_output.size()[2]
        wa,ha = x.size()[3],x.size()[2]
        l = int((wb-wa)/2)
        t = int((hb-ha)/2)
        model_output = model_output[:,:,t:t+ha,l:l+wa]
        if self.outermost:
            return model_output
        else:
            return torch.cat([x, model_output], 1)           #if not the outermost block, we concate x and self.model(x) during forward to implement unet

class UnetGenerator(nn.Module):

    def __init__(self, segment_classes=1, input_nc=3 , ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        output_nc = segment_classes
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        #maybe do some check here with softmax
        self.model = unet_block

    def forward(self, input):
        softmax = torch.nn.Softmax(dim = 1)
        return softmax(self.model(input))
    
class myWDiscriminator_location(nn.Module):
    def __init__(self, nfc, min_nfc):
        super(myWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(nfc)
        self.head = ConvBlock(3,N*2,4,0,1)
        self.body = nn.Sequential()
        block = ConvBlock(N*2, N*4,4,0,2)
        self.body.add_module('block%1',block)
        block = ConvBlock(N*4, N*4,4,0,2)
        self.body.add_module('block%2',block) 
        block = ConvBlock(N*4, N*4,3,0,1)
        self.body.add_module('block%3',block) 
        self.tail = nn.Conv2d(N*4,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    print('is conv and norm',classname)
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
	
def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)
	
class get_features(nn.Module):
    def __init__(self,network):
        super(get_features,self).__init__()

        self.network = network
        self.layers1 = {'0': 'conv1_0',
                        '1': 'conv1_1',
                        '2': 'conv1_2'
                        }
        self.layers2 = {'0': 'conv2_0',
                        '1': 'conv2_1',
                        '2': 'conv2_2',
                        '3': 'conv2_3'
                        }
        self.layers3 = {'0': 'conv3_0',
                        '1': 'conv3_1',
                        '2': 'conv3_2',
                        '3': 'conv3_3',
                        '4': 'conv3_4',
                        '5': 'conv3_5'
                        }
        self.layers4 = {'0': 'conv4_0',
                        '1': 'conv4_1',
                        '2': 'conv4_2'
                        }

    def forward(self,image):


        features = {}
        x = image
        x = self.network.module.conv1(x)

        x = self.network.module.bn1(x)
        x = self.network.module.relu(x)
        features['conv0_0'] = x
        x = self.network.module.maxpool(x)

        for name, layer in enumerate(self.network.module.layer1):
            x = layer(x)
            if str(name) in self.layers1:
                features[self.layers1[str(name)]] = x
        for name, layer in enumerate(self.network.module.layer2):
            x = layer(x)
            if str(name) in self.layers2:
                features[self.layers2[str(name)]] = x
        for name, layer in enumerate(self.network.module.layer3):
            x = layer(x)
            if str(name) in self.layers3:
                features[self.layers3[str(name)]] = x
        for name, layer in enumerate(self.network.module.layer4):
            x = layer(x)
            if str(name) in self.layers4:
                features[self.layers4[str(name)]] = x
        return features


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor
    

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    mse_loss = nn.MSELoss()
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)
		   

class LossCriterion(nn.Module):
    def __init__(self,interested_layers,interested_weights):
        super(LossCriterion,self).__init__()

        self.interested_layers = interested_layers
        self.interested_weights = interested_weights

        self.styleLosses = [styleLoss()] * len(interested_layers)
        self.contentLosses = [nn.MSELoss()] * len(interested_layers)


    def forward(self,tF,cF, interested_layer_weights):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.interested_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            if i == 0:
                loss_i = self.contentLosses[i]
                loss_c = loss_i(tf_i, cf_i)
                # loss_c = torch.exp(loss_c)
            else:
                loss_i = self.contentLosses[i]
                loss_c = loss_i(tf_i, cf_i)
                # loss_c = torch.exp(loss_c)
            totalContentLoss += interested_layer_weights[i] * loss_c
        totalContentLoss = totalContentLoss * self.interested_weights


        return totalContentLoss

class AdaIN_LossCriterion(nn.Module):
    def __init__(self,interested_layers,interested_weights):
        super(AdaIN_LossCriterion,self).__init__()

        self.interested_layers = interested_layers
        self.interested_weights = interested_weights

        self.contentLosses = [nn.MSELoss()] * len(interested_layers)


    def forward(self,tF,cF, interested_layer_weights):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.interested_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]

            loss_i = calc_style_loss(tf_i, cf_i)
            totalContentLoss += interested_layer_weights[i] * loss_i
        totalContentLoss = totalContentLoss * self.interested_weights
        return totalContentLoss
