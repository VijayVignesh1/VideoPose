import torch
import torch.nn.functional as F
from torch import nn

device = "cuda"
num_input_joints=49
input_dof=2
num_dof=4
num_output_joints=14
num_output_joints_fingers=38
num_frames=32
resnet_dim = 64
latent_dimensions = 64

# Define the residual block
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(resnet_dim,resnet_dim)
        self.bn1 = nn.BatchNorm1d(num_features=resnet_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        residual = x
        out = self.linear(x)
        out = F.relu(self.bn1(out))
        #out = self.dropout(out)
        out += residual
        return out

class BodyNetwork(nn.Module):
    def __init__(self):
        super(BodyNetwork, self).__init__()
        self.fc=torch.nn.Linear(54*2,4096)
        self.resblock=ResBlock()
        self.fc1=torch.nn.Linear(4096,14*4)
    def forward(self,x):
        x=x.view(-1,54*2)
        out=self.fc(x)
        out=self.resblock(out)
        # out=self.resblock(out)
        out=self.fc1(out)
        return torch.tanh(out)

class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        self.fc=torch.nn.Linear(49*2,4096)
        self.resblock=ResBlock()
        self.fc1=torch.nn.Linear(4096,38*4)
        self.dropout=torch.nn.Dropout(0.5)
    def forward(self,x):
        x=x.view(-1,49*2)
        out=self.fc(x)
        out=self.resblock(out)
        out=self.fc1(out)
        return torch.tanh(out)

class HandEncoder(nn.Module):
    def __init__(self):
        super(HandEncoder, self).__init__()
        self.P=latent_dimensions
        self.T=num_frames
        self.fc=torch.nn.Linear(self.T*num_input_joints*input_dof,resnet_dim)
        self.resblock=ResBlock()
        self.fc1=torch.nn.Linear(resnet_dim,self.T*self.P)
    def forward(self,x):
        out=self.fc(x)
        out=self.resblock(out)
        out=self.fc1(out)
        return out

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()

        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)

    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out



class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth

        self.AvgPool1D1 = nn.AdaptiveAvgPool1d(13)
        self.AvgPool1D2 = nn.AdaptiveAvgPool1d(3)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(size=13, mode='nearest')
        self.upsample1 = nn.Upsample(size=3, mode='nearest')
        self.upsample2 = nn.Upsample(size=latent_dimensions, mode='nearest')

        self.outcov = nn.Conv1d(self.layer_n, num_frames, kernel_size=self.kernel_size, stride=1,padding = 3)


    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)

    def forward(self, x):
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)

        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        x = torch.cat([out_2,pool_x2],1)

        x = self.layer4(x)
    
        #############Decoder####################

        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)

        up = self.upsample2(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)

        out = self.outcov(up)

        return out

class HandDecoder(torch.nn.Module):
    def __init__(self):
        super(HandDecoder, self).__init__()
        self.D=latent_dimensions
        self.T=num_frames
        self.decoder=torch.nn.Linear(self.T*self.D,self.T*num_output_joints*num_dof)
    def forward(self,parameters):
        out=self.decoder(parameters)
        return torch.tanh(out)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(125, 1)
        self.activation = nn.Sigmoid()
        self.conv= nn.Conv1d(num_frames-1, 1, kernel_size=3,bias=True)
        self.conv1= nn.Conv1d(1, 1, kernel_size=3,bias=True)
        self.max_pool = torch.nn.MaxPool1d(3)
    def frame_diff(self,decoder_output):
        result=torch.zeros(decoder_output.shape[0],decoder_output.shape[1]-1,decoder_output.shape[2])
        for i in range(2,decoder_output.shape[1]):
            result[:,i-1,:]=decoder_output[:,i,:]-decoder_output[:,i-1,:]
        return result.to(device)
    def forward(self, x):
        x = self.frame_diff(x)
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.max_pool(x)
        return self.activation(x.squeeze(1))

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.hand_encoder = HandEncoder()
        self.Unet = UNET_1D(num_frames,64,7,3)
        self.hand_decoder = HandDecoder()
    def forward(self,parameters):
        x = parameters.view(parameters.shape[0],-1)
        x = self.hand_encoder(x)
        x = x.view(x.shape[0],num_frames,latent_dimensions)
        x = self.Unet(x)
        x = x.view(x.shape[0],-1)
        x = self.hand_decoder(x)
        return x.view(x.shape[0],-1,num_output_joints*num_dof)
