import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from LambdaResnet import LambdaBlock
import torchaudio.transforms as transforms

class ResNet1(nn.Module):
    def __init__(self, args, num_blocks = [2,2,2,2], k=1):
        super(ResNet1, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})
        self.in_planes = args.nmfcc

        self.layer1 = self._make_layer(LambdaBlock, int(128*k), num_blocks[0], stride=1)
        self.in_planes = args.nmfcc
        self.layer2 = self._make_layer(LambdaBlock, int(128*k), num_blocks[1], stride=1)
        self.in_planes = args.nmfcc
        self.layer3 = self._make_layer(LambdaBlock, int(128*k), num_blocks[2], stride=1)
        self.in_planes = args.nmfcc
        self.layer4 = self._make_layer(LambdaBlock, int(128*k), num_blocks[3], stride=1)
        
        self.merge_layer = nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size=(4,1,time_instances), stride=(1,1,1), padding=(0,0,0))  # Output: (64, 5)
        
        # Second Conv1d layer: further processing
        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)  # Output: (128, 4)
        
        # Fully connected layer to get the output (1, 128)
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=128*k, out_features=2, bias=True),
            nn.Softmax(dim=1),
            #nn.ReLU(),
            nn.Identity()
        )


    def forward(self, x, val= False):
        #Getting the melspectrogram
        #torch.autograd.set_detect_anomaly(True)
        #batch_size = x.shape[0]
        y = self.MFCC(x)
 
        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y.permute(0,2,1)
        
        #forward pass
        y1 = self.layer1(y)
        y2 = self.layer2(y)
        y3 = self.layer3(y)
        y4 = self.layer4(y)

        y = torch.stack([y1,y2,y3,y4])
        y = y.permute(1,0,2,3)
        y = y.unsqueeze(1)
        
        y = self.merge_layer(y)
        y= torch.flatten(y,start_dim=1)
        
        #Classification head
        
        return self.classification_head(y)



    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
class ResNet2(nn.Module):
    def __init__(self, args, num_blocks = [2,2,3,3], k=1):
        super(ResNet2, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        
        # self.power_spec = transforms.Spectrogram(power=2,n_fft = 2048 ,
        #     win_length = 2048,
        #     hop_length = 512, pad=window_size // 2)
        # self.MFCC = transforms.MelSpectrogram(sample_rate = args.sampling_rate,
        #     n_fft = 2048 ,
        #     win_length = 2048,
        #     hop_length = 512,
        #     n_mels = args.nmels,
        #     pad = 128)#MelSpectrogramLayer(sample_rate=args.sampling_rate, window_size=window_size,hop_length= hop, n_mels=args.nmels)
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 512,
            "hop_length" : 256,
            "n_mels" : args.nmels,
            #"pad" : 128,
            'power': 2.0})


        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)
        #self.bn2 = nn.BatchNorm1d(64*k)
        #self.bn2 = nn.BatchNorm1d(1)
        #self.bn3 = nn.BatchNorm1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.flatten = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,time_instances), stride=(1,1))

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        #y = self.power_spec(x)
        #y = self.MFCC(x)
        y = self.MFCC(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)

        #y = self.bn2(y)
        
        y = self.avgpool(y)
        #y = y.unsqueeze(1)
 
        #y = self.flatten(y)
        
        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)
    

class ResNet2b(nn.Module):
    def __init__(self, args, num_blocks = [3,3,2,2], k=1):
        super(ResNet2b, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        
        # self.power_spec = transforms.Spectrogram(power=2,n_fft = 2048 ,
        #     win_length = 2048,
        #     hop_length = 512, pad=window_size // 2)
        # self.MFCC = transforms.MelSpectrogram(sample_rate = args.sampling_rate,
        #     n_fft = 2048 ,
        #     win_length = 2048,
        #     hop_length = 512,
        #     n_mels = args.nmels,
        #     pad = 128)#MelSpectrogramLayer(sample_rate=args.sampling_rate, window_size=window_size,hop_length= hop, n_mels=args.nmels)
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 1024,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})


        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)
        #self.bn2 = nn.BatchNorm1d(64*k)
        #self.bn2 = nn.BatchNorm1d(1)
        #self.bn3 = nn.BatchNorm1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.flatten = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,time_instances), stride=(1,1))

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        #y = self.power_spec(x)
        #y = self.MFCC(x)
        y = self.MFCC(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)

        #y = self.bn2(y)
        
        y = self.avgpool(y)
        #y = y.unsqueeze(1)
 
        #y = self.flatten(y)
        
        y = torch.flatten(y,start_dim=1)

        return self.classification_head(y)

class ResNet2MFCC(nn.Module):
    def __init__(self, args, num_blocks = [3,3,2,2], k=1):#num_blocks = [2,2,3,3], k=1): #Esto es para los RN2MFCC13t7 y superiores
        super(ResNet2MFCC, self).__init__()
        
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 512,
            "hop_length" : 256,
            "n_mels" : args.nmels,
            #"pad" : 128,
            'power': 2.0})


        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.MFCC(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)
        
        y = self.avgpool(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)

class ResNet2mel(nn.Module):
    def __init__(self, args, num_blocks = [3,3,2,2], k=1):
        super(ResNet2mel, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        
        self.mel = transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=2048, win_length=512, hop_length=256,
                                             n_mels=args.nmels, power=2.0)
        # self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
        #     "win_length" : 512,
        #     "hop_length" : 256,
        #     "n_mels" : args.nmels,
        #     "pad" : 128,
        #     'power': 2.0})


        self.in_planes = args.nmels
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.mel(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)
        
        y = self.avgpool(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)


class ResNet3mel(nn.Module):
    def __init__(self, args, num_blocks = [3,3,2,2], k=1):
        super(ResNet3mel, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        
        self.mel = transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=2048, win_length=512, hop_length=256,
                                             n_mels=args.nmels, power=2)
        # self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
        #     "win_length" : 512,
        #     "hop_length" : 256,
        #     "n_mels" : args.nmels,
        #     "pad" : 128,
        #     'power': 2.0})


        self.in_planes = args.nmels
        self.conv1 = nn.Conv1d(self.in_planes, args.nmfcc, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(args.nmfcc)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.in_planes = args.nmfcc
        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.mel(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.maxpool(self.relu(self.bn1(self.conv1(y))))

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)
        
        y = self.avgpool(y)

        y = torch.flatten(y,start_dim=1)

        return self.classification_head(y)
    

class ResNet3MFCC(nn.Module):
    def __init__(self, args, num_blocks = [3,3,2,2], k=1):
        super(ResNet3MFCC, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        
        #self.mel = transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=2048, win_length=512, hop_length=256,
        #                                     n_mels=args.nmels, power=2)
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
             "win_length" : 512,
             "hop_length" : 256,
             "n_mels" : args.nmels,
             #"pad" : 128,
             'power': 2.0})


        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, args.nmfcc, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(args.nmfcc)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(LambdaBlock, int(24 * k),num_blocks=num_blocks[0])
        self.layer2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(LambdaBlock, int(60 * k),num_blocks=num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classification_head = nn.Sequential(nn.Linear(in_features=60*k, out_features=60*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=60*k, out_features=2),
                                                 nn.Softmax(dim=1))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.MFCC(x)

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

        y = self.maxpool(self.relu(self.bn1(self.conv1(y))))

        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)
        
        y = self.avgpool(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)
    # def __init__(self, args, num_blocks = [4,4,4,4], k=1):
    #     super(ResNet3, self).__init__()
    #     window_size = int(args.time_window * args.sampling_rate)
    #     hop = int(args.hop * args.sampling_rate)
    #     time_instances = int(np.ceil(window_size/hop))
    #     self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
    #         "win_length" : 512,
    #         "hop_length" : 128,
    #         "n_mels" : args.nmels,
    #         #"pad" : 128,
    #         'power': 2.0})
    #     self.in_planes = args.nmfcc

    #     self.layer1_1 = self._make_layer(LambdaBlock, int(64*k), num_blocks[0], stride=1)
    #     self.layer1 = self._make_layer(LambdaBlock, int(128*k), num_blocks=num_blocks[1])
    #     self.in_planes = args.nmfcc
    #     self.layer2_1 = self._make_layer(LambdaBlock, int(64*k), num_blocks[0], stride=1)
    #     self.layer2 = self._make_layer(LambdaBlock, int(128*k), num_blocks[1], stride=1)
    #     self.in_planes = args.nmfcc
    #     self.layer3_1 = self._make_layer(LambdaBlock, int(64*k), num_blocks[0], stride=1)
    #     self.layer3 = self._make_layer(LambdaBlock, int(128*k), num_blocks[2], stride=1)
    #     self.in_planes = args.nmfcc
    #     self.layer4_1 = self._make_layer(LambdaBlock, int(64*k), num_blocks[0], stride=1)
    #     self.layer4 = self._make_layer(LambdaBlock, int(128*k), num_blocks[3], stride=1)
        
        
    #     self.bn1 = nn.BatchNorm1d(128)
    #     self.bn2 = nn.BatchNorm1d(128)
    #     self.bn3 = nn.BatchNorm1d(128)
    #     self.bn4 = nn.BatchNorm1d(128)

    #     self.merge_layer = nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size=(4,1,time_instances), stride=(1,1,1), padding=(0,0,0))  
        
    #     # Second Conv1d layer: further processing
    #     #self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)  # Output: (128, 4)
        
    #     # Fully connected layer to get the output (1, 128)
    #     self.classification_head = nn.Sequential(
    #         nn.Linear(in_features=22656*k, out_features=2),
    #         # nn.ReLU(),
    #         # nn.Linear(in_features=512, out_features=2),
    #         nn.Softmax(dim=1),
    #         nn.Identity()
    #     )


    # def forward(self, x, val= False):
    #     #Getting the melspectrogram
    #     #torch.autograd.set_detect_anomaly(True)
    #     #batch_size = x.shape[0]
    #     y = self.MFCC(x)
 
    #     if y.shape[1] == 1 and len(y.shape) > 3:
    #         y = torch.squeeze(y, dim=1)

    #     y.permute(0,2,1)
        
    #     #forward pass
    #     y11 = self.layer1_1(y)
    #     y21 = self.layer2_1(y)
    #     y31 = self.layer3_1(y)
    #     y41 = self.layer4_1(y)

    #     y1 = self.bn1(self.layer1(y11))
    #     y2 = self.bn2(self.layer2(y21))
    #     y3 = self.bn3(self.layer3(y31))
    #     y4 = self.bn4(self.layer4(y41))


    #     y = torch.stack([y1,y2,y3,y4])
    #     y = y.permute(1,0,2,3)
    #     y = y.unsqueeze(1)
        
    #     y = self.merge_layer(y)
    #     y = torch.flatten(y,start_dim=1)
        
    #     #Classification head
    #     y = self.classification_head(y)

    #     return y



    # def _make_layer(self, block, planes, num_blocks, stride=1):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for idx, stride in enumerate(strides):
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
        
    #     return nn.Sequential(*layers)
    

class ResNet4(nn.Module):
    def __init__(self, args, num_blocks = [2,2,2,2], k=1):
        super(ResNet4, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})

        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1_1 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[0])
        self.layer2_1 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[1])
        self.layer3_1 = self._make_layer(LambdaBlock, int(64 * k),num_blocks=num_blocks[2])
        
        self.layer1_2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[0])
        self.layer2_2 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[1])
        self.layer3_2 = self._make_layer(LambdaBlock, int(64 * k),num_blocks=num_blocks[2])

        self.bn2 = nn.BatchNorm1d(64*k)
        self.bn3 = nn.BatchNorm1d(64*k)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.bn3 = nn.BatchNorm1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d((64*k,1))
        #self.flatten = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,time_instances), stride=(1,1))

        self.classification_head = nn.Sequential(nn.Linear(in_features=128*k, out_features=64*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=64*k, out_features=2),
                                                 nn.Softmax(dim=1), 
                                                 nn.Identity())

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.MFCC(x)

        

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
        
        

        y1 = self.layer1_1(y)
        y1 = self.layer2_1(y1)
        y1 = self.layer3_1(y1)

        y2 = self.layer1_1(y)
        y2 = self.layer2_1(y2)
        y2 = self.layer3_1(y2)


        y1 = self.bn2(y1)
        y2 = self.bn3(y2)
        
        y = torch.stack([y1,y2])
        if torch.isnan(y).any():
                    print("NAN en test tmb")
        y = y.permute(1,0,2,3)
        y = self.avgpool(y)
        #y = y.unsqueeze(1)
 
        #y = self.flatten(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)


class ResNet5(nn.Module):
    def __init__(self, args, num_blocks = [2,2,2,2], k=1):
        super(ResNet5, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})

        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1_1 = self._make_layer(LambdaBlock, int(16 * k),num_blocks=num_blocks[0])
        self.layer2_1 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1])
        self.layer3_1 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2])
        
        self.layer1_2 = self._make_layer(LambdaBlock, int(16 * k),num_blocks=num_blocks[0])
        self.layer2_2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1])
        self.layer3_2 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2])

        self.layer1_3 = self._make_layer(LambdaBlock, int(16 * k),num_blocks=num_blocks[0])
        self.layer2_3 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[1])
        self.layer3_3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[2])

        self.bn2 = nn.BatchNorm1d(48*k)
        self.bn3 = nn.BatchNorm1d(48*k)
        self.bn4 = nn.BatchNorm1d(48*k)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.bn3 = nn.BatchNorm1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d((48*k,1))
        #self.flatten = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,time_instances), stride=(1,1))

        self.classification_head = nn.Sequential(nn.Linear(in_features=144*k, out_features=64*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=64*k, out_features=2),
                                                 nn.Softmax(dim=1), 
                                                 nn.Identity())

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.MFCC(x)

        

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
        
        

        y1 = self.layer1_1(y)
        y1 = self.layer2_1(y1)
        y1 = self.layer3_1(y1)

        y2 = self.layer1_1(y)
        y2 = self.layer2_1(y2)
        y2 = self.layer3_1(y2)

        y3 = self.layer1_1(y)
        y3 = self.layer2_1(y3)
        y3 = self.layer3_1(y3)

        y1 = self.bn2(y1)
        y2 = self.bn3(y2)
        y3 = self.bn3(y3)
        
        y = torch.stack([y1,y2, y3])
        if torch.isnan(y).any():
                    print("NAN en test tmb")
        y = y.permute(1,0,2,3)
        y = self.avgpool(y)
        #y = y.unsqueeze(1)
 
        #y = self.flatten(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)
    


class ResNet5_big(nn.Module):
    def __init__(self, args, num_blocks = [2,2,2,2], k=1):
        super(ResNet5_big, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})

        self.in_planes = args.nmfcc
        self.conv1 = nn.Conv1d(self.in_planes, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1_1 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[0])
        self.layer2_1 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[1])
        self.layer3_1 = self._make_layer(LambdaBlock, int(64 * k),num_blocks=num_blocks[2])
        
        self.layer1_2 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[0])
        self.layer2_2 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[1])
        self.layer3_2 = self._make_layer(LambdaBlock, int(64 * k),num_blocks=num_blocks[2])

        self.layer1_3 = self._make_layer(LambdaBlock, int(32 * k),num_blocks=num_blocks[0])
        self.layer2_3 = self._make_layer(LambdaBlock, int(48 * k),num_blocks=num_blocks[1])
        self.layer3_3 = self._make_layer(LambdaBlock, int(64 * k),num_blocks=num_blocks[2])

        self.bn2 = nn.BatchNorm1d(64*k)
        self.bn3 = nn.BatchNorm1d(64*k)
        self.bn4 = nn.BatchNorm1d(64*k)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.bn3 = nn.BatchNorm1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d((64*k,1))
        #self.flatten = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,time_instances), stride=(1,1))

        self.classification_head = nn.Sequential(nn.Linear(in_features=192*k, out_features=64*k),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=64*k, out_features=2),
                                                 nn.Softmax(dim=1), 
                                                 nn.Identity())

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x ):
        #torch.autograd.set_detect_anomaly(True)
        y = self.MFCC(x)

        

        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
        
        

        y1 = self.layer1_1(y)
        y1 = self.layer2_1(y1)
        y1 = self.layer3_1(y1)

        y2 = self.layer1_1(y)
        y2 = self.layer2_1(y2)
        y2 = self.layer3_1(y2)

        y3 = self.layer1_1(y)
        y3 = self.layer2_1(y3)
        y3 = self.layer3_1(y3)

        y1 = self.bn2(y1)
        y2 = self.bn3(y2)
        y3 = self.bn3(y3)
        
        y = torch.stack([y1,y2, y3])
        if torch.isnan(y).any():
                    print("NAN en test tmb")
        y = y.permute(1,0,2,3)
        y = self.avgpool(y)
        #y = y.unsqueeze(1)
 
        #y = self.flatten(y)

        y = torch.flatten(y,start_dim=1)
        
        return self.classification_head(y)
    
class LSTMnet(nn.Module):
    def __init__(self, args):
        super(LSTMnet, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})
        self.lstm_layer = nn.LSTM(input_size=args.nmfcc,num_layers=3, hidden_size=256, batch_first=True)
        self.classification_head = nn.Sequential(nn.Linear(in_features= 256, out_features=128), nn.ReLU(),
                                                 nn.Linear(in_features=128, out_features=2), nn.Softmax(dim = 1))
    
    def forward(self, x ):
         y = self.MFCC(x)
         if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
         y = y.permute(0,2,1)
         y,_ = self.lstm_layer(y)
         return self.classification_head(y[:,-1,:])
    
class BiLSTMnetmel(nn.Module):
    def __init__(self, args):
        super(BiLSTMnetmel, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.mel = transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=2048, win_length=512, hop_length=256,
                                             n_mels=args.nmels, power=2)
        self.lstm_layer = nn.LSTM(input_size=args.nmels, num_layers=4,hidden_size=2048, bidirectional=True, batch_first=True)
        self.classification_head = nn.Sequential(nn.Linear(in_features= 4096, out_features=1024), nn.ReLU(),
                                                 nn.Linear(in_features=1024, out_features=2), nn.Softmax(dim = 1))
    
    def forward(self, x  ):
         y = self.mel(x)
         if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

         y = y.permute(0,2,1)
         y,_ = self.lstm_layer(y)

         return self.classification_head(y[:,-1,:])

class BiLSTMnetMFCC(nn.Module):
    def __init__(self, args):
        super(BiLSTMnetMFCC, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 512,
            "hop_length" : 256,
            "n_mels" : args.nmels,
            #"pad" : 128,
            'power': 2.0})
        self.lstm_layer = nn.LSTM(input_size=args.nmfcc, num_layers=4,hidden_size=256, bidirectional=True, batch_first=True)
        self.classification_head = nn.Sequential(nn.Linear(in_features= 512, out_features=256), nn.ReLU(),
                                                 nn.Linear(in_features=256, out_features=2), nn.Softmax(dim = 1))
    
    def forward(self, x  ):
         y = self.MFCC(x)
         if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

         y = y.permute(0,2,1)
         y,_ = self.lstm_layer(y)
         return self.classification_head(y[:,-1,:])

class ConvLSTMnet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMnet, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})
        self.lstm_layer = nn.LSTM(input_size=args.nmfcc,num_layers=3, hidden_size=256, batch_first=True)
        self.classification_head = nn.Sequential(nn.Conv2d(in_channels = 1,out_channels=1,kernel_size=(12,1),stride=(1,1)), #nn.ReLU(),
                                                 nn.Linear(in_features=256, out_features=2), nn.Softmax(dim = 1))
    
    def forward(self, x ):
         y = self.MFCC(x)
         if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
         y = y.permute(0,2,1)
         y,_ = self.lstm_layer(y)
         y = torch.unsqueeze(y, dim = 1)
         y = self.classification_head(y)
         return torch.flatten(y, start_dim=1)
    
class ConvBiLSTMnet(nn.Module):
    def __init__(self, args):
        super(ConvBiLSTMnet, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            "pad" : 128,
            'power': 2.0})
        self.lstm_layer = nn.LSTM(input_size=args.nmfcc, num_layers=4,hidden_size=256, bidirectional=True, batch_first=True)
        self.classification_head = nn.Sequential(nn.Conv2d(in_channels = 1,out_channels=1,kernel_size=(12,1),stride=(1,1)), #nn.ReLU(),
                                                 nn.Linear(in_features=512, out_features=2), nn.Softmax(dim = 1))
    
    def forward(self, x  ):
         y = self.MFCC(x)
         if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)

         y = y.permute(0,2,1)
         y,_ = self.lstm_layer(y)

         y = torch.unsqueeze(y, dim = 1)
         #y = y.permute(1,0,2,3)
         
         y = self.classification_head(y)

         return torch.flatten(y, start_dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )
        
        out = self.fc_out(out)
        print(out.shape)
        return out


class ConvAttention(nn.Module):
    def __init__(self, embed_size, num_heads, conv_kernel_size, dropout=0.1):
        super(ConvAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads
        
        # Convolutional layers for queries, keys, and values
        self.query_conv = nn.Conv1d(embed_size, embed_size, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=num_heads)
        self.key_conv = nn.Conv1d(embed_size, embed_size, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=num_heads)
        self.value_conv = nn.Conv1d(embed_size, embed_size, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=num_heads)
        
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        print(x.shape)
        
        # Permute to match Conv1d input shape: (batch_size, embed_size, seq_len)
        #x = x.permute(0, 2, 1)
        
        # Apply convolutional layers to compute queries, keys, values
        queries = self.query_conv(x)  # (batch_size, embed_size, seq_len)
        keys = self.key_conv(x)
        values = self.value_conv(x)
        
        # Permute back to original shape for further processing
        queries = queries.permute(0, 2, 1)  # (batch_size, seq_len, embed_size)
        keys = keys.permute(0, 2, 1)
        values = values.permute(0, 2, 1)
        
        # Compute attention scores: Replace dot-product attention with conv-based interaction
        # This is a simplified example, in practice, you might still use dot products with convoluted queries/keys
        attention = torch.softmax(torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_size ** 0.5), dim=-1)
        
        # Weighted sum of values (context vector)
        out = torch.bmm(attention, values)  # (batch_size, seq_len, embed_size)
        
        # Pass through final linear layer
        out = self.fc_out(out)
        print(out.shape)
        return out
    
class ConvTransformer(nn.Module):
    def __init__(self, args):
        super(ConvTransformer, self).__init__()
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)
        #time_instances = int(np.ceil(window_size/hop))
        self.MFCC = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=args.nmfcc, melkwargs={"n_fft" : 2048 ,
            "win_length" : 2048,
            "hop_length" : 512,
            "n_mels" : args.nmels,
            #"pad" : 128,
            'power': 2.0})
        
        self.in_planes = args.nmfcc

        self.conv1 = nn.Conv1d(self.in_planes, args.nmfcc, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(args.nmfcc)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.trans_layer = nn.TransformerEncoderLayer(d_model=self.in_planes, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.trans_layer,num_layers=16)
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Sequential(nn.Linear(in_features=16, out_features=16),nn.ReLU(),
                                            nn.Linear(in_features=16,out_features= 2), nn.Softmax(dim = 1))
        


    def forward(self, x ):
        y = self.MFCC(x)
        if y.shape[1] == 1 and len(y.shape) > 3:
            y = torch.squeeze(y, dim=1)
        #print(y.shape)
        #y.permute(0,2,1)
        # Permute to (batch_size, n_mels, seq_len) for conv layers
        #y = y.permute(0, 2, 1)

        # Apply convolutional layers
        y = self.maxpool(self.relu(self.bn1(self.conv1(y))))
        
        # Permute to (batch_size, seq_len, conv_out_channels) for transformer
        y = y.permute(0, 2, 1)
        
        # Apply Transformer encoder
        y = self.transformer_encoder(y)
        
        # Use the output from the last time step for classification
        y = y[:, -1, :]  # (batch_size, conv_out_channels)
        if torch.isnan(y).any() or torch.isinf(y).any():
             print("AAAAAAAAAAAAAaaa")
        # Pass through final linear layer
        
        return self.classification(self.dropout(y))
        
             
     
