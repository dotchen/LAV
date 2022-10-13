import torch.nn.functional as F
from torch import nn


class SegNet(nn.Module):

	def __init__(self, num_classes):
		super(SegNet, self).__init__()

		#SegNet Architecture
		#Takes input of size in_chn = 3 (RGB images have 3 channels)
		#Outputs size label_chn (N # of classes)

		#ENCODING consists of 5 stages
		#Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
		#Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

		#General Max Pool 2D for ENCODING layers
		#Pooling indices are stored for Upsampling in DECODING layers

		self.in_chn = 3
		self.out_chn = num_classes

		self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

		self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
		self.BNEn11 = nn.BatchNorm2d(64)
		self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.BNEn12 = nn.BatchNorm2d(64)

		self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.BNEn21 = nn.BatchNorm2d(128)
		self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.BNEn22 = nn.BatchNorm2d(128)

		self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.BNEn31 = nn.BatchNorm2d(256)
		self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNEn32 = nn.BatchNorm2d(256)
		self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNEn33 = nn.BatchNorm2d(256)

		self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.BNEn41 = nn.BatchNorm2d(512)
		self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn42 = nn.BatchNorm2d(512)
		self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn43 = nn.BatchNorm2d(512)

		self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn51 = nn.BatchNorm2d(512)
		self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn52 = nn.BatchNorm2d(512)
		self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn53 = nn.BatchNorm2d(512)


		#DECODING consists of 5 stages
		#Each stage corresponds to their respective counterparts in ENCODING

		#General Max Pool 2D/Upsampling for DECODING layers
		self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

		self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe53 = nn.BatchNorm2d(512)
		self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe52 = nn.BatchNorm2d(512)
		self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe51 = nn.BatchNorm2d(512)

		self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe43 = nn.BatchNorm2d(512)
		self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe42 = nn.BatchNorm2d(512)
		self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
		self.BNDe41 = nn.BatchNorm2d(256)

		self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNDe33 = nn.BatchNorm2d(256)
		self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNDe32 = nn.BatchNorm2d(256)
		self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		self.BNDe31 = nn.BatchNorm2d(128)

		self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.BNDe22 = nn.BatchNorm2d(128)
		self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		self.BNDe21 = nn.BatchNorm2d(64)

		self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.BNDe12 = nn.BatchNorm2d(64)
		self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
		self.BNDe11 = nn.BatchNorm2d(self.out_chn)

	def forward(self, x):

		#ENCODE LAYERS
		#Stage 1
		x = F.relu(self.BNEn11(self.ConvEn11(x))) 
		x = F.relu(self.BNEn12(self.ConvEn12(x))) 
		x, ind1 = self.MaxEn(x)
		size1 = x.size()

		#Stage 2
		x = F.relu(self.BNEn21(self.ConvEn21(x))) 
		x = F.relu(self.BNEn22(self.ConvEn22(x))) 
		x, ind2 = self.MaxEn(x)
		size2 = x.size()

		#Stage 3
		x = F.relu(self.BNEn31(self.ConvEn31(x))) 
		x = F.relu(self.BNEn32(self.ConvEn32(x))) 
		x = F.relu(self.BNEn33(self.ConvEn33(x))) 	
		x, ind3 = self.MaxEn(x)
		size3 = x.size()

		#Stage 4
		x = F.relu(self.BNEn41(self.ConvEn41(x))) 
		x = F.relu(self.BNEn42(self.ConvEn42(x))) 
		x = F.relu(self.BNEn43(self.ConvEn43(x))) 	
		x, ind4 = self.MaxEn(x)
		size4 = x.size()

		#Stage 5
		x = F.relu(self.BNEn51(self.ConvEn51(x))) 
		x = F.relu(self.BNEn52(self.ConvEn52(x))) 
		x = F.relu(self.BNEn53(self.ConvEn53(x))) 	
		x, ind5 = self.MaxEn(x)
		size5 = x.size()

		#DECODE LAYERS
		#Stage 5
		x = self.MaxDe(x, ind5, output_size=size4)
		x = F.relu(self.BNDe53(self.ConvDe53(x)))
		x = F.relu(self.BNDe52(self.ConvDe52(x)))
		x = F.relu(self.BNDe51(self.ConvDe51(x)))

		#Stage 4
		x = self.MaxDe(x, ind4, output_size=size3)
		x = F.relu(self.BNDe43(self.ConvDe43(x)))
		x = F.relu(self.BNDe42(self.ConvDe42(x)))
		x = F.relu(self.BNDe41(self.ConvDe41(x)))

		#Stage 3
		x = self.MaxDe(x, ind3, output_size=size2)
		x = F.relu(self.BNDe33(self.ConvDe33(x)))
		x = F.relu(self.BNDe32(self.ConvDe32(x)))
		x = F.relu(self.BNDe31(self.ConvDe31(x)))

		#Stage 2
		x = self.MaxDe(x, ind2, output_size=size1)
		x = F.relu(self.BNDe22(self.ConvDe22(x)))
		x = F.relu(self.BNDe21(self.ConvDe21(x)))

		#Stage 1
		x = self.MaxDe(x, ind1)
		x = F.relu(self.BNDe12(self.ConvDe12(x)))
		x = self.ConvDe11(x)

		return x
