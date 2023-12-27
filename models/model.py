from numpy import tile
from torchvision import models

from dongtaitujuanji import dogtaitujuanji
from models.SCFC import ResNet_text_50
import transformers as ppb
from torch.nn import init
import torch.nn as nn
import torch
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cosine_dist(x, y):
	'''
	compute cosine distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	'''

	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	return torch.matmul(x, y.transpose(0,1))
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        return x1, x2, x3, x4


class Network(nn.Module):
    def __init__(self,args):
        super(Network, self).__init__()
        self.model_img = ResNet_image_50()
        self.model_txt = ResNet_text_50(args)
        self.dogtaitujuanj=dogtaitujuanji()

        self.ran1= nn.Parameter(torch.rand(1, 2048))
        self.ran2 = nn.Parameter(torch.rand(1, 2048))


        if args.embedding_type == 'BERT':
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
            self.text_embed = model_class.from_pretrained(pretrained_weights)
            self.text_embed.eval()

            self.BERT = True
            for p in self.text_embed.parameters():
                p.requires_grad = False

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self,a, img, txt, mask):
         with torch.no_grad():
          txt = self.text_embed(txt, attention_mask=mask)
          txt = txt[0]
          txt = txt.unsqueeze(1)
          txt = txt.permute(0, 3, 1, 2)

         _, _, img3, img4 = self.model_img(img)  # img4: batch x 2048 x 24 x 8
         img_f3 = self.max_pool(img3).squeeze(dim=-1).squeeze(dim=-1)
         img_f41 = self.max_pool(img4[:, :, 0:8, :]).squeeze(dim=-1).squeeze(dim=-1)

         img_f42 = self.max_pool(img4[:, :, 8:16, :]).squeeze(dim=-1).squeeze(dim=-1)
         img_f43 = self.max_pool(img4[:, :, 16:, :]).squeeze(dim=-1).squeeze(dim=-1)



         t1=self.max_pool(img4[:, :, 0:8, :])
         t2 = self.max_pool(img4[:, :, 8:16, :])
         t3 = self.max_pool(img4[:, :,16:, :])

         t9 = self.ran1.unsqueeze(dim=-1).unsqueeze(dim=-1).to(device)
         t9 = t9.repeat_interleave(a, dim=0)


         tt=torch.cat([t9,t1, t2, t3], dim=2).permute(0, 2, 1, 3).squeeze(dim=-1)
         list = tt.tolist()
         numm=len(list)

         img_f51=self.dogtaitujuanj(list,numm)
         img_f5=torch.cat([img_f51,img_f41,img_f42, img_f43], dim=1)


         txt_f3, txt41, txt42, txt43= self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64z
        #

         t81= self.ran2.unsqueeze(dim=-1).unsqueeze(dim=-1).to(device)
         t81=t81.repeat_interleave(a,dim=0)
        #

         t = torch.cat([t81,txt_f41, txt_f42, txt_f43], dim=2).permute(0, 2, 1, 3).squeeze(dim=-1)
         list = t.tolist()
         nu=len(list)
         txt_f51 = self.dogtaitujuanj(list,nu).to(device)



         txt_f41 = txt_f41.squeeze(dim=-1).squeeze(dim=-1)
         txt_f42 = txt_f42.squeeze(dim=-1).squeeze(dim=-1)
         txt_f43 = txt_f43.squeeze(dim=-1).squeeze(dim=-1)
         txt_f5=torch.cat([txt_f51,txt_f41, txt_f42, txt_f43], dim=1)


         if self.training:
            return img_f5,img_f3,img_f41,img_f42,img_f43, \
                   txt_f5,txt_f3,txt_f41,txt_f42,txt_f43
         else:
            return img_f5, txt_f5
