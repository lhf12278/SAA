import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
torch.set_printoptions(threshold=np.inf)

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
        t1=F.log_softmax(image_proj_text, dim=1)
        t2=torch.log(labels_mask_norm + self.epsilon)
        t3=(F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t4=torch.sum(i2t_loss, dim=1)
        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss




    def forward(self,  img_f5,img_f3,img_f41,img_f42,img_f43, \
                txt_f5,txt_f3,txt_f41,txt_f42,txt_f43,labels):
        loss = 0.0
        if self.CMPM:

         cmpc_loss = self.compute_cmpm_loss(img_f5, txt_f5, labels) \
                        + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
                        + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
                        + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
                        + self.compute_cmpm_loss(img_f3, txt_f3, labels)

         loss=cmpc_loss


        return loss
