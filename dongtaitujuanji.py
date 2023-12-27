import torch
import torch.nn as nn
from prometheus_client.decorator import init
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Neighbora(nn.Module):
    def __init__(self,input_dim,output_dim,w1,aggr_method="mean"):
        super(Neighbora, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim

        self.aggr_method=aggr_method
        self.weigt1=w1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weigt1)

    def forward(self,neighbor_frature):
        aggr_neighbor=neighbor_frature.mean(dim=0)
        neighbor_hidden=torch.matmul(aggr_neighbor,self.weigt1.to(device))
        # neighbor_hidden+=self.b

        return  neighbor_hidden
class SageGCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,w1,w2):
        super(SageGCN,self).__init__()
        self.activation=nn.ReLU(inplace=True)
        self.aggregaor=Neighbora(input_dim,hidden_dim,w1,aggr_method="mean")
        self.weight2=w2
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight2)
    def forward(self,src_node_features,neighbor_node_features):
        neighbor_hidden=self.aggregaor(neighbor_node_features)
        self_hidden=torch.matmul(src_node_features,self.weight2.to(device))
        hidden=self_hidden+neighbor_hidden
        return hidden
class dogtaitujuanji(nn.Module):
    def  __init__(self):
        super(dogtaitujuanji,self).__init__()
        self.input_dim=256,
        self.hidden_dim=768,
        self.num_neighbors_list=[4,4]
        self.bn3 = nn.BatchNorm1d(768)
        # self.b=nn.Parameter(torch.rand(2048))
        self.num_layers=3
        self.weigt1=nn.Parameter(torch.Tensor(256,768))
        self.weight2 = nn.Parameter(torch.Tensor(256, 768))
        self.gcn=[]
        self.sigmoid=nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        self.gcn.append(SageGCN(256,768,self.weigt1,self.weight2))
        self.gcn.append(SageGCN(256,768,self.weigt1,self.weight2))
    def forward(self,node_features_list,num):
        for i in range(num):
            hidden=node_features_list[i]

            for n in range(2):
                next_hidden=[]
                gcn=self.gcn[n]
                for hop in range(self.num_layers):
                    hidden1=hidden.copy()
                    src_node_features=torch.tensor(hidden[hop]).to(device)
                    del hidden1[hop]

                    neighbor_node_features=torch.tensor(hidden1).to(device)
                    h=gcn(src_node_features,neighbor_node_features).tolist()
                    next_hidden.append(h)
                hidden=next_hidden
            node_features_list[i]=hidden
        hidden2 = []
        for i in range(num):
            hi=node_features_list[i]
            hidden2.append((hi[0]))
        hidden3=torch.tensor(hidden2).to(device)
        hidden3=self.bn3(hidden3)
        hidden3=self.sigmoid(hidden3)

        return hidden3
