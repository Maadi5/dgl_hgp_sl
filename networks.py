import torch
from dgl.nn import AvgPooling, MaxPooling
import torch.nn.functional as F
import torch.nn
from layers import ConvPoolReadout


class HGPSLModel(torch.nn.Module):
    r"""
    Description
    -----------
    The graph classification model using HGP-SL pooling.
    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels.
    out_feat : int
        The number of output node feature's channels.
    hid_feat : int
        The number of hidden state's channels.
    dropout : float, optional
        The dropout rate. Default: 0
    pool_ratio : float, optional
        The pooling ratio for each pooling layer. Default: 0.5
    conv_layers : int, optional
        The number of graph convolution and pooling layers. Default: 3
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency.
        Currently we only support full graph. Default: :obj:`False`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    """

    def __init__(self, in_feat: int, out_feat: int, hid_feat: int,
                 dropout: float = 0., pool_ratio: float = .5, conv_layers: int = 3,
                 sample: bool = False, sparse: bool = True, sl: bool = True,
                 lamb: float = 1.):
        super(HGPSLModel, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.num_layers = conv_layers
        self.pool_ratio = pool_ratio

        convpools = []
        for i in range(conv_layers):
            c_in = in_feat if i == 0 else hid_feat
            c_out = hid_feat
            use_pool = (i != conv_layers - 1)
            convpools.append(ConvPoolReadout(c_in, c_out, pool_ratio=pool_ratio,
                                             sample=sample, sparse=sparse, sl=sl,
                                             lamb=lamb, pool=use_pool))
        self.convpool_layers = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)

    def forward(self, graph, n_feats, e_feats):
        final_readout = None
        # e_feat = None

        for i in range(self.num_layers):
            graph, n_feats, e_feats, readout = self.convpool_layers[i](graph, n_feats, e_feats)
            print(n_feats.shape)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        n_feats = F.relu(self.lin1(final_readout))
        n_feats = F.dropout(n_feats, p=self.dropout, training=self.training)
        n_feats = F.relu(self.lin2(n_feats))
        n_feats = F.dropout(n_feats, p=self.dropout, training=self.training)
        n_feats = self.lin3(n_feats)

        return F.log_softmax(n_feats, dim=-1)