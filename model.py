import torch
from torch.nn import Linear, Sequential, SiLU, ModuleList
from readout import AtomWise, MolWise, PairWise, AllPairWise, Vectorial, PairReadout, BDEPairReadout
from torch_geometric.nn import LayerNorm
from sbftransformer_conv import SBFTransformerConv
import torch.nn.functional as F
from residual_layer import ResidualLayer
from initializer import he_orthogonal_init, Glorot_Ortho_
from torch_scatter import scatter_add
from dnmp import DNMP

class NoAttn(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([DNMP(in_channels = in_channels, int_emb_size= 128, rbf_dim = rbf_dim, sbf_dim=sbf_dim,  emb_size = emb_size) for i in range(conv_layers)])
        self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](x = out, edge_index = data.edge_index, sbf = data.edge_sbf, rbf = data.node_rbf)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        return results.view(-1)

class SBFTransformer_vectorial_preds(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readout = Vectorial(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1)
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
        predicted_displacement = self.readout(x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0], node_vec = data.node_vector) # atom_wise res

        return predicted_displacement

class SBFTransformer(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        return results.view(-1)

class SBFTransformerGlobal(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8, pool_option='mean', readout_depth=3):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])
        self.UpProjection = Linear(in_channels, 2*in_channels)
        self.readouts = MolWise(mlp_depth=readout_depth, in_channels = 2*in_channels, rbf_dim=rbf_dim, num_target=1, pool_option=pool_option)
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.UpProjection.weight)
        torch.nn.init.zeros_(self.UpProjection.bias)
        #Glorot_Ortho_(self.edgenn[0].weight)
        #torch.nn.init.zeros_(self.edgenn[0].bias)
        #Glorot_Ortho_(self.edgenn[2].weight)
        #torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        #results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0], atom_batch = atom_batch, dim_size = int(data.batch.max()) + 1)

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
        
        out = self.AF(self.UpProjection(out))
        results = self.readouts(x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0], atom_batch = atom_batch, dim_size = int(data.batch.max()) + 1)

        return results.view(-1)

class SBFTransformer_radical_all(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        #self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        #self.readout = AllPairWise(in_channels=in_channels)    # a slightly adjustment:240514
        self.readouts = ModuleList([AllPairWise(in_channels = in_channels) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, num_graphs):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, is_cleave = data.is_cleave, batch=data.batch)
        #results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            #results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
            results += self.readouts[i+1](x = out, is_cleave = data.is_cleave, batch=data.batch)
        #results = self.readout(x = out, is_cleave = data.is_cleave, batch=data.batch)

        return results.view(-1)

class SBFTransformer_radical(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        #self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.readout = PairWise(in_channels=in_channels)
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, num_graphs):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        #results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            #results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
        results = self.readout(x = out, is_cleave = data.is_cleave, num_graphs = num_graphs)

        return results.view(-1)
    
class SBFTransformer_bde(torch.nn.Module):
    def __init__(self, conv_layers, emb_size, sbf_dim, rbf_dim = 16, in_channels = 128, heads = 8):
        super().__init__()
        self.in_channels =in_channels
        self.rbf_dim = rbf_dim
        self.edgenn = Sequential(Linear(emb_size,emb_size), SiLU(), Linear(emb_size,emb_size))
        self.convs = ModuleList([SBFTransformerConv(in_channels = in_channels, out_channels = int(in_channels/heads),
             heads = heads, sbf_dim = sbf_dim * rbf_dim, rbf_dim = rbf_dim, dropout = 0, edge_dim = emb_size) for i in range(conv_layers)])    #
        self.readouts = ModuleList([AtomWise(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.pair_readouts = ModuleList([PairReadout(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bde_readouts = ModuleList([BDEPairReadout(in_channels = in_channels, rbf_dim=rbf_dim, num_target=1) for i in range(conv_layers + 1)])
        self.bf_skip = ModuleList([ResidualLayer(in_channels) for i in range(conv_layers)])
        self.af_skip = ModuleList([Sequential(ResidualLayer(in_channels),ResidualLayer(in_channels)) for i in range(conv_layers)])
        self.dense_bf_skip = ModuleList([Linear(in_channels, in_channels, bias=True) for i in range(conv_layers)])
        self.AF = SiLU()
        self.LayerNorm = LayerNorm(in_channels = in_channels, eps = 1e-8, affine = False)  #之前使用出现的问题来自forward时没有传进batch参数
        self.conv_layers = conv_layers
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_bf_skip:
            Glorot_Ortho_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        Glorot_Ortho_(self.edgenn[0].weight)
        torch.nn.init.zeros_(self.edgenn[0].bias)
        Glorot_Ortho_(self.edgenn[2].weight)
        torch.nn.init.zeros_(self.edgenn[2].bias)

    def forward(self, data, edge_index_0, atom_batch):
        edge_attr = self.edgenn(data.edge_attr)
        out = data.x
        results = self.readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
        results_pairs = self.pair_readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
        bde_pairs = self.bde_readouts[0](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
        for i in range(self.conv_layers):
            out_res_0 = out
            out = self.convs[i](sbf = data.edge_sbf, rbf = data.node_rbf, x= out, edge_index = data.edge_index, edge_attr = edge_attr)     #
            out = self.LayerNorm(x = out, batch = data.batch)
            out = self.bf_skip[i](out)
            out = self.AF(self.dense_bf_skip[i](out))
            out = out + out_res_0
            out = self.af_skip[i](out)
            results += self.readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
            results_pairs += self.pair_readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])
            bde_pairs += self.bde_readouts[i+1](x = out, rbf = data.node_rbf, edge_index_0 = edge_index_0, num_atoms = atom_batch.size()[0])

        results = scatter_add(results, index= atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        results_pairs = scatter_add(results_pairs, dim=0, index = edge_index_0, dim_size = atom_batch.size()[0]) 
        results_pairs = scatter_add(results_pairs, index=atom_batch, dim = 0, dim_size=int(data.batch.max()) + 1)
        return results.view(-1), results_pairs.view(-1), bde_pairs