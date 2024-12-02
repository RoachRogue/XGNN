import torch
import torch.nn as nn
import numpy as np
from angular_basis_layer import F_B_2D
from model import SBFTransformer, SBFTransformerGlobal, SBFTransformer_radical, SBFTransformer_radical_all, NoAttn, SBFTransformer_vectorial_preds, SBFTransformer_bde
from edge_graph import vertex_to_edge
from torch.nn import SiLU
from atom_embedding import EmbeddingBlock
from envelop import poly_envelop
from torch_geometric.data import Data
from radial_basis_layer import RadialBasis
from torch_scatter import scatter_add
from initializer import Glorot_Ortho_

class xgnn_poly_vectorial(nn.Module):
    def __init__(self, include_H = True, include_S = True, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu', ):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SBFTransformer_vectorial_preds(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads)
        self.include_H = include_H
        self.include_S = include_S

        if include_H and include_S:
            self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        elif include_H or include_S:
            self.mat_trans = torch.nn.Linear(169, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        else:
            self.mat_trans = torch.nn.Linear(256, embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_vector = (data.raw_pos[data.edge_index[0]] - data.raw_pos[data.edge_index[1]]).float()
        bond_distances = torch.norm(bond_vector, dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        if self.include_H and self.include_S:
            neo_x = data.edge_attr * envelop_para
        elif self.include_S:
            neo_x = data.edge_attr[:,:169] * envelop_para
        elif self.include_H:
            neo_x = data.edge_attr[:,169:] * envelop_para
        else:
            neo_x = torch.cat((atom_embeddings[data.edge_index[0]],atom_embeddings[data.edge_index[1]]), dim=1)
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate angles and angularexpansion
        ji_vector = (data.raw_pos[atom_i] - data.raw_pos[atom_j]).float()
        jk_vector = (data.raw_pos[atom_k] - data.raw_pos[atom_j]).float()
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        if self.include_H or self.include_S:
            neo_x = self.AF(self.emb_trans(neo_x))
        else:
            neo_x = torch.cat((neo_x, self.AF(self.rbf_trans(node_rbf))), dim=1)
            neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf, node_vector = bond_vector)
        predicted_displacement = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        predicted_position = data.raw_pos.float() + predicted_displacement

        return predicted_position

class xgnn_poly_force_full(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256, K = 2, heads = 16, mat_dim = 169, embedding_size = 128, device = 'cuda',):
        super().__init__()
        self.device = device
        self.mat_dim=mat_dim
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size).to(self.device)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5 ).to(self.device)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5).to(self.device)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim).to(self.device)
        self.fin_model = SBFTransformer(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads).to(self.device)
        
        self.mat_trans = torch.nn.Linear(mat_dim, embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, embedding_size)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        data.x = data.x.long()
        data.atom_pos.requires_grad_(True)
        data.edge_attr.requires_grad_(True)

        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1).float()
        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr[:,:self.mat_dim] * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]
        #neo_x = torch.cat((atom_embeddings[data.edge_index[0]],atom_embeddings[data.edge_index[1]]), dim=1)    # 拼接原子表示的平均会使表现下降

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0]).float()
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = torch.cat((neo_x, self.AF(self.rbf_trans(node_rbf))), dim=1)
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        result = results.sum()
        e_a_g, a_p_g = torch.autograd.grad(outputs = result, inputs = (data.edge_attr, data.atom_pos), create_graph= True)
        data.edge_attr.requires_grad_(False)
        data.edge_attr.requires_grad_(False)

        edge_grads = (data.edge_attr_grad.permute(1,2,0,3)[:,:,:,:self.mat_dim] * e_a_g[:,:self.mat_dim]).sum(dim = [3], keepdim=False) # natom x 3 x |e| -> (bsz x natm) x 3
        # scatter add by edge affiliation
        S_grads = scatter_add(src = edge_grads, index = batch, dim=-1, ).permute(2,0,1).reshape(-1,3)

        force = -(a_p_g + S_grads)

        return results, force

class xgnn_poly_noattn(nn.Module):
    def __init__(self, include_H = True, include_S = True, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu', ):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = NoAttn(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim = rbf_dim, in_channels = in_channels)
        self.include_H = include_H
        self.include_S = include_S

        if include_H and include_S:
            self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        elif include_H or include_S:
            self.mat_trans = torch.nn.Linear(169, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        else:
            raise NotImplementedError

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        if self.include_H and self.include_S:
            neo_x = data.edge_attr * envelop_para
        elif self.include_S:
            neo_x = data.edge_attr[:,:169] * envelop_para
        elif self.include_H:
            neo_x = data.edge_attr[:,169:] * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        return results

class xgnn_poly(nn.Module):
    def __init__(self, include_H = True, include_S = True, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu', ):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SBFTransformer(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads)
        self.include_H = include_H
        self.include_S = include_S

        if include_H and include_S:
            self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        elif include_H or include_S:
            self.mat_trans = torch.nn.Linear(169, 2*embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)
        else:
            self.mat_trans = torch.nn.Linear(256, embedding_size)
            Glorot_Ortho_(self.mat_trans.weight)
            torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        if self.include_H and self.include_S:
            neo_x = data.edge_attr * envelop_para
        elif self.include_S:
            neo_x = data.edge_attr[:,:169] * envelop_para
        elif self.include_H:
            neo_x = data.edge_attr[:,169:] * envelop_para
        else:
            neo_x = torch.cat((atom_embeddings[data.edge_index[0]],atom_embeddings[data.edge_index[1]]), dim=1)
        neo_x = self.AF(self.mat_trans(neo_x))

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        if self.include_H or self.include_S:
            neo_x = self.AF(self.emb_trans(neo_x))
        else:
            neo_x = torch.cat((neo_x, self.AF(self.rbf_trans(node_rbf))), dim=1)
            neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        return results

class xgnn_poly_force(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256, K = 2, heads = 16, mat_dim = 169, embedding_size = 128, device = 'cuda'):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size).to(self.device)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5 ).to(self.device)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5).to(self.device)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim).to(self.device)
        self.fin_model = SBFTransformer(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads).to(self.device)
        
        self.mat_trans = torch.nn.Linear(mat_dim, embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, embedding_size)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        data.x = data.x.long()
        data.atom_pos.requires_grad_(True)
        data.edge_attr.requires_grad_(True)

        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1).float()
        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr[:,:169] * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]
        #neo_x = torch.cat((atom_embeddings[data.edge_index[0]],atom_embeddings[data.edge_index[1]]), dim=1)    # 拼接原子表示的平均会使表现下降

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0]).float()
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = torch.cat((neo_x, self.AF(self.rbf_trans(node_rbf))), dim=1)
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        result = results.sum()
        e_a_g, a_p_g = torch.autograd.grad(outputs = result, inputs = (data.edge_attr, data.atom_pos), create_graph= True)
        data.edge_attr.requires_grad_(False)
        data.edge_attr.requires_grad_(False)

        edge_grads = (data.edge_attr_grad.permute(1,0,2) * e_a_g).sum(dim = [2], keepdim=False) #3x|e|

        S_grads_0 = scatter_add(src = edge_grads.permute(1,0), index = data.edge_index[0], dim = 0, dim_size = data.x.size()[0])
        S_grads_1 = scatter_add(src = -edge_grads.permute(1,0), index = data.edge_index[1], dim = 0, dim_size = data.x.size()[0])
        #S_grads = torch.cat(S_grads_list, dim = 0)
        S_grads = S_grads_0 + S_grads_1

        force = -(a_p_g + S_grads)

        return results, force

class xgnn_poly_global(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu', pool_option='mean'):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5 )
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SBFTransformerGlobal(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads, pool_option=pool_option)
        
        self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        return results

class xgnn_poly_radical_all(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SBFTransformer_radical_all(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads)
        
        self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf, is_cleave = data.is_cleave)
        results = self.fin_model(data = neo_data, num_graphs = data.num_graphs)
        
        return results

class xgnn_poly_radical(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, device = 'cpu'):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = 5.0, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, 5.0, 5)
        self.rbf_layer = RadialBasis(cutoff=5.0, embedding_size=rbf_dim)
        self.fin_model = SBFTransformer_radical(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads)
        
        self.mat_trans = torch.nn.Linear(338, 2*embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf, is_cleave = data.is_cleave)
        results = self.fin_model(data = neo_data, num_graphs = data.num_graphs)
        
        return results

class xgnn_poly_bde(nn.Module):
    def __init__(self, conv_layers = 4, sbf_dim=7, rbf_dim=16, in_channels = 256,  heads = 16, embedding_size = 128, attr_size = 338,device = 'cpu',cutoff=5.0):
        super().__init__()
        self.device = device
        self.AF = SiLU()
        self.emb_block = EmbeddingBlock(embedding_size=embedding_size)
        self.envelop_function = poly_envelop( cutoff = cutoff, exponent=5)
        self.sbf_layer = F_B_2D(sbf_dim, rbf_dim, cutoff, 5)
        self.rbf_layer = RadialBasis(cutoff=cutoff, embedding_size=rbf_dim)
        self.fin_model = SBFTransformer_bde(conv_layers=conv_layers, emb_size=embedding_size, sbf_dim=sbf_dim, rbf_dim=rbf_dim, in_channels = in_channels, heads = heads)
        
        self.mat_trans = torch.nn.Linear(attr_size, 2*embedding_size)
        Glorot_Ortho_(self.mat_trans.weight)
        torch.nn.init.zeros_(self.mat_trans.bias)

        self.rbf_trans = torch.nn.Linear(rbf_dim, embedding_size)
        Glorot_Ortho_(self.rbf_trans.weight)
        torch.nn.init.zeros_(self.rbf_trans.bias)

        self.emb_trans = torch.nn.Linear(embedding_size*2, in_channels)
        Glorot_Ortho_(self.emb_trans.weight)
        torch.nn.init.zeros_(self.emb_trans.bias)

    def forward(self, data):
        bond_distances = torch.norm((data.atom_pos[data.edge_index[0]] - data.atom_pos[data.edge_index[1]]), dim = 1)

        if "batch" in data._store:
            if "num_graphs" not in data._store:
                data.num_graphs = 1
            batch = torch.arange(data.num_graphs).to(self.device).repeat_interleave(data.edge_num)
        else:
            data.batch = torch.zeros(data.x.size()[0], dtype = torch.int64).to(self.device)
            batch = torch.zeros(data.edge_num, dtype = torch.int64).to(self.device)

        envelop_para = self.envelop_function(bond_distances)
        envelop_para = envelop_para[:,None]

        neo_edge_index, atom_j, atom_i, atom_k = vertex_to_edge(data.edge_index.to('cpu'), data.x.size()[0])
        neo_edge_index = neo_edge_index.to(self.device)
        neo_x = data.edge_attr * envelop_para
        neo_x = self.AF(self.mat_trans(neo_x))

        atom_embeddings = self.emb_block(data.x)
        neo_edge_attr = atom_embeddings[atom_j]

        # calculate angles and angularexpansion
        ji_vector = data.atom_pos[atom_i] - data.atom_pos[atom_j]
        jk_vector = data.atom_pos[atom_k] - data.atom_pos[atom_j]
        bond_angle_cos = torch.sum(ji_vector * jk_vector,dim=1)
        bond_angle_sin = torch.norm(torch.cross(ji_vector, jk_vector), dim = 1)
        edge_sbf = self.sbf_layer(bond_distances, torch.atan2(bond_angle_sin, bond_angle_cos), neo_edge_index[0])
        
        # calculate radial expansion
        node_rbf = self.rbf_layer(bond_distances)
        node_rbf = node_rbf * envelop_para
        neo_x = self.AF(self.emb_trans(neo_x))

        neo_data = Data(x = neo_x, edge_index = neo_edge_index, edge_attr=neo_edge_attr, batch = batch, edge_sbf = edge_sbf, node_rbf = node_rbf)
        results = self.fin_model(neo_data, edge_index_0 =data.edge_index[0], atom_batch = data.batch)
        
        return results