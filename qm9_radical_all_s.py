import os
from pathlib import Path
import torch
import numpy as np
from multiprocessing import Pool
from utils import read_xyz_allprop
from atom_graph import calculate_Dij, gen_bonds_mini
from torch_geometric.data import Data, InMemoryDataset
import pickle
from ase.io import read

atom_ref = torch.zeros(12,10)
atom_ref[9] = torch.tensor([torch.nan,-0.497912,torch.nan,torch.nan,torch.nan,torch.nan,
                            -37.844411,-54.581501,-75.062219,-99.716370])

# all posible cleave_position in each molecule(atom index in both end of bond)
with open('./scripts/cleave_pos_record.pkl','rb') as f:
    cleave_pos_rec = pickle.load(f)
with open('./scripts/all_frags_rec.pickle','rb') as f:
    all_frags_rec = pickle.load(f)
with open('./scripts/reactions.pickle','rb') as f:
    reactions = pickle.load(f)
with open('./scripts/reaction_count.pickle','rb') as f:
    reaction_count = pickle.load(f)

# key:value, index of frag: H of the optimized frag
new_frags_h_label = torch.load('./raw/new_frags_label.pt')
frags_h_label  = torch.load('./raw/frags_h_label.pt')
mol_list = read(f'./raw/new_frags_all_prop.extxyz',index=':')
legacy_mol_list = read(f'./raw/frags_all_prop.extxyz',index=':')
mol_list = [mol for mol in mol_list if len(mol.get_atomic_numbers())!=1]
index_list = [list(mol.info.keys())[-1] for mol in mol_list]
legacy_mol_list = [mol for mol in legacy_mol_list if len(mol.get_atomic_numbers())!=1]
legacy_index_list = [list(mol.info.keys())[-1] for mol in legacy_mol_list]
frags_label_dict = {eval(index_list[i].split('.')[0]):new_frags_h_label[i] for i in range(len(new_frags_h_label))}
legacy_label_dict = {eval(legacy_index_list[i].split('.')[0].split('_')[-1].lstrip('0')):frags_h_label[i] for i in range(len(frags_h_label)) if legacy_index_list[i]!='frag_000000.log'}
legacy_label_dict.update({0:frags_h_label[6479]})
legacy_label_dict.update({1:torch.tensor(0.)})
legacy_label_dict.update({165:torch.tensor(0.)})
frags_label_dict.update(legacy_label_dict)

final_id_2 = np.load('./final_id_2.npy',allow_pickle=False)
        # edge_attrs will be directly load from other files
def mapping(mol_object):
    Dij = calculate_Dij(mol_object.R)
    edge_index = gen_bonds_mini(Dij, cutoff=5.0)
    is_cleave = torch.zeros_like(edge_index[0])
    cleave_en = torch.zeros_like(is_cleave, dtype=torch.float32)
    reaction_idx = torch.zeros_like(is_cleave, dtype=torch.long)-1
    reaction_num = torch.zeros_like(is_cleave, dtype=torch.long)
    all_pairs = cleave_pos_rec[mol_object.idx]
    # need an injection from cleave_pos 2 frags_record
    for i,pair in enumerate(edge_index.T.tolist()):
        if pair in all_pairs:
            pair = pair
        elif pair[::-1] in all_pairs:
            pair = pair[::-1]
        else:
            continue
        pair_idx = all_pairs.index(pair)
        cleave_frags = all_frags_rec[mol_object.idx][pair_idx]
        if cleave_frags != [] and cleave_frags[0] in frags_label_dict and cleave_frags[1] in frags_label_dict:
            if cleave_frags[0] in final_id_2 and cleave_frags[1] in final_id_2:
                s_radical, e_radical = cleave_frags
                is_cleave[i]=1
                # set disso energy to delta-H, in eV
                cleave_en[i] = frags_label_dict[cleave_frags[0]]+ frags_label_dict[cleave_frags[1]]-(mol_object.Label[0][9]-atom_ref[9][mol_object.Z].sum())*27.211385056
                reaction_idx[i] = reactions[s_radical][e_radical]
                reaction_num[i] = reaction_count[reactions[s_radical][e_radical]]

    return Data(x = mol_object.Z, edge_index = edge_index, y = mol_object.Label, edge_num = edge_index.size()[1],
          idx = mol_object.idx, atom_pos = mol_object.R, is_cleave=is_cleave.bool(), cleave_en=cleave_en, bde_idx =reaction_idx, bde_num = reaction_num)

def paralle(mol_list):
    P = Pool(processes=int(os.cpu_count()))
    datas = P.map(func=mapping, iterable=mol_list)
    P.close()
    P.join()
    return datas

class QM9_radical_all_s2(InMemoryDataset):
    def __init__(self, root = '.', input_file = './sdfs/qm9U0_std.sdf', transform = None, pre_transform = None, pre_filter = None, is_small = None):
        self.root = Path(root)
        self.input_file = input_file
        self.prefix = input_file.split('.')[0]
        self.suffix = input_file.split('.')[1]
        self.is_small = is_small
        if '/' in self.input_file:
            self.prefix = self.input_file.split('/')[-1].split('.')[0]
            self.suffix = self.input_file.split('/')[-1].split('.')[1]
        super(QM9_radical_all_s2, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.input_file
    
    @property
    def processed_file_names(self):
        if self.is_small is None:
            return "qm9_radical_all_s2.pt"
        else:
            return "qm9_radical_s2_%d.pt"%(self.is_small)
    
    def download(self):
        pass

    def process(self):
        assert self.suffix in ['xyz'], "file type not supported"
        if self.suffix == 'xyz':
            mol_list = read_xyz_allprop(self.input_file)
        if self.is_small is None:
            datas = paralle(mol_list)
            torch.save(self.collate(datas),self.processed_dir +'/qm9_radical_all_s2.pt')
        else:
            datas = paralle(mol_list[:self.is_small])
            torch.save(self.collate(datas),self.processed_dir +'/qm9_radical_s2_%d.pt'%(self.is_small))
        
        
        print('done')

if __name__=='__main__':
    import datetime
    start_time = datetime.datetime.now()

    torch.multiprocessing.set_sharing_strategy('file_system')

    import sys
    from torch.utils.data import dataloader
    from multiprocessing.reduction import ForkingPickler
 
    default_collate_func = dataloader.default_collate
 
 
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
 
    setattr(dataloader, 'default_collate', default_collate_override)
 
    for t in torch._storage_classes:
      if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
      else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]
            
    dataset = QM9_radical_all_s2(input_file='./raw/qm9_origin.xyz')
    end_time = datetime.datetime.now()
    print(f'time consumed: {-(start_time - end_time).total_seconds() :.2f}')