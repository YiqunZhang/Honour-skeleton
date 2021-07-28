import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import TensorDataset
import math


benchmarks = {
    # 'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu': ('ntu/xsub',),
    # 'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'ntu120': ('ntu120/xsub',),
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}
# parts = {'val'}

multires = 3

class Embedder_DCT:
    def __init__(self):
        self.frm_len = 300.0
        self.kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            # 'periodic_fns': [torch.sin, torch.cos],
            'periodic_fns': [torch.cos],
        }

        self.create_embedding_fn()

    def get_out_dim(self):
        return int(2. ** self.kwargs['num_freqs']) + 1

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x, y: x)  # with x
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = []
        for k in range(1, N_freqs+1):
            freq_bands.append(math.pi / self.frm_len * k)  # This is DCT

        freq_bands = torch.tensor(freq_bands)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * p_fn(freq * (frm_idx + 1/2))))  # this is DCT
                # embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (torch.ones_like(x) * p_fn(freq * (frm_idx + 1/2))))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        t_len_all = inputs.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = inputs[:, :, t_idx, :, :].unsqueeze(2)
            # new_time_list = torch.cat([fn(a_series, t_idx) for fn in self.embed_fns], dim)  # DCT

            # To try positional encoding
            new_time_list = []
            for fn in self.embed_fns:
                a_new_one = fn(a_series, t_idx)
                new_time_list.append(a_new_one)
            new_time_list = torch.cat(new_time_list, dim)

            # To sum encodes
            # new_time_list = None
            # for fn in self.embed_fns:
            #     if new_time_list is None:
            #         new_time_list = fn(a_series, t_idx)
            #     else:
            #         new_time_list += fn(a_series, t_idx)

            # print('new_time_list: ', new_time_list.squeeze())
            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        return rtn


class Embedder:
    def __init__(self):
        multires = -1
        self.kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.create_embedding_fn()

    def get_out_dim(self):
        return int(2. ** self.kwargs['num_freqs']) + 1

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: x * p_fn(freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, dim):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim)


def gen_nerf_data(fea_type, dataset_type):
    if fea_type == 'joint':
        # save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
        #             'data/{}/{}_data_jnt_dct_{}.npy'


        save_name = '/home/ankin/FP-data/NTU_120/{}/{}_data_jnt_dct_{}_pos2.npy'
        # save_name = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/{}/{}_data_jnt_nerf.npy'
    elif fea_type == 'bone':
        # save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
        #             'data/{}/{}_data_bon_dct_{}.npy'
        save_name = '../data/{}/{}_data_bon_dct_{}.npy'
    else:
        raise NotImplementedError
    print('save name: ', save_name)

    for benchmark in benchmarks[dataset_type]:
        for part in parts:
            if fea_type == 'joint':
                # data_path = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                #                'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part)
                data_path = '/home/ankin/FP-data/NTU_120/{}/{}_data_joint.npy'.format(benchmark, part)
                # data_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data' \
                #                 '/{}/{}_data_joint.npy'.format(benchmark, part)
                # data = np.load('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
                #                '/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
            elif fea_type == 'bone':
                data_path = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/' \
                            'MS-G3D/data/{}/{}_data_bone.npy'.format(benchmark, part)
            else:
                raise NotImplementedError

            data = np.load(data_path.format(benchmark, part), mmap_mode='r')
            print('load data path: ', data_path)

            # For debugging
            # data = data[:100]
            # data = torch.ones([1, 1, 4, 1, 1])

            N, C, T, V, M = data.shape
            print('data shape: ', data.shape)
            print('saving to: ', save_name.format(benchmark, part, multires))

            fp_sp = open_memmap(
                save_name.format(benchmark, part, multires),
                dtype='float32',
                mode='w+',
                shape=(N, (multires+1)*3, T, V, M))
                # shape=(N, multires*6 + 3, T, V, M))
                # shape=(N, 3, T, V, M))

            print(benchmark, part)

            # an_embed = Embedder()
            an_embed = Embedder_DCT()
            load_bch_sz = 3000

            a_dataset = TensorDataset(torch.tensor(data))
            a_dataloader = torch.utils.data.DataLoader(
                dataset=a_dataset,
                batch_size=load_bch_sz,
                shuffle=False,
                num_workers=4,
                drop_last=False
            )

            for bch_idx, a_bch in enumerate(a_dataloader):
                print('bch idx: ', bch_idx, 'a bch: ', a_bch[0].shape)
                a_piece = an_embed.embed(a_bch[0].to('cuda'), dim=1).cpu().numpy()
                print('piece shape: ', a_piece.shape)
                # print('a_piece: ', a_piece[0, 0, :, 0, 0])
                # print('a_piece: ', a_piece[0, 1, :, 0, 0])
                # print('a_piece: ', a_piece[0, 2, :, 0, 0])
                fp_sp[bch_idx * load_bch_sz:(bch_idx + 1) * load_bch_sz] = a_piece
            print('fp_sp: ', fp_sp.shape)


if __name__ == '__main__':
    gen_nerf_data('joint', 'ntu120')
