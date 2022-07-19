import os
import json
import time
import math
import torch
import pdb
import numpy as np

from GTLR.utils import rnns

from torch.utils.data import Dataset

import extension as ext
from extension.utils_tlg import io_utils
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torchtext
import random
import typing

class SemiSampler(SubsetRandomSampler):
    def __init__(self, indices: typing.Sequence[int], generator=None) -> None:
        super().__init__(indices, generator=generator)
        self.epoch_seed = None

    def __iter__(self):
        if self.generator is None:
            self.generator = torch.Generator()
            if self.epoch_seed:
                torch.manual_seed(self.epoch_seed)
                self.epoch_seed=None
            self.generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

def create_loaders(loader_configs):
    dsets, L = {}, {}
    for di,dt in enumerate(loader_configs.keys()):
        shuffle = True if dt == "train" else False
        drop_last = True if dt == "train" else False
        dsets[dt] = DENSE_ACNET_C3D(loader_configs[dt])
        if ext.distributed.get_world_size() > 1:
            loader_sampler = DistributedSampler(dsets[dt],  num_replicas=ext.distributed.get_world_size(), rank=ext.distributed.get_rank(), shuffle=shuffle)
            shuffle = False
        else:
        #     loader_sampler = None
        # L[dt] = torch.utils.data.DataLoader(
        #     dsets[dt],
        #     batch_size = loader_configs[dt]["batch_size"],
        #     num_workers = loader_configs[dt]["num_workers"],
        #     shuffle = shuffle, # shuffle
        #     collate_fn = dsets[dt].collate_fn,
        #     sampler = loader_sampler,
        #     drop_last= drop_last #drop_last
        # )
            loader_sampler = None
            shuffle = False
            indices = list(range(dsets[dt].__len__()))
            label_sampler = SemiSampler(indices[:dsets[dt].split_loc])
            L[dt] = torch.utils.data.DataLoader(
                    dsets[dt],
                    batch_size = max(1, int(loader_configs[dt]["batch_size"])),
                    num_workers = loader_configs[dt]["num_workers"],
                    shuffle = shuffle, # shuffle
                    collate_fn = dsets[dt].collate_fn,
                    sampler = label_sampler,
                    drop_last= drop_last #drop_last
                    )
    return dsets, L

class DENSE_ACNET_C3D(Dataset):

    def __init__(self, config):

        self.feature_path = config['features_path']
        self.ann_file_path = config['ann_file_path']
        self.embeddings_path = config['embeddings_path']
        self.data_dir = config['data_dir']
        self.min_count = config['min_count']
        self.train_max_length = config['train_max_length']
        self.test_max_length = config['test_max_length']
        self.feature_sample_num = config["feature_sample_num"]
        self.dense = config.get("dense_query", True)
        self.dataset_type = config["dataset"]
        self.proposal_sample = False
        self.proposal_shuffle = False
        self.word_dict = io_utils.load_json(config["tokens_json"])


        self.embeddings_file_path = os.path.join(self.data_dir, f'activitynet_embeddings_{self.min_count}_{self.train_max_length}.pth') 
        self.vocab_file_path = os.path.join(self.data_dir, f'activitynet_vocab_{self.min_count}_{self.train_max_length}.pickle')
        
        self.vocab = torchtext.vocab.pretrained_aliases['glove.840B.300d'](cache=config['embeddings_path'])
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.itos.extend(['<cls>'])
        self.vocab.stoi['<cls>'] = self.vocab.vectors.shape[0]
        self.vocab.itos.extend(['<end>'])
        self.vocab.stoi['<end>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat([self.vocab.vectors, torch.zeros(1, self.vocab.dim)], dim=0)
        self.word_embedding = torch.nn.Embedding.from_pretrained(self.vocab.vectors)

        self.is_training = 'train' in config['split']
        self.i3dfeat = None
        print(self.is_training)

        print('loading annotations into memory...', end=" ")
        tic = time.time()
        
        aux = json.load(open(self.ann_file_path, 'r'))

        self.dataset = aux
        
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        
        self.createIndex()
        # pdb.set_trace()
        
        self.ids   = list(self.anns.keys())
        self.epsilon = 1E-10

        self.split_loc = int(self.__len__())


    def createIndex(self):
        print("Creating index..", end=" ")
        anns = {}
        # size = int(round(len(self.dataset) * 1.))
        counter = 0
        # word_count = 1
        # self.word_dict["<pad>"] = 0
        # for row in self.dataset[:size]:

        if self.dense:
            for vid, row in self.dataset.items():
                row["video_id"] = vid
                if self.is_training:

                    if float(row['number_features']) < 10:
                        continue            # print(row) 
                    if float(row['number_features']) >= 1200 and self.dataset_type in ['tacos']:
                        continue            # print(row)
                    
                # if float(row['feature_start']) > float(row['feature_end']):
                #     # print(row)
                #     continue

                # if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                #     row['feature_end'] = float(row['number_features'])-1

                if self.is_training:

                    row['augmentation'] = 1
                    anns[counter] = row.copy()
                    counter += 1
                    continue

                row['augmentation'] = 0
                anns[counter] = row
                counter+=1
        
        else:
            for _, row in enumerate(self.dataset):
                if self.is_training:
                    if float(row['number_features']) < 10:
                        continue            # print(row) 
                    if float(row['number_features']) >= 1200 and self.dataset_type in ['tacos']:
                        continue            # print(row)
                    
                if self.is_training:

                    row['augmentation'] = 1
                    anns[counter] = row.copy()
                    counter += 1
                    continue


                row['augmentation'] = 0
                anns[counter] = row
                counter+=1


        self.anns = anns
        # for _, row in self.anns.items():
        #     for word in row["total_text"][0]:
        #         if word not in self.word_dict.keys():
        #             self.word_dict[word] = word_count
        #             word_count += 1
        # self.word_dict["<cls>"] = word_count

        # self.word_dict["<end>"] = word_count + 1

        print(" Ok! {}".format(len(anns.keys())))

    def __getitem__(self, index):
        if self.i3dfeat is None:
            self.i3dfeat = io_utils.load_hdf5(self.feature_path, verbose=False)
        ann = self.anns[index]
        if 'v_' in ann['video_id']:
            i3dfeat = self.i3dfeat[ann['video_id']]['c3d_features'][:] 
        else:
            if self.dataset_type in ["charades"]:
                try:
                    i3dfeat = self.i3dfeat[ann['video_id']]["c3d_fc6_features"][:] 
                except:
                    i3dfeat = self.i3dfeat[ann['video_id']][:] 
            elif self.dataset_type in ["tacos"]:
                i3dfeat = self.i3dfeat[ann['video_id']][:]
            else:
                i3dfeat = self.i3dfeat['v_'+ann['video_id']]['c3d_features'][:] 
        i3dfeat = torch.from_numpy(i3dfeat).float()

        # i3dfeat = "{}/{}.npy".format(self.feature_path, ann['video'].split('v_')[-1])
        # i3dfeat = np.load(i3dfeat)
        # i3dfeat = np.squeeze(i3dfeat)
        # i3dfeat = torch.from_numpy(i3dfeat)
        
        slice_num = self.feature_sample_num
        if i3dfeat.shape[0] > slice_num:
            idx = np.linspace(0, i3dfeat.shape[0]-1, num=slice_num, dtype = int)
            i3dfeat = i3dfeat[idx]

            if self.dataset_type in ["anet", "charades", "tacos"]:
                ann['feature_start'] = torch.tensor(ann['feature_timestamps'])[:,0] * (slice_num/ann['number_features'])
                ann['feature_end'] = torch.tensor(ann['feature_timestamps'])[:,1] * (slice_num/ann['number_features'])
                ann['feature_start'][ann['feature_start'] == slice_num] = slice_num - 1
                ann['feature_end'][ann['feature_end'] == slice_num] = slice_num - 1

                ann['feature_timestamps'] = torch.stack((ann['feature_start'], ann['feature_end']),dim=-1)
            else:
                ann['feature_start'] = torch.tensor(ann['feature_timestamps'])[:,0] * slice_num
                ann['feature_end'] = torch.tensor(ann['feature_timestamps'])[:,1] * slice_num
                
                ann['feature_start'][ann['feature_start'] == slice_num] = slice_num - 1
                ann['feature_end'][ann['feature_end'] == slice_num] = slice_num - 1

                ann['feature_timestamps'] = torch.stack((ann['feature_start'], ann['feature_end']),dim=-1)
            
            ann['number_features'] = slice_num
            ann['number_features'] = slice_num
            
        feat_length = i3dfeat.shape[0]
        text_tokens = []
        timestamps = []
        feature_timestamps = []
        word_labels = []

        sample_idx_list = range(len(ann['total_text']))

        if self.proposal_sample:
            sample_num = random.randint(min(len(sample_idx_list),100), min(len(sample_idx_list),128))
            sample_idx_list = random.sample(sample_idx_list, sample_num)

        if not self.proposal_shuffle:
            sample_idx_list = sorted(sample_idx_list)
        for i in sample_idx_list:
            sentence_tokens = ann['total_text'][i]
            if self.is_training:
                raw_tokens = sentence_tokens[:self.train_max_length]
            else:
                raw_tokens = sentence_tokens[:self.test_max_length]
            # input_raw_tokens = ["<cls>"] + raw_tokens + ["<end>"]
            input_raw_tokens = raw_tokens
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 1438) for w in input_raw_tokens], dtype=torch.long)
            sentence_emb = self.word_embedding(word_idxs)
            text_tokens.append(sentence_emb)
            timestamps.append(ann["timestamps"][i])
            feature_timestamps.append(ann["feature_timestamps"][i])
            word_labels.append(torch.tensor([self.word_dict.get(w.lower(), 0) for w in raw_tokens], dtype=torch.long))
        # indices = self.vocab.tokens2indices(raw_tokens)
        # tokens = [self.embedding_matrix[index] for index in indices]
        # tokens = torch.stack(tokens)


        time_start = torch.tensor(timestamps)[:,0]
        time_end = torch.tensor(timestamps)[:,1]
        # feature_start = torch.tensor(feature_timestamps)[:,0]
        # feature_end = torch.tensor(feature_timestamps)[:,1]

        feat_mask_total = []
        for i, (feat_start, feat_end) in enumerate(feature_timestamps):
            # feat_start = int(feat_start*feat_length)
            # feat_end = int(feat_end*feat_length)
            feat_mask = torch.zeros(feat_length)
            feat_start = int(feat_start)
            feat_end = int(feat_end)
            if feat_start == feat_end:
                feat_end = min(feat_length, feat_end+1)
            if feat_start == feat_end:
                feat_end = max(0, feat_start-1)
            feat_mask[feat_start:feat_end] = 1.0
            feat_mask_total.append(feat_mask)
        


        return index, i3dfeat, text_tokens, time_start, time_end, ann['duration'], ann['total_text'], ann['video_id'], feat_mask_total,word_labels


    def collate_fn(self, batch):
        transposed_batch = list(zip(*batch))

        index      = transposed_batch[0]
        videoFeat  = transposed_batch[1]
        text_tokens     = transposed_batch[2]
        time_start      = transposed_batch[3]
        time_end        = transposed_batch[4]
        duration   = transposed_batch[5]
        raw_text_tokens = transposed_batch[6]
        video_id = transposed_batch[7]
        # feature_start = transposed_batch[8]
        # feature_end = transposed_batch[9]
        feat_mask = transposed_batch[8]
        word_labels = transposed_batch[9]




        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat, instant_padding=False, padding_num=256)
        time_start, num_proposal = rnns.pad_sequence(time_start, instant_padding=False, padding_num=256)
        time_end, _ = rnns.pad_sequence(time_end, instant_padding=False, padding_num=256)
        # feature_start, _ = rnns.pad_sequence(feature_start, instant_padding=False, padding_num=256)
        # feature_end, _ = rnns.pad_sequence(feature_end, instant_padding=False, padding_num=256)
        feat_mask_list = []
        for i in range(len(feat_mask)):
            feat_mask_list = feat_mask_list + feat_mask[i] 
        
        feat_mask, _ = rnns.pad_sequence(feat_mask_list, instant_padding=False, padding_num=256)
        
        sentences_emb_list = []
        for _, sentences in enumerate(text_tokens):     
            for s_emb in sentences:
                sentences_emb_list.append(s_emb)

        # word_label_list = []
        # for _, word_label in enumerate(word_labels):     
        #     for s_label in word_label:
        #         word_label_list.append(s_label)


        sentences_emb, sentences_length = rnns.pad_sequence(sentences_emb_list, instant_padding=False, padding_num=256)

        # word_labels, _  = rnns.pad_sequence(word_label_list, instant_padding=True, padding_num=self.train_max_length+2)


        text_emb, text_len = rnns.pad_sequence([torch.cat(sen, dim=0) for sen in text_tokens], instant_padding=True, padding_num=self.train_max_length+2)
        # tokens, tokens_lengths   = rnns.pad_sequence(tokens)

        word_labels, _  = rnns.pad_sequence([torch.cat(wl, dim=0) for wl in word_labels], instant_padding=True, padding_num=self.train_max_length+2)



        return {
                # 'index':index, #pair's index
                'raw_text_tokens': raw_text_tokens, # list[list]
               'duration': torch.tensor(duration), 
               'video_id': video_id,

               'time_start':  time_start, # tensor [B, max_proposal_num]
               'time_end':  time_end, # tensor [B, max_proposal_num]
               'feature_mask': feat_mask,
               'proposal_num': num_proposal, # tensor [B,]

               'videoFeat': videoFeat, # tensor [B, max_T, D]
               'videoFeat_lengths': videoFeat_lengths, # tensor [B,]
               'textEmbedding': text_emb, # tensor [B, max_len_text, D]
               'textEmbedding_lengths': text_len, # tensor [B,]

               "sentences_emb": sentences_emb, # tensor(sum(proposal_num), max_length, D)
               "sentences_lengths": sentences_length,  # tensor(sum(proposal_num))

               "word_labels" : word_labels
        }

    def __len__(self):
        return len(self.ids)
