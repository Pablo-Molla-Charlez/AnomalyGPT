from header import *
from .samplers import DistributedBatchSampler
from .sft_dataset import *
from .mvtec import *
from .visa import VisaDataset
from . import all_supervised_with_cn
from torch.utils.data import Subset
'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''


""" ORIGINAL load_sft_dataset with complete dataset 
def load_sft_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler
"""
class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.collate = dataset.collate
    def collate(self, batch):
        return self.dataset.collate(batch)

""" NEW load_sft_dataset with limited samples """
def load_sft_dataset(args, limit_samples=None):
    data = SupervisedDataset(args['data_path'], args['image_root_path'])

    if limit_samples is not None:
        indices = list(range(len(data)))
        subset_indices = indices[:limit_samples]
        data = CustomSubset(data, subset_indices)

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


""" ORIGINAL load_mvtec_dataset with complete dataset
def load_mvtec_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = MVtecDataset('../data/mvtec_anomaly_detection')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler
"""

""" NEW load_mvtec_dataset with limited samples """

def load_mvtec_dataset(args, limit_samples=None):
    data = MVtecDataset('../data/mvtec_anomaly_detection')

    if limit_samples is not None:
        indices = list(range(len(data)))
        subset_indices = indices[:limit_samples]
        data = CustomSubset(data, subset_indices)

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


def load_visa_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = VisaDataset('../data/VisA')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


def load_supervised_dataset_with_cn(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = all_supervised_with_cn.SupervisedDataset('../data/all_anomalygpt')

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler