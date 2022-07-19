
import argparse
from extension import backbones
from GTLR.model.GTLR import create_gtlr

# from extension.config import get_cfg

models_dict = {"GTLR": create_gtlr}

def dataset_specific_config_update(config, dset):
    mconfig = config["arch_params"]
    # Query Encoder
    # mconfig["query_enc_emb_idim"] = len(list(dset.wtoi.keys()))
    # mconfig["loc_word_emb_vocab_size"] = len(list(dset.wtoi.keys()))
    # mconfig["dataset"] = dset.config["dataset"]
    mconfig["word_pred_dim"] = len(list(dset.word_dict))
    return config

def options():
    pass

def make(cfg: argparse.Namespace):
    backbone = backbones.make(cfg)
    # print(cfg)
    model = models_dict[cfg["arch"]](cfg, backbone = backbone)
    return model
