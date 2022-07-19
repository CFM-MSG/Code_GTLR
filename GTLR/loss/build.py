

from GTLR.loss.GTLR_loss import create_GTLR_Loss


loss_dict = {"gtlr_loss": create_GTLR_Loss}

def options():
    pass

def make(cfg):
    loss_type = cfg["loss_type"]
    loss_meters = cfg["loss_params"]
    return loss_dict[loss_type](loss_meters)