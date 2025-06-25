from .dsgd import DSgd

__all__ = [
    "load_optimizer",
    "DSgd",
]

def load_optimizer(args, org_optimizer, parameters):
    # DSGD
    optimizer = DSgd(args, org_optimizer, parameters)

    return optimizer

