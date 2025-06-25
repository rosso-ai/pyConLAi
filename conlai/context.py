from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class ConLArguments:
    server_url: str = "localhost:9200"
    repo_name: str = "conlai"
    ws_max_size: int = 1048576000
    random_seed: int = 42
    device: str = "cpu"

    # train args
    fed_optimizer: str = "adpkei"
    comm_round: int = 10
    batch_size: int = 100
    inner_loop: int = 10
    eta: float = 0.1

    # data args
    data_cache_dir: str = "./data/"
    partition_method: str = "hetero"
    partition_alpha: float = 10.

    # poc-mode
    worker_num: int = 1

    @classmethod
    def from_yml(cls, yml_path: str):
        loaded = OmegaConf.load(yml_path)
        read_conf = ConLArguments(**loaded.ns_vfl)

        base_conf = OmegaConf.structured(ConLArguments)
        merged = OmegaConf.merge(base_conf, read_conf)
        return cls(**merged)
