from contextlib import redirect_stdout

def save_cfg(cfg, path):
    with open(path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

def load_cfg(dir):
    from mnists.config import cfg
    cfg.merge_from_file(dir)
    return cfg
