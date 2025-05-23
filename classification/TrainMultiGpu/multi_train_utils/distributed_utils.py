import os

import torch
import torch.distributed as dist

def init_distributed_mode(args):
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    """
    Get the world size (number of processes) in distributed training.
    """
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    """
    Get the rank (process ID) in distributed training.
    """
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    else:
        return 0

def is_main_process():
    """
    Check if the current process is the main process in distributed training.
    """
    return get_rank() == 0

def reduce_value(value, average=True):
    """
    Reduce a value across all processes in distributed training.
    """
    word_size = get_world_size()
    if word_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= word_size

    return value