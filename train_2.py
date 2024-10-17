# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
import textgrid
from tqdm import tqdm
import time
from contextlib import nullcontext
import shutil
from pathlib import Path
from tqdm import tqdm

# ML
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb

# Local
from supervoice_flow.config import config
from supervoice_flow.model import AudioFlow
from supervoice_flow.tensors import count_parameters, probability_binary_mask, drop_using_mask, random_interval_masking
from training.dataset import load_distorted_loader, load_clean_loader, load_audio_data

# Train parameters
train_experiment = "test-rb"
train_project="supervoice-flow-multigpu"
train_datasets = "/mnt/afs/zhangjinouwen/Dataset/tmp_dataset"
train_eval_datasets = "/mnt/afs/zhangjinouwen/Dataset/tmp_dataset"
train_duration = 15 # seconds, 15s x 5 (batches) = 75s per GPU
train_source_experiment = None
train_auto_resume = False
train_batch_size = 5 # Per GPU
train_clean = True
train_grad_accum_every = 16 # 16x2 = 32 GPU to match paper
train_steps = 600000 # Directly matches paper
train_loader_workers = 5
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_evaluate_every = 1
train_evaluate_batch_size = 10
train_lr_start = 5e-5
train_lr_max = 1e-5
train_decay_steps = 100000
train_warmup_steps = 500000
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 0.2
train_sigma = 1e-5

# Train
def main():

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("/mnt/afs/zhangjinouwen/Project/TTSFLOW")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    # set_seed(42) enabling this would force each GPU to have same samples
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")

    # train_loader = load_clean_loader(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)
    replay_buffer = load_audio_data(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)
    replay_buffer_eval = load_audio_data(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size, eval=True)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0
    raw_model = AudioFlow(config)
    model = raw_model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr_max, betas=[0.9, 0.99], weight_decay=0.01, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Load
    source = None
    if (output_dir / f"{train_experiment}.pt").exists():
        source = train_experiment
    elif train_source_experiment and (output_dir / f"{train_source_experiment}.pt").exists():
        source = train_source_experiment

    if train_auto_resume and source is not None:
        accelerator.print("Resuming training...")
        print(str(output_dir / f"{source}.pt"))
        checkpoint = torch.load(str(output_dir / f"{source}.pt"), map_location="cpu")
        raw_model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        accelerator.print(f'Loaded at #{step}')
        
    # Accelerate
    model, optim = accelerator.prepare(model, optim)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "batch_size": train_batch_size, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': raw_model.state_dict(), 

            # Optimizer
            'step': step,
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = train_lr_start
        elif step < train_warmup_steps + train_decay_steps:
            decay_step = step - train_warmup_steps
            lr = train_lr_start - (train_lr_start - train_lr_max) * (decay_step / train_decay_steps)
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        successful_cycles = 0
        failed_steps = 0
        while successful_cycles < train_grad_accum_every:
            with accelerator.accumulate(model):
                with accelerator.autocast():

                    spec = replay_buffer.sample().to(device=device)

                    batch_size = spec.shape[0]
                    seq_len = spec.shape[1]

                    spec = (spec - config.audio.norm_mean) / config.audio.norm_std
                    if not train_clean:
                        spec_aug = (spec_aug - config.audio.norm_mean) / config.audio.norm_std

                    # Prepare target flow (CFM)
                    times = torch.rand((batch_size,), dtype = spec.dtype, device = device)
                    t = rearrange(times, 'b -> b 1 1')
                    source_noise = torch.randn_like(spec, device=device)
                    noise = (1 - (1 - train_sigma) * t) * source_noise + t * spec
                    flow = spec - (1 - train_sigma) * source_noise

                    # Masking 
                    # 70% - 100% of the sequence is masked, with segments of at least 10 frames
                    mask = random_interval_masking(batch_size, seq_len, 
                                                   min_size = 10, 
                                                   min_count = int(seq_len * 0.7), 
                                                   max_count = seq_len, 
                                                   device = device)

                    # Drop everything for unconditional generation
                    # 0.1 probability of full mask
                    conditional_drop_mask = probability_binary_mask(shape = (batch_size,), true_prob = 0.1, device = device)

                    # Merge masks
                    mask = drop_using_mask(source = mask, replacement = True, mask = conditional_drop_mask)

                     # Prepare condition spec
                    if not train_clean:
                        condition_spec = torch.where(mask.unsqueeze(-1), spec_aug, spec)
                    else:
                        condition_spec = drop_using_mask(source = spec, replacement = 0, mask = mask)

                    # Train step
                    predicted, loss = model(

                        # Audio
                        audio = condition_spec, 
                        noise = noise, 

                        # Time
                        times = times, 

                        # Loss
                        mask = mask, 
                        target = flow,
                        mask_loss = True
                    )
                    # e_t = time.time()
                    # print(f'forward cost {e_t - s_t} sec')

                    # s_t = time.time()
                    # Backprop
                    optim.zero_grad()
                    # print(f'RANK:{accelerator.process_index}','LOSS:',loss)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()
                    # e_t = time.time()
                    # print(f'backward & step cost {e_t - s_t} sec')
                    # Log skipping step
                    if optim.step_was_skipped:
                        failed_steps = failed_steps + 1
                        if torch.isnan(loss).any():
                            accelerator.print("Step was skipped with NaN loss")
                        else:
                            accelerator.print("Step was skipped")
                        if failed_steps > 80:
                            raise Exception("Too many failed steps")
                    else:
                        successful_cycles = successful_cycles + 1
                        failed_steps = 0
                    # total_e_t = time.time()
                    # print(f'1 iter cost {total_e_t - total_s_t} sec')

        return loss, predicted, flow, lr


    def eval_logp():
        model.eval()

        logp_list = []
        logp_per_dim_list = []

        for i in range(20):
            spec = replay_buffer_eval.sample().to(device=device)
            with torch.no_grad():
                logp = model.logp(audio=spec)
                bits_ratio = torch.prod(torch.tensor(spec.shape[1:], device=spec.device)) * torch.log(torch.tensor(2.0, device=spec.device))
                logp_per_dim = logp / bits_ratio
                logp_list.append(logp)
                logp_per_dim_list.append(logp_per_dim)

        logp_all = torch.concatenate(logp_list,dim=0)
        logp_per_dim_all = torch.concatenate(logp_per_dim_list,dim=0)

        return logp_all, logp_per_dim_all


    accelerator.print("Training started at step", step)
    while step < train_steps:
        start = time.time()
        loss, predicted, flow, lr = train_step()
        end = time.time()
        # print(f'time={end - start} sec')
        # Advance
        step = step + 1

        if step % (train_log_every*100) == 0:
            logp_all, logp_per_dim_all = eval_logp()

        # Summary
        if step % train_log_every == 0 and accelerator.is_main_process:
            if step % (train_log_every*100) == 0:
                accelerator.log({
                    "learning_rate": lr,
                    "loss": loss,
                    "predicted/mean": predicted.mean(),
                    "predicted/max": predicted.max(),
                    "predicted/min": predicted.min(),
                    "target/mean": flow.mean(),
                    "target/max": flow.max(),
                    "target/min": flow.min(),
                    "logp_per_dim": logp_per_dim_all.mean()
                }, step=step)
            else:
                accelerator.log({
                    "learning_rate": lr,
                    "loss": loss,
                    "predicted/mean": predicted.mean(),
                    "predicted/max": predicted.max(),
                    "predicted/min": predicted.min(),
                    "target/mean": flow.mean(),
                    "target/max": flow.max(),
                    "target/min": flow.min(),
                }, step=step)
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}, time={end - start} sec')
        
        # Evaluate
        # if step % train_evaluate_every == 0:
        #     accelerator.print("Evaluating...")
        #     mos = train_eval()
        #     accelerator.print(f"Step {step}: MOS={mos}")
        #     accelerator.log({"eval/mos": mos}, step=step)
        
        # Save
        if step % train_save_every == 0 and accelerator.is_main_process:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()
