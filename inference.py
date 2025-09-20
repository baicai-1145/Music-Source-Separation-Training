# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import threading
from queue import Queue
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


def _writer_worker(task_queue: Queue):
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break
        try:
            backend = item['backend']
            if backend == 'torchaudio':
                import torchaudio
                torchaudio.save(item['path'], item['tensor'], item['sr'])
            else:
                sf.write(item['path'], item['array'], item['sr'], subtype=item['subtype'])
        except Exception as e:
            print(f"[async_write] error writing {item['path']}: {e}")
        finally:
            task_queue.task_done()


def run_folder(model, args, config, device, verbose: bool = False):
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : Namespace
        Arguments containing input folder, output folder, and processing options.
    config : Dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    # async writing setup
    writer_queue = None
    writer_threads = []
    if getattr(args, 'async_write', False):
        writer_queue = Queue(maxsize=8)
        num_workers = max(1, int(getattr(args, 'async_workers', 2)))
        for _ in range(num_workers):
            t = threading.Thread(target=_writer_worker, args=(writer_queue,), daemon=True)
            t.start()
            writer_threads.append(t)

    for path in mixture_paths:
        print(f"Processing track: {path}")
        track_t0 = time.time()
        try:
            t0 = time.time()
            if getattr(args, 'prefer_torchaudio_io', False):
                import torchaudio
                wav, sr0 = torchaudio.load(path)
                # to stereo if needed
                if wav.shape[0] == 1 and getattr(config.audio, 'num_channels', 2) == 2:
                    wav = wav.repeat(2, 1)
                # resample if needed
                if sr0 != sample_rate:
                    if getattr(args, 'resample_on_gpu', False) and torch.cuda.is_available():
                        wav = wav.to('cuda')
                        resampler = torchaudio.transforms.Resample(orig_freq=sr0, new_freq=sample_rate).to('cuda')
                        wav = resampler(wav).cpu()
                    else:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr0, new_freq=sample_rate)
                        wav = resampler(wav)
                mix = wav.numpy()
                sr = sample_rate
            else:
                mix, sr = librosa.load(path, sr=sample_rate, mono=False, res_type=getattr(args, 'librosa_res_type', 'kaiser_fast'))
            t1 = time.time()
            print(f"[profile] load: {(t1 - t0):.2f}s (backend={'torchaudio' if getattr(args, 'prefer_torchaudio_io', False) else 'librosa'})")
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                t0 = time.time()
                mix, norm_params = normalize_audio(mix)
                t1 = time.time()
                print(f"[profile] normalize: {(t1 - t0):.2f}s")

        # model inference (synchronize to measure GPU work precisely)
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        print(f"[profile] demix(): {(t1 - t0):.2f}s")

        if args.use_tta:
            t0 = time.time()
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)
            t1 = time.time()
            print(f"[profile] TTA: {(t1 - t0):.2f}s")

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(path))[0]

        output_dir = os.path.join(args.store_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        write_total = 0.0
        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    t0 = time.time()
                    estimates = denormalize_audio(estimates, norm_params)
                    t1 = time.time()
                    print(f"[profile] denormalize[{instr}]: {(t1 - t0):.2f}s")

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = os.path.join(output_dir, f"{instr}.{codec}")
            t0 = time.time()
            backend = 'torchaudio' if (codec == 'wav' and getattr(args, 'use_torchaudio_save', False)) else 'soundfile'
            if writer_queue is not None:
                if backend == 'torchaudio':
                    tensor = torch.from_numpy(estimates)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    writer_queue.put({'backend': 'torchaudio', 'tensor': tensor.cpu(), 'sr': sr, 'path': output_path})
                else:
                    writer_queue.put({'backend': 'soundfile', 'array': estimates.T, 'sr': sr, 'path': output_path, 'subtype': subtype})
                t1 = time.time()
                dt = (t1 - t0)
                write_total += dt
                print(f"[profile] write[{instr}]-queued: {dt:.2f}s -> {output_path} (backend={backend})")
            else:
                if backend == 'torchaudio':
                    import torchaudio
                    tensor = torch.from_numpy(estimates)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    torchaudio.save(output_path, tensor, sr)
                else:
                    sf.write(output_path, estimates.T, sr, subtype=subtype)
                t1 = time.time()
                write_total += (t1 - t0)
                print(f"[profile] write[{instr}]: {(t1 - t0):.2f}s -> {output_path} (backend={backend})")
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{instr}.jpg")
                t0 = time.time()
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                t1 = time.time()
                print(f"[profile] draw_spectrogram[{instr}]: {(t1 - t0):.2f}s -> {output_img_path}")

        track_t1 = time.time()
        print(f"[profile] track total: {(track_t1 - track_t0):.2f}s (write total: {write_total:.2f}s)")

    # finalize async writers
    if writer_queue is not None:
        writer_queue.join()
        for _ in writer_threads:
            writer_queue.put(None)
        for t in writer_threads:
            t.join()

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    # Optional: torch.compile acceleration (PyTorch 2.0+)
    try:
        if getattr(args, 'torch_compile', False):
            # Disable CUDA Graphs globally for Inductor to avoid rotary embedding cache issues
            try:
                from torch._inductor import config as inductor_config
                inductor_config.cudagraphs = False
                try:
                    inductor_config.triton.cudagraphs = False
                except Exception:
                    pass
            except Exception:
                pass

            compile_kwargs = {
                'backend': getattr(args, 'compile_backend', 'inductor'),
                'fullgraph': getattr(args, 'compile_fullgraph', False),
                'dynamic': getattr(args, 'compile_dynamic', False),
            }
            # Note: when passing options, PyTorch 2.8 disallows also passing 'mode'.
            # We purposely omit 'mode' to avoid the "Either mode or options" error.
            # If DataParallel, compile the underlying module and rewrap
            if isinstance(model, nn.DataParallel):
                compiled = torch.compile(model.module, **compile_kwargs)  # type: ignore[attr-defined]
                model = nn.DataParallel(compiled, device_ids=args.device_ids)
            else:
                model = torch.compile(model, **compile_kwargs)  # type: ignore[attr-defined]
            print("torch.compile is enabled with:", compile_kwargs)
    except Exception as e:
        print(f"Warning: torch.compile failed to initialize ({e}). Proceeding without compilation.")

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)

