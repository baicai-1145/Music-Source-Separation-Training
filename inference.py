# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from pathlib import Path

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


import torchaudio as ta
from torchaudio import functional as AF


def _resample_if_needed(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample waveform [C, T] to target_sr using torchaudio if needed.
    """
    if sr == target_sr:
        return waveform
    return AF.resample(waveform, sr, target_sr)


def _load_audio_ta(path: str, target_sr: int) -> (np.ndarray, int):
    """
    Load audio with torchaudio for maximum speed and return ndarray [C, T].
    """
    waveform, sr = ta.load(path)  # [C, T], float32 in [-1, 1]
    waveform = _resample_if_needed(waveform, sr, target_sr)
    return waveform.numpy(), target_sr


def _save_audio_ta(path: str, data: np.ndarray, sr: int, codec: str, subtype: str):
    """
    Save audio with torchaudio. `data` expected [C, T] float32 in [-1,1].

    codec: 'wav' | 'flac'
    subtype: 'FLOAT' | 'PCM_16' | 'PCM_24' (mapped to torchaudio encoding)
    """
    tensor = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    assert tensor.dim() == 2 and tensor.shape[0] in (1, 2), "Expected [C, T] tensor"

    fmt = 'FLAC' if codec.lower() == 'flac' else 'WAV'

    if fmt == 'FLAC':
        bits = 24 if subtype == 'PCM_24' else 16
        ta.save(path, tensor, sr, format=fmt, bits_per_sample=bits)
    else:  # WAV
        if subtype == 'FLOAT':
            ta.save(path, tensor, sr, format=fmt, encoding='PCM_F', bits_per_sample=32)
        elif subtype == 'PCM_24':
            ta.save(path, tensor, sr, format=fmt, encoding='PCM_S', bits_per_sample=24)
        else:  # default 16-bit signed
            ta.save(path, tensor, sr, format=fmt, encoding='PCM_S', bits_per_sample=16)


def _maybe_torch_compile(model: nn.Module, args) -> nn.Module:
    """
    Optionally compile model with torch.compile (PyTorch 2+). Returns (maybe) compiled model.
    Skips compile when DataParallel is used.
    """
    use_compile = getattr(args, "torch_compile", False)
    if not use_compile:
        return model

    if isinstance(model, nn.DataParallel):
        print("WARNING: DataParallel detected; skipping torch.compile for compatibility.")
        return model

    if not hasattr(torch, "compile"):
        print("WARNING: torch.compile is not available in this PyTorch version; skipping.")
        return model

    backend = getattr(args, "compile_backend", "inductor")
    mode = getattr(args, "compile_mode", "reduce-overhead")
    dynamic = getattr(args, "compile_dynamic", False)
    fullgraph = getattr(args, "compile_fullgraph", False)

    try:
        compiled = torch.compile(model, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        print(f"Model compiled with torch.compile backend={backend}, mode={mode}, dynamic={dynamic}.")
        return compiled
    except Exception as e:
        print(f"WARNING: torch.compile(dynamic/fullgraph) failed: {e}. Retrying without dynamic/fullgraph...")
        try:
            compiled = torch.compile(model, backend=backend, mode=mode)
            print(f"Model compiled with torch.compile backend={backend}, mode={mode}.")
            return compiled
        except Exception as e2:
            print(f"WARNING: torch.compile failed again: {e2}. Proceeding without compilation.")
            return model


def _setup_persistent_compile_cache():
    """
    Ensure persistent disk caches for Inductor and Triton to avoid recompilation across runs.
    """
    base = Path(current_dir) / ".torch_compile_cache"
    ind_dir = base / "inductor"
    tri_dir = base / "triton"
    try:
        ind_dir.mkdir(parents=True, exist_ok=True)
        tri_dir.mkdir(parents=True, exist_ok=True)
        # Only set if user didn't specify custom dirs
        if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ind_dir)
        if "TRITON_CACHE_DIR" not in os.environ:
            os.environ["TRITON_CACHE_DIR"] = str(tri_dir)
        print(f"[Cache] Inductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
        print(f"[Cache] Triton   cache: {os.environ['TRITON_CACHE_DIR']}")
    except Exception as e:
        print(f"[Cache] Failed to prepare persistent cache dirs: {e}")


def _try_warmup_or_fallback(model: nn.Module, config, device, args):
    """
    Run a tiny warmup forward to validate fullgraph. If it trips on
    data-dependent branching, auto-disable fullgraph and recompile.
    """
    if not getattr(args, "torch_compile", False):
        return model
    # Determine expected input shape [B,C,T]
    try:
        if args.model_type == 'htdemucs':
            chunk_size = config.training.samplerate * config.training.segment
        else:
            chunk_size = config.inference.chunk_size if 'chunk_size' in config.inference else config.audio.chunk_size
        channels = getattr(config.audio, 'num_channels', 2)
        bsz = 1
        dummy = torch.zeros((bsz, channels, chunk_size), dtype=torch.float32, device=device)
        with torch.inference_mode():
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                # Allow math fallback in warmup to avoid \"No available kernel\" errors
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                    model(dummy)
            else:
                model(dummy)
        print("[Compile] warmup forward ok.")
        return model
    except Exception as e:
        msg = str(e)
        if "Should not compile partial graph" in msg or "torch._dynamo.exc.Unsupported" in msg:
            if getattr(args, "compile_fullgraph", False):
                print("[Compile] fullgraph hit data-dependent branch; retrying without fullgraph...")
                setattr(args, "compile_fullgraph", False)
                model = _maybe_torch_compile(model, args)
                try:
                    with torch.inference_mode():
                        model(dummy)
                    print("[Compile] warmup forward ok (no fullgraph).")
                except Exception as e2:
                    print(f"[Compile] warmup still failed after disabling fullgraph: {e2}")
            else:
                print(f"[Compile] warmup failed: {e}")
        else:
            print(f"[Compile] warmup failed with unexpected error: {e}")
        return model
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

    for path in mixture_paths:
        print(f"Processing track: {path}")
        try:
            mix, sr = _load_audio_ta(path, sample_rate)  # [C, T]
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        # Handle mono/stereo as required by model
        if mix.ndim == 1:
            mix = np.expand_dims(mix, axis=0)  # -> [1, T]

        # Prefer model.stereo flag over config when available
        try:
            model_stereo = bool(getattr(model, "stereo", None))
        except Exception:
            model_stereo = None

        target_channels = None
        if model_stereo is not None:
            target_channels = 2 if model_stereo else 1
        elif 'num_channels' in config.audio:
            target_channels = int(config.audio['num_channels'])

        if target_channels is not None:
            ch = mix.shape[0]
            if target_channels == 2:
                if ch == 1:
                    print('Convert mono -> stereo (duplicate) to match model.stereo=True...')
                    mix = np.concatenate([mix, mix], axis=0)  # [2, T]
                elif ch > 2:
                    print(f'Truncate multi-channel ({ch}) -> stereo (first two channels) to match model.stereo=True...')
                    mix = mix[:2, :]  # keep first two channels
            elif target_channels == 1:
                if ch > 1:
                    print(f'Downmix {ch}-channel -> mono (mean) to match model.stereo=False...')
                    mix = mix.mean(axis=0, keepdims=True)  # [1, T]

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(path))[0]

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = args.pcm_type

            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=os.path.dirname(path),
                model_type=args.model_type,
                model=os.path.splitext(os.path.basename(args.start_check_point))[0]
            )

            output_dir = os.path.join(args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{fname}.{codec}")
            _save_audio_ta(output_path, estimates, sr, codec, subtype)
            print("Wrote file:", output_path)
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

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

    # CUDA matmul preferences (speed on Ampere/Hopper/RTX40) 
    try:
        if isinstance(device, str) and device.startswith('cuda') or (isinstance(device, torch.device) and device.type == 'cuda'):
            # Environment summary
            try:
                cuda_ver = getattr(torch.version, "cuda", "unknown")
                dev_idx = int(str(device).split(":")[1]) if isinstance(device, str) and ":" in device else 0
                name = torch.cuda.get_device_name(dev_idx)
                cap = ".".join(map(str, torch.cuda.get_device_capability(dev_idx)))
                print(f"[Env] torch={torch.__version__}, torchaudio={getattr(ta, '__version__', 'unknown')}, cuda={cuda_ver}")
                print(f"[CUDA] device='{name}', capability={cap}, cudnn.benchmark=True")
            except Exception:
                print(f"[Env] torch={torch.__version__}, torchaudio={getattr(ta, '__version__', 'unknown')}")

            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            # Report matmul preferences
            prec = None
            if hasattr(torch, 'get_float32_matmul_precision'):
                try:
                    prec = torch.get_float32_matmul_precision()
                except Exception:
                    prec = None
            print(f"[MatMul] TF32={torch.backends.cuda.matmul.allow_tf32}, float32_precision={prec if prec else 'default'}")
    except Exception:
        pass

    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))
    # I/O path info
    print("[IO] 使用 torchaudio: load/save/resample via torchaudio.functional.resample")

    # Config/shape summary for compile
    try:
        if args.model_type == 'htdemucs':
            mode = 'demucs'
            chunk_size = config.training.samplerate * config.training.segment
        else:
            mode = 'generic'
            chunk_size = config.inference.chunk_size if 'chunk_size' in config.inference else config.audio.chunk_size
        bs = config.inference.batch_size
        nover = config.inference.num_overlap
        step = chunk_size // nover
        print(f"[Shape] mode={mode}, chunk_size={chunk_size}, batch_size={bs}, num_overlap={nover}, step={step}")
    except Exception as e:
        print(f"[Shape] 形状参数汇总失败: {e}")

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    # Prepare persistent caches before compiling
    _setup_persistent_compile_cache()

    # Optional torch.compile for maximal inference speed (PyTorch 2+)
    print(f"[Compile] enabled={getattr(args, 'torch_compile', False)}, backend={getattr(args, 'compile_backend', None)}, mode={getattr(args, 'compile_mode', None)}, dynamic={getattr(args, 'compile_dynamic', False)}, fullgraph={getattr(args, 'compile_fullgraph', False)}")
    model = _maybe_torch_compile(model, args)
    # Validate fullgraph and auto-fallback if necessary
    model = _try_warmup_or_fallback(model, config, device, args)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
