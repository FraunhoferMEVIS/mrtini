# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate separate water/fat NPZ maps from the BrainWeb fuzzy tissue data.

Numerical conventions follow the MRtini generator supplied by the user.
Historical tissue constants: MRzero-Core, commit d7362c876bb39d7221354a2388d0e6df3c3ff959.
BrainWeb source: https://brainweb.bic.mni.mcgill.ca/brainweb/
"""
from pathlib import Path
import gzip
import json
import os
import tempfile
import zipfile
import numpy as np

_RAW_SHAPE = (362, 434, 362)
_FOV = (0.181, 0.217, 0.181)
_URL = 'https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1'
_CONFIG = Path(__file__).with_name('brainweb_presets.json')
_KEYS = {'PD': 'PD', 'T1': 'T1', 'T2': 'T2', "T2'": 'T2dash', 'D': 'D'}
_FAT = {'fat', 'mus', 'm-s', 'dura', 'fat2'}


def _raw_bytes(path, size):
    with gzip.open(path, 'rb') as f:
        raw = f.read(size + 1)
    if len(raw) != size:
        raise ValueError(f'{path}: expected {size} uncompressed bytes, got {len(raw)}')
    return raw


def _load_raw(subject, alias, cache_dir, shape, verbose):
    tag = f'subject{subject:02d}_{alias}'
    path = cache_dir / (tag + '.i8.gz')
    size = int(np.prod(shape))
    if path.exists():
        try:
            raw = _raw_bytes(path, size)
            if verbose: print(f'Using cached {tag}', flush=True)
            return np.frombuffer(raw, np.uint8).reshape(shape).swapaxes(0, 2)
        except (OSError, EOFError, ValueError):
            if verbose: print(f'Invalid cache {path}; downloading again', flush=True)
    import requests  # only needed when raw data is absent
    if verbose: print(f'Downloading {tag} ...', flush=True)
    fd, name = tempfile.mkstemp(dir=cache_dir, suffix='.part')
    os.close(fd)
    tmp = Path(name)
    try:
        with requests.post(_URL, data={'do_download_alias':tag, 'format_value':'raw_byte', 'zip_value':'gnuzip'}, stream=True, timeout=(20,180)) as response:
            response.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
        raw = _raw_bytes(tmp, size)  # validate gzip and shape before publishing cache
        os.replace(tmp, path)
    except Exception as exc:
        raise RuntimeError(f'BrainWeb download failed for {tag}: {exc}') from exc
    finally:
        tmp.unlink(missing_ok=True)
    return np.frombuffer(raw, np.uint8).reshape(shape).swapaxes(0, 2)


def _tissue_slab(raws, aliases, x0, x1, ny, nz, factor):
    """Equivalent to sum(load_tissue(...)) followed by the original downsample."""
    slab = None
    for raw, alias in zip(raws, aliases):
        b = raw[x0*factor:x1*factor, :ny*factor, :nz*factor] + np.uint8(128)
        if alias == 'ves': b[b == 1] = 0
        v = b.astype(np.float32) / 255.0
        if slab is None: slab = v
        else: slab += v
    out = np.zeros((x1-x0, ny, nz), np.float64)
    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                out += slab[i::factor, j::factor, k::factor]
    return out / factor**3


def _validate_header(path, shape, tissues):
    """Read NPY headers only; avoids decompressing huge existing maps on reuse."""
    required = [k+s for k in _KEYS.values() for s in ('_map','_fat')]
    required += ['tissue_'+t for t in tissues]
    with zipfile.ZipFile(path) as z:
        for key in required + ['FOV']:
            with z.open(key+'.npy') as f:
                version = np.lib.format.read_magic(f)
                if version == (1,0): actual, _, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2,0): actual, _, dtype = np.lib.format.read_array_header_2_0(f)
                else: raise ValueError(f'Unsupported NPY version {version}')
            if actual != ((3,) if key == 'FOV' else shape) or dtype.kind != 'f':
                raise ValueError(f'Unexpected map shape or dtype: {key}: {actual}, {dtype}')
    # Check FOV and (when relevant) both species at generation time in _generate.


def _generate(output, subject, cfg, table, cache, shape, verbose, slab_depth):
    factor = cfg['downsample']
    if cfg['noise'] != 0: raise ValueError('Noisy configurations are not implemented; no silent substitution.')
    if type(factor) is not int or factor < 1: raise ValueError('Invalid downsample factor')
    res = tuple(n//factor for n in (shape[2],shape[1],shape[0]))
    nx,ny,nz = res
    tissue_names = cfg['tissues']
    with tempfile.TemporaryDirectory(prefix='brainweb-build-', dir=output.parent) as td:
        td = Path(td)
        files = {}
        def alloc(key, dtype, dims=res):
            path = td / (key+'.npy'); files[key] = path
            return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=dims)
        fov = alloc('FOV', np.float64, (3,)); fov[:] = _FOV; del fov
        maps = {k+s:alloc(k+s,np.float32) for k in _KEYS.values() for s in ('_map','_fat')}
        for arr in maps.values(): arr[:] = 0
        del arr
        for name in tissue_names:
            aliases = table['download-aliases'][name]
            raws = [_load_raw(subject,a,cache,shape,verbose) for a in aliases]
            seg = alloc('tissue_'+name,np.float64)
            props = table['tissues'][cfg['field-strength']][name]
            suffix = '_fat' if name in _FAT else '_map'
            if verbose: print(f'Building {name} → {suffix}', flush=True)
            for x0 in range(0,nx,slab_depth):
                x1 = min(nx,x0+slab_depth)
                v = _tissue_slab(raws,aliases,x0,x1,ny,nz,factor)
                seg[x0:x1] = v
                for prop,key in _KEYS.items():
                    maps[key+suffix][x0:x1] += props[prop]*v
            seg.flush(); del seg, raws
        for key, arr in maps.items():
            if key.startswith('PD') and (key.endswith('_map') or any(t in _FAT for t in tissue_names)) and not np.any(arr>0):
                raise ValueError(f'Generated {key} is empty; check BrainWeb data')
            arr.flush()
        del arr; maps.clear()
        if verbose: print(f'Compressing {output.name} ...',flush=True)
        archive = td/'phantom.npz'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for key,path in files.items(): z.write(path,key+'.npy')
        _validate_header(archive,res,tissue_names)
        os.replace(archive,output)


def download_brainwebphantom(subject=4, config='3T-highres-fat', output_dir='data', *, cache_dir=None, force=False, verbose=True, config_path=None):
    """Return Path to a single generated phantom; reuse existing output by default.

    subject: 4 or '04'. No other subjects are downloaded.
    config: '3T-highres-fat' (default) or '3T' (downsampled water only).
    output_dir: relative to current working directory unless absolute.
    cache_dir: defaults to output_dir/brainweb_cache.
    config_path: optional explicit historical/custom configuration JSON (noise must be zero).
    force: regenerate NPZ from cached raw tissue maps, without redownloading valid raw files.
    """
    if isinstance(subject,bool) or not isinstance(subject,(int,str)) or not str(subject).isdigit():
        raise ValueError('subject must be a BrainWeb integer ID, e.g. 4 or "04"')
    subject = int(subject)
    table = json.loads(Path(config_path or _CONFIG).read_text(encoding='utf-8'))
    if subject not in table['subjects']: raise ValueError(f'Unknown BrainWeb subject {subject}; choose from {table["subjects"]}')
    if config not in table['configs'] or not config or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in config):
        raise ValueError(f'Unknown/invalid configuration: {config!r}')
    cfg = table['configs'][config]
    if cfg['noise'] != 0: raise ValueError('Noisy configurations are not supported by this deterministic generator')
    factor = cfg['downsample']
    if type(factor) is not int or factor < 1: raise ValueError('Invalid downsample factor')
    root = Path(output_dir).expanduser(); root.mkdir(parents=True,exist_ok=True)
    output = root / f'subject{subject:02d}_{config}.npz'
    res = tuple(n//factor for n in (_RAW_SHAPE[2],_RAW_SHAPE[1],_RAW_SHAPE[0]))
    if output.exists() and not force:
        _validate_header(output,res,cfg['tissues'])
        if verbose: print(f'Using existing phantom: {output}',flush=True)
        return output
    cache = Path(cache_dir).expanduser() if cache_dir is not None else root/'brainweb_cache'
    cache.mkdir(parents=True,exist_ok=True)
    _generate(output,subject,cfg,table,cache,_RAW_SHAPE,verbose,8)
    if verbose: print(f'Phantom ready: {output}',flush=True)
    return output


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',default='04'); p.add_argument('--config',default='3T-highres-fat')
    p.add_argument('--output-dir',default='data'); p.add_argument('--cache-dir'); p.add_argument('--config-path')
    p.add_argument('--force',action='store_true')
    a=p.parse_args(); download_brainwebphantom(a.subject,a.config,a.output_dir,cache_dir=a.cache_dir,force=a.force,config_path=a.config_path)

if __name__=='__main__': main()
