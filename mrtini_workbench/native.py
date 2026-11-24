"""Local MRtini bridge, following the user's sim_run.py import and physics path."""
import importlib.util, time, os, sys, hashlib, copy
from pathlib import Path
from dataclasses import asdict
import numpy as np

PRESETS={'flash':'FLASH_sequence_raw.json','rare':'RARE_sequence_raw.json','epi':'SE_EPI_sequence_raw.json'}
class PhantomCache:
    """Single-entry cache of pristine built CPU phantom data. No simulation state."""
    def __init__(self):
        self.key=None; self.data=None
    @staticmethod
    def identity(path,resolution,slice_index,thickness):
        stat=path.stat()
        return (str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,
                int(resolution),31,int(slice_index),float(thickness))
    def get(self,key,builder):
        hit=self.data is not None and key==self.key
        if not hit:
            # Discard the old entry before loading a potentially large new phantom.
            self.key=None; self.data=None
            candidate=builder()
            self.data=candidate; self.key=key
        # .cuda(), graph construction and RF baking are allowed to mutate their input.
        # Never expose the cached CPU master to these operations.
        try:
            return copy.deepcopy(self.data),hit
        except (TypeError, RuntimeError, copy.Error, AttributeError):
            # An installed phantom class may not support deepcopy. In that case
            # retain correctness by rebuilding rather than sharing mutable data.
            candidate=self.data if not hit else None
            self.key=None; self.data=None
            return (candidate if candidate is not None else builder()),False

class NativeRunner:
    def __init__(self,root=None,adapter=None):
        self.root=Path(root or Path(__file__).resolve().parent.parent).resolve()
        self.adapter_path=Path(adapter).resolve() if adapter else None
        self.phantoms=PhantomCache();self.adapter=None;self.cached_key=None;self.engine=None;self.seq=None;self.k=None;self.meta={};self.design=None
    @property
    def configured(self):return (self.root/'src/mrtini/mrtini_main.py').is_file() and (self.root/'src/sequence/sequence.py').is_file()
    @property
    def parameterized(self):return True
    def catalog(self):
        seqdir=self.root/'src/sequence/gstar_seqs'; datadir=self.root/'data'
        return {'sequences':[x.name for x in sorted(seqdir.glob('*.json'))], 'phantoms':[x.name for x in sorted(datadir.glob('*.npz'))], 'presets':{k:v for k,v in PRESETS.items() if (seqdir/v).is_file()},'parameterized':self.parameterized}
    def path(self,folder,name):
        base=(self.root/folder).resolve(); path=(base/name).resolve()
        if path.parent!=base or not path.is_file():raise ValueError(f'File not found in {folder}: {name}')
        return path
    def load(self):
        if self.adapter is None:
            spec=importlib.util.spec_from_file_location('mrtini_workbench_recipe',self.adapter_path);self.adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.adapter)
        return self.adapter
    def run(self,p,workdir,progress,metrics):
        if not self.configured:raise RuntimeError('MRtini source not found. Place the workbench in MRtini_release or pass --mrtini-root.')
        if str(self.root) not in sys.path:sys.path.insert(0,str(self.root))
        # The native importer may resolve paths relative to the repository, as sim_run.py does.
        import torch
        from src.mrtini.mrtini_main import MRtiniSim
        from src.voxelgridphantom import voxel_grid_phantom
        from src.mrtini.config import SLICE_THICKNESS
        from src.sequence.sequence import Sequence
        from src.recon.nufft_recon import _reconstruct
        from src.mrtini.optimized import compute_graph_fast, prepare_gpu_execution
        if not torch.cuda.is_available():raise RuntimeError('Native mode requires NVIDIA CUDA in the current Python environment.')
        cfg=asdict(p); seqfile=None if p.native_edit and not self.adapter_path else self.path('src/sequence/gstar_seqs',p.sequence_file or PRESETS[p.sequence]); phantomfile=self.path('data',p.phantom_file or 'subject04_3T-highres-fat_custom_dense.npz')
        if p.native_edit and not self.parameterized:raise ValueError('No native parameter recipe configured.')
        if not p.native_edit:
            # No nominal slider values enter the native imported sequence path.
            key=(p.sequence,p.resolution,p.slice_index,p.workspace_mb,p.kernel_mode,str(seqfile),hashlib.sha256(seqfile.read_bytes()).hexdigest(),str(phantomfile),phantomfile.stat().st_mtime_ns,phantomfile.stat().st_size,p.max_states,p.min_state_mag)
        else:
            key=(tuple((k,v) for k,v in cfg.items() if k not in ('backend','recon','cuda_graph')),hashlib.sha256(seqfile.read_bytes()).hexdigest() if seqfile else None,str(phantomfile),phantomfile.stat().st_mtime_ns,phantomfile.stat().st_size,self.adapter_path.stat().st_mtime_ns if self.adapter_path else None)
        thickness=p.slice_mm*1e-3 if p.native_edit and not self.adapter_path else float(SLICE_THICKNESS)
        phantom_key=self.phantoms.identity(phantomfile,p.resolution,p.slice_index,thickness)
        key=(key,phantom_key)
        def sync():torch.cuda.synchronize()
        def timed(stage,fn):
            progress(stage,None,stage.replace('_',' ').capitalize());sync();t=time.perf_counter();val=fn();sync();metrics[stage]=time.perf_counter()-t;return val
        def cpu(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
        with torch.no_grad():
            reuse=self.engine is not None and key==self.cached_key
            metrics['prepared_reused']=reuse
            metrics['phantom_reused']=reuse
            if reuse:metrics['phantom']=0.0
            if not reuse:
                self.engine=None;self.cached_key=None;self.seq=None;self.k=None;self.meta={};self.design=None
                design_info={}
                def import_seq():
                    if p.native_edit and not self.adapter_path:
                        from native_sequences import build_native_sequence
                        seq,design=build_native_sequence(p,return_design=True)
                        self.design=design
                        design_info.update(actual_rf_ms=design.actual_rf_ms,encoding_fov_mm=p.fov_mm,expected_adc_samples=len(design.adc_times),generator='finite-RF native held-gradient research sequence')
                        return seq
                    path=seqfile
                    if p.native_edit:
                        path=Path(self.load().generate_sequence(cfg,seqfile,Path(workdir)))
                        if not path.is_file():raise ValueError('Sequence recipe did not return an existing JSON file.')
                    return Sequence.import_file_for_mrtini_sim(str(path))
                seq=timed('sequence',import_seq)
                def load_phantom():
                    t=time.perf_counter();obj=voxel_grid_phantom.VoxelGridPhantom.brainweb(str(phantomfile));sync();metrics['phantom_load_seconds']=time.perf_counter()-t
                    t=time.perf_counter();obj=obj.interpolate(p.resolution,p.resolution,31).slices([p.slice_index]);obj.size[-1]=torch.tensor([thickness]);sync();metrics['phantom_interpolate_seconds']=time.perf_counter()-t
                    t=time.perf_counter();data=obj.build();sync();metrics['phantom_build_seconds']=time.perf_counter()-t
                    return data
                data,hit=timed('phantom',lambda:self.phantoms.get(phantom_key,load_phantom))
                metrics['phantom_reused']=hit
                art=timed('rf_bake',lambda:MRtiniSim(device='cuda',seq=seq).bake_bloch_kernels(phantom_data=data,z_span_m=thickness,global_RF=False))
                seq=seq.cuda();data=data.cuda()
                graph,keys=timed('graph',lambda:compute_graph_fast(seq,data,art,max_state_count=p.max_states,min_state_mag=p.min_state_mag,slice_thickness=thickness))
                self.engine=timed('prepare',lambda:prepare_gpu_execution(graph,seq,data,keys,art,slice_thickness=thickness,signal_workspace_mb=p.workspace_mb,kernel_mode=p.kernel_mode))
                self.seq=seq; self.k=seq.get_kspace()
                if self.design is not None:
                    xy=cpu(self.k); target=self.design.expected_k
                    if xy.ndim==2 and xy.shape[0]!=len(target) and xy.shape[1]==len(target):xy=xy.T
                    if xy.ndim!=2 or xy.shape[0]!=len(target) or xy.shape[1]<2 or not np.allclose(xy[:,:2],target,rtol=0,atol=.02):
                        self.engine=None
                        raise ValueError('Native get_kspace does not match the generated physical xy trajectory. Run check_native_sequences.py to inspect the installed convention.')
                    design_info['native_trajectory_max_abs_error_cycles_per_m']=float(np.max(np.abs(xy[:,:2]-target)))
                self.cached_key=key
                self.meta={'repetitions':len(seq),'active_voxels':int(data.PD.numel()),'sequence_file':seqfile.name if seqfile else 'Built-in '+p.sequence.upper(),'phantom_file':phantomfile.name,'slice_thickness_m':thickness,'slice_index':p.slice_index,'sequence_recipe':bool(p.native_edit),'preset_parameters_read_only':not p.native_edit,'max_states':p.max_states,'min_state_mag':p.min_state_mag,'gpu':torch.cuda.get_device_name(data.PD.device),**design_info}
            if self.design is not None:np.savez_compressed(Path(workdir)/'sequence_design.npz',expected_k_xy=self.design.expected_k,adc_times_s=self.design.adc_times)
            torch.cuda.reset_peak_memory_stats();perf={}
            if p.cuda_graph and self.engine.capture_graph is None:timed('capture',self.engine.capture)
            sig=timed('execute',lambda:self.engine.run(cuda_graph=p.cuda_graph,publish=False,performance=perf,print_progress=True))
            metrics.update(gpu_seconds=perf.get('gpu_execution_seconds'),peak_mib=torch.cuda.max_memory_allocated()/1048576,prepared_calls=perf.get('prepared_calls'),cuda_graph=bool(p.cuda_graph),kernel_mode=p.kernel_mode,compile_cache='not instrumented',active_voxels=self.meta['active_voxels'],repetitions=self.meta['repetitions'])
            reco=timed('reconstruct',lambda:_reconstruct(sig,self.k,im_size=[p.resolution,p.resolution]))
            arr=cpu(sig); arr=arr[:,None] if arr.ndim==1 else arr
            if arr.ndim!=2:raise ValueError(f'Unexpected native signal shape {arr.shape}')
            image=cpu(reco);image=np.squeeze(image)
            if image.ndim==2:mag=np.abs(image);phase=np.angle(image)
            elif image.ndim==3 and image.shape[-2:]==(p.resolution,p.resolution):mag=np.sqrt(np.sum(np.abs(image)**2,axis=0));phase=np.angle(image[0])
            else:raise ValueError(f'Unexpected native reconstruction shape {image.shape}')
            # Preserve native k-space coordinates untouched for export, no assumed unit/axis conversion.
            k=cpu(self.k)
            if k.ndim!=2:raise ValueError(f'Unexpected native k-space shape {k.shape}')
            if k.shape[0]!=len(arr) and k.shape[1]==len(arr):k=k.T
            if k.shape[0]!=len(arr):raise ValueError(f'ADC/k-space length mismatch: {arr.shape}, {k.shape}')
            # ADC-order view, not gridded k-space; row length is display-only.
            width=p.resolution; env=np.linalg.norm(arr,axis=1); padded=np.pad(env,(0,(-len(env))%width)); kview=np.log1p(padded.reshape(-1,width))
            if self.meta.get('expected_adc_samples',len(arr))!=len(arr):raise ValueError('Generated sequence ADC count differs from native execution.')
            if not np.isfinite(arr).all() or not np.isfinite(image).all():raise ValueError('Non-finite native signal or reconstruction.')
            return arr,k,dict(self.meta),dict(magnitude=mag,phase=phase,kspace=kview,complex_image=image,method='MRtini native _reconstruct / adjoint',design_k=self.design.expected_k if self.design is not None else None,adc_times=self.design.adc_times if self.design is not None else None)
