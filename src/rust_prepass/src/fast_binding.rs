// SPDX-License-Identifier: AGPL-3.0-or-later
use pyo3::prelude::*;
use pyo3::types::{PyAny,PyDict,PyList,PyBytes};
use crate::{read_cplx64_1d,read_i32_1d_any,tensor_numel,to_cpu_contig};
use crate::pre_pass::Repetition;
use crate::pre_pass::{RotTaps1D,Support};
use crate::pre_pass_fast::{self,Case,Graph};
use std::time::Instant;

// Unlike the legacy parser, validate lengths and explicitly convert event
// arrays to their native dtypes before exposing their contiguous buffers.
fn parse_sequence(seq:&PyList)->PyResult<Vec<Repetition>>{
    let mut result=Vec::with_capacity(seq.len());
    for rep in seq.iter(){
        let pulse=rep.getattr("pulse")?;let count:usize=rep.getattr("event_count")?.extract()?;
        if count==0{return Err(pyo3::exceptions::PyValueError::new_err("Empty repetition"));}
        let dt=to_cpu_contig(rep.getattr("event_time")?.call_method0("float")?)?;
        let grad=to_cpu_contig(rep.getattr("gradm")?.call_method0("float")?)?;
        let adc=to_cpu_contig(rep.getattr("adc_usage")?)?;
        if tensor_numel(dt)?!=count||tensor_numel(grad)?!=3*count||tensor_numel(adc)?!=count{return Err(pyo3::exceptions::PyValueError::new_err("Event-array length mismatch"));}
        let times=unsafe{std::slice::from_raw_parts(dt.call_method0("data_ptr")?.extract::<usize>()? as *const f32,count)}.to_vec();
        let gradients=unsafe{std::slice::from_raw_parts(grad.call_method0("data_ptr")?.extract::<usize>()? as *const [f32;3],count)}.to_vec();
        if times.iter().any(|x|!x.is_finite()||*x<0.)||gradients.iter().flatten().any(|x|!x.is_finite()){return Err(pyo3::exceptions::PyValueError::new_err("Invalid event data"));}
        result.push(Repetition{pulse_angle:pulse.getattr("angle")?.extract()?,pulse_phase:pulse.getattr("phase")?.extract()?,event_time:times,gradm_event:gradients,adc_mask:read_i32_1d_any(adc,count)?.iter().map(|x|*x>0).collect()});
    }Ok(result)
}

fn read_case(taps:&PyList,supports:&PyList,tissue:(f32,f32,f32,f32))->PyResult<Case>{
    if taps.len()!=supports.len(){return Err(pyo3::exceptions::PyValueError::new_err("Taps/support bank length mismatch"));}
    let mut tv=Vec::new();let mut sv=Vec::new();
    for i in 0..taps.len(){let t=taps.get_item(i)?;let n=tensor_numel(to_cpu_contig(t.getattr("pp")?)?)?;
        for name in ["pp","pz","mp","mz","zp","zz","deltas"]{if tensor_numel(to_cpu_contig(t.getattr(name)?)?)?!=n{return Err(pyo3::exceptions::PyValueError::new_err("Coefficient length mismatch"));}}
        let dk:f32=t.getattr("dk")?.extract().or_else(|_|t.getattr("dk")?.call_method0("item")?.extract())?;
        tv.push(RotTaps1D{pp:read_cplx64_1d(t.getattr("pp")?,n)?,pz:read_cplx64_1d(t.getattr("pz")?,n)?,mp:read_cplx64_1d(t.getattr("mp")?,n)?,mz:read_cplx64_1d(t.getattr("mz")?,n)?,zp:read_cplx64_1d(t.getattr("zp")?,n)?,zz:read_cplx64_1d(t.getattr("zz")?,n)?,deltas:read_i32_1d_any(t.getattr("deltas")?,n)?,dk});
        let s:&PyDict=supports.get_item(i)?.downcast()?;
        let read=|name:&str|->PyResult<Vec<i32>>{let a=s.get_item(name)?.ok_or_else(||pyo3::exceptions::PyKeyError::new_err(name.to_string()))?;let v=read_i32_1d_any(a,tensor_numel(to_cpu_contig(a)?)?)?;if v.iter().any(|x|*x<0||*x as usize>=n){return Err(pyo3::exceptions::PyValueError::new_err("Invalid support index"));}if v.windows(2).any(|a|a[0]>=a[1]){return Err(pyo3::exceptions::PyValueError::new_err("Support indices must be sorted and unique"));}Ok(v)};
        sv.push(Support{pp:read("++")?,pz:read("+z")?,mp:read("-+")?,mz:read("-z")?,zp:read("z+")?,zz:read("zz")?});
    }
    Ok(Case{tissue:[tissue.0,tissue.1,tissue.2,tissue.3*1e-9],taps:tv,support:sv})
}
fn bytes<'p,T:Copy>(py:Python<'p>,v:&[T])->&'p PyBytes{
    assert!(cfg!(target_endian="little"));
    let raw=unsafe{std::slice::from_raw_parts(v.as_ptr() as *const u8,std::mem::size_of_val(v))};PyBytes::new(py,raw)
}
fn export<'p,const S:usize>(py:Python<'p>,g:Graph<S>,elapsed:f64)->PyResult<&'p PyDict>{
    let d=PyDict::new(py);let started=Instant::now();
    d.set_item("version",1)?;d.set_item("cases",S)?;d.set_item("nodes",g.nodes.len())?;d.set_item("edges",g.edges.len())?;
    d.set_item("candidates",g.candidates)?;d.set_item("candidate_edges",g.candidate_edges)?;d.set_item("build_seconds",elapsed)?;
    d.set_item("threads",rayon::current_num_threads())?;
    let layers:Vec<u64>=g.layers.iter().map(|x|*x as u64).collect();d.set_item("layers",bytes(py,&layers))?;
    let kind:Vec<u8>=g.nodes.iter().map(|n|n.kind).collect();d.set_item("kind",bytes(py,&kind))?;
    let masks:Vec<u8>=g.nodes.iter().map(|n|n.mask).collect();d.set_item("node_mask",bytes(py,&masks))?;
    let kt:Vec<f64>=g.nodes.iter().flat_map(|n|n.kt_start).collect();d.set_item("kt_start",bytes(py,&kt))?;
    let kt:Vec<f64>=g.nodes.iter().flat_map(|n|n.kt).collect();d.set_item("kt_end",bytes(py,&kt))?;
    for (key,field) in [("emitted",0),("latent",1),("signal",2)]{let a:Vec<f32>=g.nodes.iter().flat_map(|n|match field{0=>n.emitted,1=>n.latent,_=>n.signal}).collect();d.set_item(key,bytes(py,&a))?;}
    let nominal:Vec<f32>=g.nodes.iter().flat_map(|n|n.mag.iter().flat_map(|m|[m.re,m.im])).collect();d.set_item("nominal",bytes(py,&nominal))?;
    let mut rows:Vec<u64>=g.nodes.iter().map(|n|n.edge_start as u64).collect();rows.push(g.edges.len() as u64);d.set_item("row_ptr",bytes(py,&rows))?;
    let a:Vec<u32>=g.edges.iter().map(|e|e.parent as u32).collect();d.set_item("parent",bytes(py,&a))?;
    let a:Vec<u8>=g.edges.iter().map(|e|e.relation).collect();d.set_item("relation",bytes(py,&a))?;
    let a:Vec<i32>=g.edges.iter().map(|e|e.tap).collect();d.set_item("tap",bytes(py,&a))?;
    let a:Vec<u8>=g.edges.iter().map(|e|e.mask).collect();d.set_item("edge_mask",bytes(py,&a))?;
    d.set_item("export_seconds",started.elapsed().as_secs_f64())?;Ok(d)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_graph_packed<'p>(py:Python<'p>,seq:&PyList,pulse_id:&PyAny,tissues:Vec<(f32,f32,f32,f32)>,tap_cases:&PyList,support_cases:&PyList,max_state_count:usize,min_state_mag:f32,nyquist:[f32;3],size:[f32;3],normalized_grads:bool,slice_thickness:f32)->PyResult<&'p PyDict>{
    if tissues.len()<1||tissues.len()>2||tissues.len()!=tap_cases.len()||tissues.len()!=support_cases.len(){return Err(pyo3::exceptions::PyValueError::new_err("Expected one or two aligned species cases"));}
    if max_state_count==0{return Err(pyo3::exceptions::PyValueError::new_err("State budget must be positive"));}
    let sequence=parse_sequence(seq)?;if tensor_numel(to_cpu_contig(pulse_id)?)?!=sequence.len(){return Err(pyo3::exceptions::PyValueError::new_err("pulse_id length mismatch"));}let ids=read_i32_1d_any(pulse_id,sequence.len())?;let ids:Vec<usize>=ids.iter().map(|x|*x as usize).collect();
    if !min_state_mag.is_finite()||min_state_mag<0.||!slice_thickness.is_finite()||slice_thickness<=0.{return Err(pyo3::exceptions::PyValueError::new_err("Invalid cutoff or slice thickness"));}
    let mut cases=Vec::new();for s in 0..tissues.len(){cases.push(read_case(tap_cases.get_item(s)?.downcast()?,support_cases.get_item(s)?.downcast()?,tissues[s])?);}
    for c in &cases{if ids.iter().any(|i|*i>=c.taps.len()){return Err(pyo3::exceptions::PyValueError::new_err("Invalid pulse id"));}}
    if cases.len()==2{if cases[0].taps.len()!=cases[1].taps.len(){return Err(pyo3::exceptions::PyValueError::new_err("Species pulse banks must align"));}for (a,b) in cases[0].taps.iter().zip(&cases[1].taps){if a.deltas!=b.deltas||a.dk!=b.dk{return Err(pyo3::exceptions::PyValueError::new_err("Species tap grids must match exactly"));}}}
    let scale=std::array::from_fn(|a|if normalized_grads&&size[a]!=0.{1./size[a]}else{1.});
    let started=Instant::now();
    if cases.len()==1{let g=py.allow_threads(||pre_pass_fast::build::<1>(&sequence,&ids,&cases,max_state_count,min_state_mag,nyquist,scale,slice_thickness));export(py,g,started.elapsed().as_secs_f64())}
    else{let g=py.allow_threads(||pre_pass_fast::build::<2>(&sequence,&ids,&cases,max_state_count,min_state_mag,nyquist,scale,slice_thickness));export(py,g,started.elapsed().as_secs_f64())}
}

// Fused CPU CSR RF application. A row owns its output, so no atomics or
// temporary E*V tensor is required. Inputs must not be mutated concurrently.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn apply_rf_cpu<'p>(py:Python<'p>,parent:&'p PyAny,table:&PyAny,phase:&PyAny,ids:&PyAny,rows:&PyAny,src:&PyAny,slot:&PyAny,rel:&PyAny,mask:&PyAny)->PyResult<&'p PyAny>{
    use rayon::prelude::*;
    use num_complex::Complex32;
    fn check(t:&PyAny,dtype:&str)->PyResult<(usize,usize)>{
        let cpu:String=t.getattr("device")?.getattr("type")?.extract()?;
        let dt:String=t.getattr("dtype")?.repr()?.extract()?;
        if cpu!="cpu"||!dt.ends_with(dtype)||!t.call_method0("is_contiguous")?.extract::<bool>()?{return Err(pyo3::exceptions::PyValueError::new_err(format!("Expected contiguous CPU {dtype}")));}
        Ok((t.call_method0("data_ptr")?.extract()?,tensor_numel(t)?))
    }
    fn ints(t:&PyAny)->PyResult<&[i64]>{let (p,n)=check(t,"int64")?;Ok(if n==0{&[]}else{unsafe{std::slice::from_raw_parts(p as *const i64,n)}})}
    let (p,_) = check(parent,"complex64")?;let (c,_) = check(table,"complex64")?;let (ph,nph)=check(phase,"complex64")?;
    let (parents,v):(usize,usize)=parent.getattr("shape")?.extract()?;let (coeffs,cv):(usize,usize)=table.getattr("shape")?.extract()?;
    let row=ints(rows)?;let source=ints(src)?;let slots=ints(slot)?;let relations=ints(rel)?;let masks=ints(mask)?;let cases=ints(ids)?;
    let e=source.len();
    if row.is_empty()||cv!=v||nph!=v||cases.len()!=v||slots.len()!=e||relations.len()!=e||masks.len()!=e||row[0]!=0||row[row.len()-1]!=e as i64||row.windows(2).any(|x|x[0]>x[1])||source.iter().any(|x|*x<0||*x as usize>=parents)||slots.iter().any(|x|*x<0||*x as usize>=coeffs)||relations.iter().any(|x|*x<0||*x>5)||cases.iter().any(|x|*x<0||*x>1){return Err(pyo3::exceptions::PyValueError::new_err("Invalid RF CSR arrays or dimensions"));}
    let children=row.len()-1;let out=parent.call_method1("new_zeros",((children,v),))?;
    if v==0||children==0{return Ok(out)}
    let parent_data=if parents==0{&[]}else{unsafe{std::slice::from_raw_parts(p as *const Complex32,parents*v)}};
    let coefficients=if coeffs==0{&[]}else{unsafe{std::slice::from_raw_parts(c as *const Complex32,coeffs*v)}};
    let phases=unsafe{std::slice::from_raw_parts(ph as *const Complex32,v)};
    let op:usize=out.call_method0("data_ptr")?.extract()?;
    let output=unsafe{std::slice::from_raw_parts_mut(op as *mut Complex32,children*v)};
    py.allow_threads(||output.par_chunks_mut(v).enumerate().for_each(|(d,y)|{
        for e in row[d] as usize..row[d+1] as usize{
            let r=relations[e];let bits=masks[e];let x=&parent_data[source[e] as usize*v..][..v];let c=&coefficients[slots[e] as usize*v..][..v];
            // Relation branches are outside the inner voxel loop.
            match r{
                0|3=>for i in 0..v{if (bits>>cases[i])&1!=0{y[i]+=x[i]*c[i];}},
                1=>for i in 0..v{if (bits>>cases[i])&1!=0{y[i]+=x[i]*(c[i]*phases[i].conj());}},
                2=>for i in 0..v{if (bits>>cases[i])&1!=0{y[i]+=x[i]*(c[i]*phases[i]);}},
                4=>for i in 0..v{if (bits>>cases[i])&1!=0{y[i]+=x[i].conj()*(c[i]*(phases[i]*phases[i]));}},
                _=>for i in 0..v{if (bits>>cases[i])&1!=0{y[i]+=x[i].conj()*(c[i]*phases[i]);}},
            }
        }
    }));Ok(out)
}
