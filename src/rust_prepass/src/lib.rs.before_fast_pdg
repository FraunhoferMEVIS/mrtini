//# SPDX-FileCopyrightText: 2023 MRzero
// SPDX-FileCopyrightText: 2026 Fraunhofer Institute for Digital Medicine MEVIS
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of MRTini.
//
// Portions of this file are derived from MRzero-Core:
// https://github.com/MRsources/MRzero-Core
//
// This file contains modifications and extensions made for MRTini.
// The original MRzero-Core copyright and attribution notices are retained.
//
// MRTini is research software and is not qualified for use as a medical
// device or as part of a medical device.

use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{wrap_pyfunction, PyTraverseError};
use pyo3::IntoPy;

mod pre_pass;
use pre_pass::{comp_graph, Repetition, RotTaps1D, Support};

use num_complex::Complex32;
use std::{collections::HashMap, slice::from_raw_parts, time::Instant};

#[pyclass(module = "_prepass")]
struct PyDistribution {
    #[pyo3(get, set)]
    dist_type: String,
    #[pyo3(get, set)]
    ancestors: Option<Py<PyList>>,
    #[pyo3(get, set)]
    mag: Option<PyObject>,
    #[pyo3(get, set)]
    kt_vec: Option<PyObject>,

    #[pyo3(get)]
    prepass_mag: Option<Py<PyComplex>>,
    #[pyo3(get)]
    latent_signal: f32,
    #[pyo3(get)]
    latent_signal_unormalized: f32,
    #[pyo3(get)]
    signal: f32,
    #[pyo3(get)]
    emitted_signal: f32,
    #[pyo3(get)]
    prepass_kt_vec: [f64; 4],
}

#[pymethods]
impl PyDistribution {
    #[new]
    fn new(py: Python) -> Self {
        PyDistribution {
            dist_type: "?".to_owned(),
            ancestors: Some(PyList::empty(py).into()),
            mag: Some(py.None()),
            kt_vec: Some(py.None()),
            prepass_mag: Some(PyComplex::from_doubles(py, 0.0, 0.0).into()),
            latent_signal: 0.0,
            latent_signal_unormalized: 0.0,
            signal: 0.0,
            emitted_signal: 0.0,
            prepass_kt_vec: [0.0, 0.0, 0.0, 0.0],
        }
    }

    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        if let Some(ancestors) = &self.ancestors { visit.call(ancestors)?; }
        if let Some(mag) = &self.mag { visit.call(mag)?; }
        if let Some(kt_offset_vec) = &self.kt_vec { visit.call(kt_offset_vec)?; }
        if let Some(prepass_mag) = &self.prepass_mag { visit.call(prepass_mag)?; }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.ancestors = None;
        self.mag = None;
        self.kt_vec = None;
        self.prepass_mag = None;
    }

    fn __repr__(&self) -> String {
        Python::with_gil(|py| -> String {
            format!(
                "Dist(type: {}, mag: {:?}, signal: {}, kt: {:?}, #ancestors: {})",
                self.dist_type,
                self.prepass_mag,
                self.signal,
                self.prepass_kt_vec,
                match &self.ancestors {
                    Some(a) => a.as_ref(py).len(),
                    None => 0,
                },
            )
        })
    }
}

// ---- helpers ----
fn to_cpu_contig<'py>(t: &'py PyAny) -> PyResult<&'py PyAny> {
    Ok(t.getattr("contiguous")?.call0()?.getattr("cpu")?.call0()?)
}
fn tensor_numel(t: &PyAny) -> PyResult<usize> { t.getattr("numel")?.call0()?.extract() }
fn tensor_data_ptr(t: &PyAny) -> PyResult<usize> { t.getattr("data_ptr")?.call0()?.extract() }
fn tensor_dtype_str(t: &PyAny) -> PyResult<String> { t.getattr("dtype")?.repr()?.extract() }

fn read_cplx64_1d(t: &PyAny, n: usize) -> PyResult<Vec<Complex32>> {
    let tt = to_cpu_contig(t)?;
    let dtype = tensor_dtype_str(tt)?;
    if !dtype.contains("complex64") {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected torch.complex64, got {dtype}"
        )));
    }
    let ptr = tensor_data_ptr(tt)?;
    let raw = unsafe { from_raw_parts(ptr as *const f32, 2 * n) };
    let mut out = Vec::with_capacity(n);
    for i in 0..n { out.push(Complex32::new(raw[2*i], raw[2*i+1])); }
    Ok(out)
}

fn read_i32_1d_any(t: &PyAny, n: usize) -> PyResult<Vec<i32>> {
    let tt = to_cpu_contig(t)?;
    let dtype = tensor_dtype_str(tt)?;
    let ptr = tensor_data_ptr(tt)?;
    if dtype.contains("int32") {
        Ok(unsafe { from_raw_parts(ptr as *const i32, n).to_vec() })
    } else if dtype.contains("int64") {
        let raw = unsafe { from_raw_parts(ptr as *const i64, n) };
        Ok(raw.iter().map(|&x| x as i32).collect())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected torch.int32 or torch.int64, got {dtype}"
        )))
    }
}

fn parse_sequence(seq: &PyList) -> PyResult<Vec<Repetition>> {
    let mut sequence = Vec::with_capacity(seq.len());
    for rep in seq.iter() {
        let pulse = rep.getattr("pulse")?;
        let pulse_angle: f32 = pulse.getattr("angle")?.extract()?;
        let pulse_phase: f32 = pulse.getattr("phase")?.extract()?;

        let event_count: usize = rep.getattr("event_count")?.extract()?;
        let data_ptr = |name: &str| -> PyResult<usize> {
            let tensor = rep.getattr(name)?.getattr("cpu")?.call0()?;
            tensor.getattr("data_ptr")?.call0()?.extract()
        };

        let event_time =
            unsafe { from_raw_parts(data_ptr("event_time")? as *const f32, event_count).to_vec() };
        let gradm_event =
            unsafe { from_raw_parts(data_ptr("gradm")? as *const [f32; 3], event_count).to_vec() };
        let adc_usage =
            unsafe { from_raw_parts(data_ptr("adc_usage")? as *const i32, event_count).to_vec() };

        sequence.push(Repetition {
            pulse_angle,
            pulse_phase,
            event_time,
            gradm_event,
            adc_mask: adc_usage.iter().map(|&u| u > 0).collect(),
        });
    }
    Ok(sequence)
}


// --- NEW multi-pulse API ---
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_graph_multi<'p>(
    py: Python<'p>,
    seq: &PyList,
    pulse_id_per_rep: &PyAny,

    t1: f32, t2: f32, t2dash: f32, d: f32,
    max_dist_count: usize, min_dist_mag: f32,
    nyquist: [f32; 3], size: [f32; 3], normalized_grads: bool,

    taps_bank: &PyList,        // list of taps objects (must have attrs pp,pz,mp,mz,zp,zz,deltas,dk)
    support_bank: &PyList,     // list of dicts: {"++":..., "+z":..., "-+":..., "-z":..., "z+":..., "zz":...}

    nz: usize, slice_thickness: f32, pad: f32,
) -> PyResult<&'p PyList> {
    println!(">>>> Rust - compute_graph_multi(...) >>>");
    let start = Instant::now();

    let sequence = parse_sequence(seq)?;
    let r = sequence.len();

    // pulse_id_per_rep
    let pid_i32 = read_i32_1d_any(pulse_id_per_rep, r)?;
    let pid: Vec<usize> = pid_i32.iter().map(|&x| x as usize).collect();

    let p = taps_bank.len();
    if support_bank.len() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "taps_bank and support_bank must have same length",
        ));
    }
    if pid.iter().any(|&x| x >= p) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "pulse_id_per_rep contains an id >= len(taps_bank)",
        ));
    }

    // parse taps_bank
    let mut taps_vec: Vec<RotTaps1D> = Vec::with_capacity(p);
    let mut support_vec: Vec<Support> = Vec::with_capacity(p);

    // Determine K from first taps
    let first = taps_bank.get_item(0)?;
    let k = tensor_numel(to_cpu_contig(first.getattr("pp")?)?)?;

    for idx in 0..p {
        let t = taps_bank.get_item(idx)?;
        let pp = t.getattr("pp")?;
        let pz_ = t.getattr("pz")?;
        let mp_ = t.getattr("mp")?;
        let mz_ = t.getattr("mz")?;
        let zp_ = t.getattr("zp")?;
        let zz_ = t.getattr("zz")?;
        let deltas = t.getattr("deltas")?;
        let dk: f32 = t.getattr("dk")?.extract().or_else(|_| {
            // dk could be numpy scalar or 0-d tensor; try .item()
            t.getattr("dk")?.call_method0("item")?.extract()
        })?;

        // enforce same K for all taps
        let k_pp = tensor_numel(to_cpu_contig(pp)?)?;
        if k_pp != k {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All taps in taps_bank must have same length K",
            ));
        }

        taps_vec.push(RotTaps1D {
            pp: read_cplx64_1d(pp, k)?,
            pz: read_cplx64_1d(pz_, k)?,
            mp: read_cplx64_1d(mp_, k)?,
            mz: read_cplx64_1d(mz_, k)?,
            zp: read_cplx64_1d(zp_, k)?,
            zz: read_cplx64_1d(zz_, k)?,
            deltas: read_i32_1d_any(deltas, k)?,
            dk,
        });

        // support dict
        let s = support_bank.get_item(idx)?;
        let sdict: &PyDict = s.downcast()?;

        let get = |k: &str| -> PyResult<&PyAny> {
            match sdict.get_item(k)? {        // <-- note the `?`
                Some(v) => Ok(v),
                None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("support missing key {k}")
                )),
            }
        };

        support_vec.push(Support {
            pp: read_i32_1d_any(get("++")?, tensor_numel(to_cpu_contig(get("++")?)?)?)?,
            pz: read_i32_1d_any(get("+z")?, tensor_numel(to_cpu_contig(get("+z")?)?)?)?,
            mp: read_i32_1d_any(get("-+")?, tensor_numel(to_cpu_contig(get("-+")?)?)?)?,
            mz: read_i32_1d_any(get("-z")?, tensor_numel(to_cpu_contig(get("-z")?)?)?)?,
            zp: read_i32_1d_any(get("z+")?, tensor_numel(to_cpu_contig(get("z+")?)?)?)?,
            zz: read_i32_1d_any(get("zz")?, tensor_numel(to_cpu_contig(get("zz")?)?)?)?,
        });
    }

    println!("Converting Python -> Rust: {} s", start.elapsed().as_secs_f32());
    let start = Instant::now();

    let grad_scale = if normalized_grads {
        [1.0 / size[0], 1.0 / size[1], 1.0 / size[2]]
    } else {
        [1.0, 1.0, 1.0]
    };

    let mut graph = comp_graph(
        &sequence,
        &pid,
        t1, t2, t2dash,
        d * 1e-9,
        max_dist_count, min_dist_mag,
        nyquist, grad_scale,
        &taps_vec,
        &support_vec,
        nz, slice_thickness, pad,
    );

    println!("Computing Graph: {} s", start.elapsed().as_secs_f32());
    let start = Instant::now();

    pre_pass::analyze_graph(&mut graph);
    println!("Analyzing Graph: {} s", start.elapsed().as_secs_f32());

    pre_pass_to_python(py, graph)
}

fn pre_pass_to_python<'p>(py: Python<'p>, graph: Vec<Vec<pre_pass::RcDist>>) -> PyResult<&'p PyList> {
    let mut mapping: HashMap<usize, Py<PyDistribution>> = HashMap::new();
    let address = |rc: &pre_pass::RcDist| rc.as_ptr() as usize;

    for rep in graph.iter() {
        for (dist, addr) in rep.iter().map(|d| (d.borrow(), address(d))) {
            let mut py_dist = PyDistribution::new(py);
            py_dist.dist_type = pre_pass::DIST_TYPE_STR[dist.dist_type as usize].to_owned();

            for ancestor in dist.ancestors.iter() {
                let py_ancestor = mapping
                    .get(&address(&ancestor.dist))
                    .expect("Setting ancestors: Dist missing in Rust->Python mapping")
                    .clone_ref(py);

                let tap_idx_obj = ancestor.tap_idx.into_py(py);

                py_dist.ancestors.as_ref().unwrap().as_ref(py).append(PyTuple::new(py, [
                    PyString::new(py, pre_pass::DIST_RELATION_STR[ancestor.relation as usize]).as_ref(),
                    py_ancestor.as_ref(py),
                    PyComplex::from_doubles(py, ancestor.rot_mat_factor.re as f64, ancestor.rot_mat_factor.im as f64).as_ref(),
                    tap_idx_obj.as_ref(py),
                ]))?;
            }

            py_dist.prepass_mag = {
                let mag = dist.mag;
                Some(PyComplex::from_doubles(py, mag.re as f64, mag.im as f64).into())
            };
            py_dist.latent_signal = dist.latent_signal;
            py_dist.latent_signal_unormalized = dist.latent_signal_unormalized;
            py_dist.signal = dist.signal;
            py_dist.emitted_signal = dist.emitted_signal;
            py_dist.prepass_kt_vec = dist.kt_vec;

            mapping.insert(addr, Py::new(py, py_dist)?);
        }
    }

    Ok(PyList::new(
        py,
        graph.into_iter().map(|rep| {
            PyList::new(py, rep.into_iter().map(|dist| mapping.get(&address(&dist)).unwrap()))
        }),
    ))
}

#[pymodule]
fn mrtini_prepass_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(compute_graph, m)?)?;
    m.add_function(wrap_pyfunction!(compute_graph_multi, m)?)?;
    m.add_class::<PyDistribution>()?;
    Ok(())
}