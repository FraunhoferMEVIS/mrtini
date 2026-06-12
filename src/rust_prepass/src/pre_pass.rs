
// '''
// # SPDX-License-Identifier: AGPL-3.0-only
// #
// # This file is part of MRTini Simulator.
// #
// # Portions of this file are derived from MRzero-Core:
// #   https://github.com/MRsources/MRzero-Core
// #
// # Original MRzero-Core copyright belongs to its respective authors.
// # Modifications and extensions in this file are Copyright (C) 2026
// # Copyright (C) 2026 Fraunhofer MEVIS, Germany
// # Contact <Snawar Hussain / Fraunhofer MEVIS> for details.
// # This derived work is distributed under the terms of the GNU Affero
// # GNU Affero General Public License v3.0 only. See LICENSE and NOTICE files
// # for details.
// '''
use num_complex::{Complex32, ComplexFloat};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;
use std::iter;
use std::rc::Rc;
pub type RcDist = Rc<RefCell<Distribution>>;
type DistVec = Vec<RcDist>;
type DistMap = HashMap<[i32; 4], RcDist>;

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub enum DistType {
    P,
    Z,
    #[default]
    Z0,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum DistRelation {
    PP,
    PZ,
    ZP,
    ZZ,
    MP,
    MZ,
}

pub static DIST_TYPE_STR: [&str; 3] = ["+", "z", "z0"];
pub static DIST_RELATION_STR: [&str; 6] = ["++", "+z", "z+", "zz", "-+", "-z"];

pub struct Edge {
    pub relation: DistRelation,
    pub rot_mat_factor: Complex32,
    pub dist: RcDist,
    pub tap_idx: i32, // NEW FUNCTIONALITY
}

#[derive(Clone, Copy)]
struct PState {
    mag: num_complex::Complex32,
    kt: [f64; 4],
    signal: f32,
}

#[derive(Default)]
pub struct Distribution {
    pub mag: Complex32,
    pub regrown_mag: f32, // Don't propagate to ancestors if its the own mag
    pub signal: f32,
    pub emitted_signal: f32, // relative to the strongest dist in the repetition
    pub kt_vec: [f64; 4],
    pub dist_type: DistType,
    pub ancestors: Vec<Edge>,
    /// latent_signal metric: if you want to measure states with a minimum signal of x,
    /// you should simulate states with a minimum latent_signal of <= x
    pub latent_signal: f32,
    pub latent_signal_unormalized: f32,
}

impl Distribution {
    fn measure(
        &mut self,
        t2dash: f32,
        mut nyquist: [f32; 3],
        nz: usize,
        slice_thickness: f32,
        pad: f32,
    ) {
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let pass_axis = |nyq: f32, k: f64| sigmoid((nyq - k.abs() as f32 + 0.5) * 100.0);

        nyquist[2] = (nz as f32) / 2.0 / (slice_thickness * pad);

        let kx = self.kt_vec[0];
        let ky = self.kt_vec[1];
        let kz = self.kt_vec[2];
        let deph =
            pass_axis(nyquist[0], kx) * pass_axis(nyquist[1], ky) * pass_axis(nyquist[2], kz);

        // numpy sinc(x) = sin(pi x)/(pi x)
        let x = (kz as f32) * slice_thickness;
        let wz = if x == 0.0 {
            1.0
        } else {
            (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
        };

        let k_len = ((kx * kx + ky * ky + kz * kz).sqrt()) as f32;

        self.signal += self.mag.norm()
            * (-(self.kt_vec[3] as f32 / t2dash.max(1e-12)).abs()).exp()
            * (1.0 + k_len).powf(-2.2)
            * deph
            * wz.abs();
    }
}

pub struct Repetition {
    pub pulse_angle: f32,
    pub pulse_phase: f32,
    pub event_time: Vec<f32>,
    pub gradm_event: Vec<[f32; 3]>,
    pub adc_mask: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub fn comp_graph(
    seq: &[Repetition],
    pulse_id_per_rep: &[usize],
    t1: f32,
    t2: f32,
    t2dash: f32,
    d: f32,
    max_dist_count: usize,
    min_dist_mag: f32,
    nyquist: [f32; 3],
    grad_scale: [f32; 3],
    taps_bank: &[RotTaps1D],
    support_bank: &[Support],
    nz: usize,
    slice_thickness: f32,
    pad: f32,
) -> Vec<Vec<RcDist>> {
    // Make precision of state merging dependent on the units used in the seq.
    // For this, we use a fraction of the smallest used step.

    let min_kt_step = [
        seq.iter()
            .flat_map(|r| r.gradm_event.iter().map(|g| g[0].abs() as f64))
            .filter(|x| *x > 1e-3)
            .min_by(f64::total_cmp),
        seq.iter()
            .flat_map(|r| r.gradm_event.iter().map(|g| g[1].abs() as f64))
            .filter(|x| *x > 1e-3)
            .min_by(f64::total_cmp),
        seq.iter()
            .flat_map(|r| r.gradm_event.iter().map(|g| g[2].abs() as f64))
            .filter(|x| *x > 1e-3)
            .min_by(f64::total_cmp),
        seq.iter()
            .flat_map(|r| r.event_time.iter().map(|t| *t as f64))
            .filter(|x| *x > 1e-6)
            .min_by(f64::total_cmp),
    ];
    let inv_kt_grid = [
        1.0 / (0.1 * min_kt_step[0].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[1].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[2].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[3].unwrap_or(1e-3)).clamp(1e-9, 1e-3),
    ];
    let mut graph: Vec<Vec<RcDist>> = Vec::new();

    let mut dists_p = DistVec::new();
    let mut dists_z = DistVec::new();
    let mut dist_z0 = Rc::new(RefCell::new(Distribution {
        mag: Complex32::new(1.0, 0.0),
        dist_type: DistType::Z0,
        ..Default::default()
    }));

    graph.push(vec![dist_z0.clone()]);

    assert_eq!(
        seq.len(),
        pulse_id_per_rep.len(),
        "pulse_id_per_rep must match #reps"
    );
    for (ri, rep) in seq.iter().enumerate() {
        let pulse_id = pulse_id_per_rep[ri];
        assert!(pulse_id < taps_bank.len() && pulse_id < support_bank.len());
        {
            let taps = rephase_taps(&taps_bank[pulse_id], rep.pulse_phase);
            let (_dists_p, _dists_z, _dist_z0) = apply_pulse(
                &dists_p,
                &dists_z,
                &dist_z0,
                &taps,
                &support_bank[pulse_id],
                max_dist_count,
                min_dist_mag,
                inv_kt_grid,
            );
            dists_p = _dists_p;
            dists_z = _dists_z;
            dist_z0 = _dist_z0;
        }
        graph.push(
            iter::once(&dist_z0)
                .chain(&dists_p)
                .chain(&dists_z)
                .cloned()
                .collect(),
        );
        // Build per-dist states (cheap)
        let mut states: Vec<PState> = dists_p
            .iter()
            .map(|d| {
                let b = d.borrow();
                PState {
                    mag: b.mag,
                    kt: b.kt_vec,
                    signal: b.signal,
                }
            })
            .collect();

        let r2_vec: Vec<f32> = rep.event_time.iter().map(|dt| (-dt / t2).exp()).collect();
        use std::f32::consts::TAU;

        // Parallel update
        let event_time = &rep.event_time;
        let gradm = &rep.gradm_event;
        let adc_mask = &rep.adc_mask;
        let d_si = d; // already in m^2/s in your comp_graph call

        // Optional: avoid rayon overhead for tiny work
        if states.len() < 1 {
            for s in states.iter_mut() {
                for (((r2, g), dt), adc) in r2_vec.iter().zip(gradm).zip(event_time).zip(adc_mask) {
                    let k1 = s.kt;
                    s.kt[0] += g[0] as f64 * grad_scale[0] as f64;
                    s.kt[1] += g[1] as f64 * grad_scale[1] as f64;
                    s.kt[2] += g[2] as f64 * grad_scale[2] as f64;
                    s.kt[3] += *dt as f64;
                    let k2 = s.kt;

                    let b = (1.0 / 3.0)
                        * TAU
                        * TAU
                        * dt
                        * ((k1[0] * k1[0] + k1[0] * k2[0] + k2[0] * k2[0])
                            + (k1[1] * k1[1] + k1[1] * k2[1] + k2[1] * k2[1])
                            + (k1[2] * k1[2] + k1[2] * k2[2] + k2[2] * k2[2]))
                            as f32;

                    s.mag *= (*r2) * (-b * d_si).exp();
                    if *adc {
                        // inline of measure() to avoid borrowing Distribution
                        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                        let pass = |nyq: f32, k: f64| sigmoid((nyq - k.abs() as f32 + 0.5) * 100.0);
                        // let nyq_z = (nz as f32) / 2.0 / (slice_thickness * pad); // we need correct nyquist for z because of slice profiles
                        let deph = pass(nyquist[0], s.kt[0])
                            * pass(nyquist[1], s.kt[1]);
                        let klen = ((s.kt[0] * s.kt[0] + s.kt[1] * s.kt[1] + s.kt[2] * s.kt[2])
                            .sqrt()) as f32;
                        let x = (s.kt[2] as f32) * slice_thickness;
                        let wz = if x.abs()<1e-8 {1.0} else {(std::f32::consts::PI*x).sin()/(std::f32::consts::PI*x)};
                        s.signal += s.mag.norm()
                            * (-(s.kt[3] as f32 / t2dash).abs()).exp()
                            * (1.0 + klen).powf(-2.2)
                            * wz.abs()
                            * deph;
                    }
                }
            }
        } else {
            states.par_iter_mut().for_each(|s| {
                for (((r2, g), dt), adc) in r2_vec.iter().zip(gradm).zip(event_time).zip(adc_mask) {
                    let k1 = s.kt;
                    s.kt[0] += g[0] as f64 * grad_scale[0] as f64;
                    s.kt[1] += g[1] as f64 * grad_scale[1] as f64;
                    s.kt[2] += g[2] as f64 * grad_scale[2] as f64;
                    s.kt[3] += *dt as f64;
                    let k2 = s.kt;

                    let b = (1.0 / 3.0)
                        * TAU
                        * TAU
                        * dt
                        * ((k1[0] * k1[0] + k1[0] * k2[0] + k2[0] * k2[0])
                            + (k1[1] * k1[1] + k1[1] * k2[1] + k2[1] * k2[1])
                            + (k1[2] * k1[2] + k1[2] * k2[2] + k2[2] * k2[2]))
                            as f32;

                    s.mag *= (*r2) * (-b * d_si).exp();
                    if *adc {
                        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                        let pass = |nyq: f32, k: f64| sigmoid((nyq - k.abs() as f32 + 0.5) * 100.0);
                        // let nyq_z = (nz as f32) / 2.0 / (slice_thickness * pad); // we need correct nyquist for z because of slice profiles
                        let deph = pass(nyquist[0], s.kt[0])
                            * pass(nyquist[1], s.kt[1]);
                        let klen = ((s.kt[0] * s.kt[0] + s.kt[1] * s.kt[1] + s.kt[2] * s.kt[2])
                            .sqrt()) as f32;
                        let x = (s.kt[2] as f32) * slice_thickness;
                        let wz = if x.abs()<1e-8 {1.0} else {(std::f32::consts::PI*x).sin()/(std::f32::consts::PI*x)};
                        s.signal += s.mag.norm()
                            * (-(s.kt[3] as f32 / t2dash).abs()).exp()
                            * (1.0 + klen).powf(-2.2)
                            *wz.abs()
                            * deph;
                    }
                }
            });
        }

        // Write back into the Rc<RefCell<...>> (serial, cheap)
        for (d, s) in dists_p.iter().zip(states.iter()) {
            let mut b = d.borrow_mut();
            b.mag = s.mag;
            b.kt_vec = s.kt;
            b.signal = s.signal;
        }

        // Apply relaxation to z states
        let rep_time: f32 = rep.event_time.iter().sum();
        let r1 = (-rep_time / t1).exp();

        for mut dist in dists_z.iter().map(|d| d.borrow_mut()) {
            let sqr = |x| x * x;
            let k2 = sqr(dist.kt_vec[0]) + sqr(dist.kt_vec[1]) + sqr(dist.kt_vec[2]);

            dist.mag *= r1 * (-d * rep_time * k2 as f32).exp();
        }
        // Z0 has no diffusion because it's not dephased
        dist_z0.borrow_mut().mag *= r1;
        dist_z0.borrow_mut().mag += 1.0 - r1;
        dist_z0.borrow_mut().regrown_mag += 1.0 - r1;
    }
    graph
}

fn apply_pulse(
    dists_p: &[RcDist],
    dists_z: &[RcDist],
    dist_z0: &RcDist,
    taps: &RotTaps1D,
    support: &Support,
    max_dist_count: usize,
    min_dist_mag: f32,
    inv_kt_grid: [f64; 4],
) -> (DistVec, DistVec, RcDist) {
    let mut dist_dict_p: DistMap = HashMap::default();
    let mut dist_dict_z: DistMap = HashMap::default();
    // rough guess: previous states * avg support fanout
    dist_dict_p.reserve(dists_p.len() * 8 + dists_z.len() * 4 + 64);
    dist_dict_z.reserve(dists_p.len() * 8 + dists_z.len() * 4 + 64);
    let mut add_dist = |kt_vec: [f64; 4],
                        mag: Complex32,
                        w: Complex32,
                        relation: DistRelation,
                        ancestor: &RcDist,
                        tap_idx: i32| {
        let key = [
            (kt_vec[0] * inv_kt_grid[0]).round() as i32,
            (kt_vec[1] * inv_kt_grid[1]).round() as i32,
            (kt_vec[2] * inv_kt_grid[2]).round() as i32,
            (kt_vec[3] * inv_kt_grid[3]).round() as i32,
        ];
        let dist_type = match relation {
            DistRelation::PP | DistRelation::MP | DistRelation::ZP => DistType::P,
            DistRelation::PZ | DistRelation::MZ | DistRelation::ZZ => DistType::Z,
        };
        let dict = if dist_type == DistType::P {
            &mut dist_dict_p
        } else {
            &mut dist_dict_z
        };
        let mag_new = mag * w;

        match dict.get(&key) {
            Some(d) => {
                let mut b = d.borrow_mut();
                b.mag += mag_new;
                b.ancestors.push(Edge {
                    relation,
                    rot_mat_factor: w,
                    dist: ancestor.clone(),
                    tap_idx,
                });
            }
            None => {
                let dist = Distribution {
                    mag: mag_new,
                    kt_vec,
                    dist_type,
                    ancestors: {
                        let mut v = Vec::with_capacity(1);
                        v.push(Edge {
                            relation,
                            rot_mat_factor: w,
                            dist: ancestor.clone(),
                            tap_idx,
                        });
                        v
                    },
                    ..Default::default()
                };
                dict.insert(key, Rc::new(RefCell::new(dist)));
            }
        }
    };

    let dk = taps.dk as f64;

    // ---- From z0 and z: ZZ -> z, ZP -> +
    for dist in std::iter::once(dist_z0).chain(dists_z.iter()) {
        let mag = dist.borrow().mag;
        let kt = dist.borrow().kt_vec;
        let (kx, ky, kz, t) = (kt[0], kt[1], kt[2], kt[3]);

        for &j in &support.zz {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx, ky, kz + delta * dk, t],
                mag,
                taps.zz[jj],
                DistRelation::ZZ,
                dist,
                j,
            );
        }
        for &j in &support.zp {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx, ky, kz + delta * dk, t],
                mag,
                taps.zp[jj],
                DistRelation::ZP,
                dist,
                j,
            );
        }
    }

    // ---- From +: PP -> +, PZ -> z, and minus branches MP/MZ with -kt and kz_n - delta*dk
    for dist in dists_p.iter() {
        let mag = dist.borrow().mag;
        let kt = dist.borrow().kt_vec;
        let (kx, ky, kz, t) = (kt[0], kt[1], kt[2], kt[3]);

        for &j in &support.pp {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx, ky, kz + delta * dk, t],
                mag,
                taps.pp[jj],
                DistRelation::PP,
                dist,
                j,
            );
        }
        for &j in &support.pz {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx, ky, kz + delta * dk, t],
                mag,
                taps.pz[jj],
                DistRelation::PZ,
                dist,
                j,
            );
        }

        let magc = mag.conj();
        let (kx_n, ky_n, kz_n, t_n) = (-kx, -ky, -kz, -t);

        for &j in &support.mp {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx_n, ky_n, kz_n + delta * dk, t_n],
                magc,
                taps.mp[jj],
                DistRelation::MP,
                dist,
                j,
            );
        }
        for &j in &support.mz {
            let jj = j as usize;
            let delta = taps.deltas[jj] as f64;
            add_dist(
                [kx_n, ky_n, kz_n + delta * dk, t_n],
                magc,
                taps.mz[jj],
                DistRelation::MZ,
                dist,
                j,
            );
        }
    }

    // ---- Extract z0; if vanished, recreate (your Python behavior)
    let dist_z0 = match dist_dict_z.remove(&[0, 0, 0, 0]) {
        Some(z0) => {
            z0.borrow_mut().dist_type = DistType::Z0;
            z0
        }
        None => Rc::new(RefCell::new(Distribution {
            mag: Complex32::new(0.0, 0.0),
            kt_vec: [0.0, 0.0, 0.0, 0.0],
            dist_type: DistType::Z0,
            ..Default::default()
        })),
    };

    // ---- sort + truncate (same as old)
// ---- sort + truncate (keep any P dist that has delta==0 contribution)
let mut dists_p: DistVec = dist_dict_p.into_values().collect();
let mut dists_z: DistVec = dist_dict_z.into_values().collect();
let mag2 = |d: &RcDist| d.borrow().mag.norm_sqr();

// sort descending by |mag|^2
dists_p.sort_unstable_by(|a, b| mag2(b).partial_cmp(&mag2(a)).unwrap());
dists_z.sort_unstable_by(|a, b| mag2(b).partial_cmp(&mag2(a)).unwrap());

// "came from delta==0 tap" predicate (robust to kz float noise)
let has_delta0 = |d: &RcDist, taps: &RotTaps1D| {
    let b = d.borrow();
    b.ancestors.iter().any(|e| {
        let j = e.tap_idx as usize;            // tap_idx is j from support arrays
        taps.deltas[j] == 0                    // delta*dk == 0 branch
    })
};

let truncate_keep_pred = |dists: &mut DistVec,
                          taps: &RotTaps1D,
                          keep_pred: &dyn Fn(&RcDist, &RotTaps1D) -> bool| {
    let thr = min_dist_mag * min_dist_mag;
    let mut kept: DistVec = Vec::with_capacity(max_dist_count);

    // 1) keep all "must keep" (even if tiny), in current sorted order
    for d in dists.iter() {
        if keep_pred(d, taps) {
            kept.push(d.clone());
            if kept.len() == max_dist_count { break; }
        }
    }

    // 2) fill remaining slots by magnitude until threshold/cap
    for d in dists.iter() {
        if kept.len() == max_dist_count { break; }
        if keep_pred(d, taps) { continue; } // already kept
        if mag2(d) < thr { break; }         // sorted => can stop
        kept.push(d.clone());
    }

    dists.clear();
    dists.extend(kept);
};

// Guardrail only for P (your issue); Z can stay old behavior
truncate_keep_pred(&mut dists_p, &taps, &has_delta0);

// original truncation for Z
{
    let thr = min_dist_mag * min_dist_mag;
    let first_small = dists_z.iter().position(|d| d.borrow().mag.norm_sqr() < thr);
    dists_z.truncate(first_small.map_or(max_dist_count, |i| max_dist_count.min(i)));
}

(dists_p, dists_z, dist_z0)
}

pub fn analyze_graph(graph: &mut Vec<Vec<RcDist>>) {
    // Tell each dist how large its signal is relative to the strongest dist
    for rep in graph.iter() {
        let max_signal = rep
            .iter()
            .map(|d| d.borrow().signal)
            .reduce(f32::max)
            .expect("Tried to find maximum signal but repetition is empty");

        for mut dist in rep.iter().map(|d| d.borrow_mut()) {
            dist.emitted_signal = if max_signal < 1e-9 {
                0.0
            } else {
                dist.signal / max_signal
            };
        }
    }

    // Calculate the latent_signal metric that is used to determine which states to simulate:
    // The latent_signal is the maximum of the own signal of the state or one of the propagated
    // latent_signals of its children. Propagation is relattive to the state with the largest
    // contribution: This ancestor is propagated the whole latent_signal.
    for rep in graph.iter().rev() {
        for mut dist in rep.iter().map(|d| d.borrow_mut()) {
            // Own signal is latent_signal if larger than what children produce
            dist.latent_signal = f32::max(dist.latent_signal, dist.emitted_signal);
            dist.latent_signal_unormalized = f32::max(dist.latent_signal_unormalized, dist.signal);

            let max_contrib = std::iter::once(dist.regrown_mag)
                .chain(
                    dist.ancestors
                        .iter()
                        .map(|e| (e.rot_mat_factor * e.dist.borrow().mag).norm()),
                )
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();

            // max_contrib could be the regrown mag, in this case no ancestor gets the full latent_signal
            for anc in dist.ancestors.iter() {
                let contrib = (anc.rot_mat_factor * anc.dist.borrow().mag).norm() / max_contrib;
                let tmp = anc.dist.borrow().latent_signal;
                let tmp_unormalized = anc.dist.borrow().latent_signal_unormalized;
                anc.dist.borrow_mut().latent_signal = f32::max(tmp, dist.latent_signal * contrib);
                anc.dist.borrow_mut().latent_signal_unormalized =
                    f32::max(tmp_unormalized, dist.latent_signal_unormalized * contrib);
            }
        }
    }
}

pub struct Support {
    pub pp: Vec<i32>, // indices j
    pub pz: Vec<i32>,
    pub mp: Vec<i32>,
    pub mz: Vec<i32>,
    pub zp: Vec<i32>,
    pub zz: Vec<i32>,
}

pub struct RotTaps1D {
    pub pp: Vec<Complex32>,
    pub pz: Vec<Complex32>,
    pub mp: Vec<Complex32>,
    pub mz: Vec<Complex32>,
    pub zp: Vec<Complex32>,
    pub zz: Vec<Complex32>,
    pub deltas: Vec<i32>, // signed deltas per tap index j
    pub dk: f32,          // cycles/m per delta
}

#[inline]
pub fn rephase_taps(t0: &RotTaps1D, phase: f32) -> RotTaps1D {
    let i = Complex32::i();
    let e_p = (i * phase).exp(); // e^{+iφ}
    let e_m = (-i * phase).exp(); // e^{-iφ}
    let e_2p = (i * (2.0 * phase)).exp(); // e^{+i2φ}
    RotTaps1D {
        pp: t0.pp.clone(),
        pz: t0.pz.iter().map(|v| *v * e_m).collect(),
        mp: t0.mp.iter().map(|v| *v * e_2p).collect(),
        mz: t0.mz.iter().map(|v| *v * e_p).collect(),
        zp: t0.zp.iter().map(|v| *v * e_p).collect(),
        zz: t0.zz.clone(),
        deltas: t0.deltas.clone(),
        dk: t0.dk,
    }
}
