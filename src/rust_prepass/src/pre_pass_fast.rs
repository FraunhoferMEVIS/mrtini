// SPDX-License-Identifier: AGPL-3.0-or-later
//! Arena prepass: one topology traversal for one or two independent species.
//! No candidate Rc/RefCell allocations and no Python edge objects.
use num_complex::Complex32;
use rayon::prelude::*;
use rustc_hash::FxHashMap as Map;
use crate::pre_pass::{Repetition,RotTaps1D,Support};

pub struct Case { pub tissue:[f32;4],pub taps:Vec<RotTaps1D>,pub support:Vec<Support> }
#[derive(Clone,Copy)]
pub struct Edge<const S:usize>{pub parent:usize,pub relation:u8,pub tap:i32,pub mask:u8,pub contribution:[f32;S]}
pub struct Node<const S:usize>{
    pub kind:u8,pub mask:u8,pub kt_start:[f64;4],pub kt:[f64;4],pub mag:[Complex32;S],
    pub regrown:[f32;S],pub signal:[f32;S],pub emitted:[f32;S],pub latent:[f32;S],pub latent_abs:[f32;S],
    pub edge_start:usize,pub edge_end:usize,
    rank:[usize;S],
}
impl<const S:usize> Node<S>{fn new(kind:u8,mask:u8,kt:[f64;4],mag:[Complex32;S])->Self{
    Self{kind,mask,kt_start:kt,kt,mag,regrown:[0.;S],signal:[0.;S],emitted:[0.;S],latent:[0.;S],latent_abs:[0.;S],edge_start:0,edge_end:0,rank:[0;S]}
}}
pub struct Graph<const S:usize>{pub nodes:Vec<Node<S>>,pub edges:Vec<Edge<S>>,pub layers:Vec<usize>,pub candidates:usize,pub candidate_edges:usize}
struct Candidate<const S:usize>{kind:u8,key:[i32;4],case_kt:[[f64;4];S],mag:[Complex32;S],exists:u8,delta0:u8}
struct Proposal<const S:usize>{child:usize,value:[Complex32;S],edge:Edge<S>}
struct Tap<const S:usize>{index:i32,delta:i32,mask:u8,weight:[Complex32;S]}
// The legacy sorter operates on non-Copy Rc handles. A non-Copy index handle
// avoids the primitive-integer small-sort specialization changing equal-score
// order while retaining allocation-free candidates.
struct SortIndex(usize);
impl Drop for SortIndex{fn drop(&mut self){}}
const ZERO:Complex32=Complex32::new(0.,0.);
fn coeff(t:&RotTaps1D,r:usize,j:usize)->Complex32{match r{0=>t.pp[j],1=>t.pz[j],2=>t.zp[j],3=>t.zz[j],4=>t.mp[j],_=>t.mz[j]}}
fn support(s:&Support,r:usize)->&[i32]{match r{0=>&s.pp,1=>&s.pz,2=>&s.zp,3=>&s.zz,4=>&s.mp,_=>&s.mz}}
fn inv_grid(seq:&[Repetition])->[f64;4]{
    let mut m=[f64::INFINITY;4];
    for r in seq{for g in &r.gradm_event{for a in 0..3{let v=g[a].abs() as f64;if v>1e-3{m[a]=m[a].min(v)}}}for t in &r.event_time{let v=*t as f64;if v>1e-6{m[3]=m[3].min(v)}}}
    std::array::from_fn(|a|{let fallback=if a==3{1e-3}else{1.};let v=if m[a].is_finite(){m[a]}else{fallback};1./(0.1*v).clamp(if a==3{1e-9}else{1e-6},if a==3{1e-3}else{1.})})
}
fn make_taps<const S:usize>(cases:&[Case],pid:usize,phase:f32)->[Vec<Tap<S>>;6]{
    let ep=Complex32::new(phase.cos(),phase.sin());
    let phases=[Complex32::new(1.,0.),ep.conj(),ep,Complex32::new(1.,0.),ep*ep,ep];
    std::array::from_fn(|r|{
        let mut indices=Vec::new();for c in cases.iter().take(S){indices.extend_from_slice(support(&c.support[pid],r));}indices.sort_unstable();indices.dedup();
        indices.into_iter().map(|j|{
            let mut mask=0;let mut weight=[ZERO;S];
            for s in 0..S{if support(&cases[s].support[pid],r).contains(&j){mask|=1<<s;weight[s]=coeff(&cases[s].taps[pid],r,j as usize)*phases[r];}}
            Tap{index:j,delta:cases[0].taps[pid].deltas[j as usize],mask,weight}
        }).collect()
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build<const S:usize>(seq:&[Repetition],pid:&[usize],cases:&[Case],cap:usize,min_mag:f32,nyquist:[f32;3],scale:[f32;3],thickness:f32)->Graph<S>{
    assert!(S==1||S==2);assert_eq!(cases.len(),S);assert_eq!(seq.len(),pid.len());assert!(cap>0);
    let all=(1u8<<S)-1;let inv=inv_grid(seq);
    let mut graph=Graph{nodes:vec![Node::new(2,all,[0.;4],[Complex32::new(1.,0.);S])],edges:Vec::new(),layers:vec![0,1],candidates:0,candidate_edges:0};
    for (ri,rep) in seq.iter().enumerate(){
        let prev_start=graph.layers[ri];let prev_end=graph.layers[ri+1];let prev=&graph.nodes[prev_start..prev_end];
        let np=prev.iter().filter(|x|x.kind==0).count();let nz=prev.len()-np-1;let reserve=np*8+nz*4+64;
        let mut pm:Map<[i32;4],usize>=Map::default();let mut zm:Map<[i32;4],usize>=Map::default();pm.reserve(reserve);zm.reserve(reserve);
        let taps=make_taps::<S>(cases,pid[ri],rep.pulse_phase);let dk=cases[0].taps[pid[ri]].dk as f64;
        let mut candidates:Vec<Candidate<S>>=Vec::with_capacity(reserve*2);let mut proposals:Vec<Proposal<S>>=Vec::new();let mut parent_ranges=vec![(0,0);prev.len()];
        for is_transverse in [false,true]{for (parent,p) in prev.iter().enumerate(){
            if (p.kind==0)!=is_transverse{continue}
            parent_ranges[parent].0=proposals.len();
            let relations:&[usize]=if is_transverse{&[0,1,4,5]}else{&[3,2]};
            for &relation in relations{for tap in &taps[relation]{
                let mask=p.mask&tap.mask;if mask==0{continue}
                let negative=relation==4||relation==5;let mut kt=p.kt;
                if negative{for a in &mut kt{*a = -*a;}}kt[2]+=tap.delta as f64*dk;
                let key=std::array::from_fn(|a|(kt[a]*inv[a]).round() as i32);
                let kind=if relation==0||relation==2||relation==4{0}else{1};
                let map=if kind==0{&mut pm}else{&mut zm};
                let child=*map.entry(key).or_insert_with(||{let i=candidates.len();candidates.push(Candidate{kind,key,case_kt:[kt;S],mag:[ZERO;S],exists:0,delta0:0});i});
                let c=&mut candidates[child];
                if tap.delta==0{c.delta0|=mask;}
                let mut contribution=[0.;S];let mut value=[ZERO;S];
                for s in 0..S{if mask&(1<<s)!=0{let m=if negative{p.mag[s].conj()}else{p.mag[s]};value[s]=m*tap.weight[s];contribution[s]=(p.mag[s]*tap.weight[s]).norm();}}
                if S==1{if c.exists&1==0{c.mag[0]=value[0];}else{c.mag[0]+=value[0];}}c.exists|=mask;
                proposals.push(Proposal{child,value,edge:Edge{parent,relation:relation as u8,tap:tap.index,mask,contribution}});
            }}
            parent_ranges[parent].1=proposals.len();
        }}
        graph.candidates+=candidates.len();graph.candidate_edges+=proposals.len();
        let z0=zm.remove(&[0;4]).unwrap_or_else(||{let i=candidates.len();candidates.push(Candidate{kind:2,key:[0;4],case_kt:[[0.;4];S],mag:[ZERO;S],exists:all,delta0:0});i});
        // Reuse geometry, but preserve each species' original parent/summation
        // order and equal-score hash/sort order. This prevents fat selections
        // from drifting merely because the water-first union reordered parents.
        let mut case_p:[Map<[i32;4],usize>;S]=std::array::from_fn(|s|{let np=prev.iter().filter(|n|n.kind==0&&n.mask&(1<<s)!=0).count();let nz=prev.iter().filter(|n|n.kind==1&&n.mask&(1<<s)!=0).count();let mut m=Map::default();if S>1{m.reserve(np*8+nz*4+64);}m});
        let mut case_z:[Map<[i32;4],usize>;S]=std::array::from_fn(|s|{let np=prev.iter().filter(|n|n.kind==0&&n.mask&(1<<s)!=0).count();let nz=prev.iter().filter(|n|n.kind==1&&n.mask&(1<<s)!=0).count();let mut m=Map::default();if S>1{m.reserve(np*8+nz*4+64);}m});
        let mut seen=vec![0u8;if S==1{0}else{candidates.len()}];
        if S==1{case_p[0]=pm;case_z[0]=zm;}else{
        for s in 0..S{let bit=1<<s;let mut parents:Vec<usize>=(0..prev.len()).filter(|i|prev[*i].mask&bit!=0).collect();parents.sort_unstable_by_key(|i|((prev[*i].kind==0) as u8,prev[*i].rank[s]));
            for parent in parents{let (lo,hi)=parent_ranges[parent];for p in &proposals[lo..hi]{if p.edge.mask&bit==0{continue}let c=&mut candidates[p.child];
                if seen[p.child]&bit==0{seen[p.child]|=bit;c.mag[s]=p.value[s];{let mut kt=prev[parent].kt;if p.edge.relation>=4{for k in &mut kt{*k = -*k;}}kt[2]+=cases[0].taps[pid[ri]].deltas[p.edge.tap as usize] as f64*dk;c.case_kt[s]=kt;}let m=if c.kind==0{&mut case_p[s]}else{&mut case_z[s]};m.insert(c.key,p.child);}else{c.mag[s]+=p.value[s];}
            }}case_z[s].remove(&[0;4]);if seen[z0]&(1<<s)==0{candidates[z0].case_kt[s]=[0.;4];}
        }}
        let mut selected=vec![0u8;candidates.len()];selected[z0]=all;
        let mut selected_rank=vec![[usize::MAX;S];candidates.len()];selected_rank[z0]=[0;S];let mut next_rank=[1usize;S];
        let mut order=vec![z0];let mut listed=vec![false;candidates.len()];listed[z0]=true;
        for kind in 0..2{for s in 0..S{
            let bit=1<<s;let base=if kind==0{&case_p[s]}else{&case_z[s]};
            let mut rank:Vec<SortIndex>=base.values().copied().map(SortIndex).collect();
            // Preserve the existing score order and delta0 guard. Arena
            // allocation/export changes are isolated from pruning policy.
            rank.sort_unstable_by(|a,b|candidates[b.0].mag[s].norm_sqr().partial_cmp(&candidates[a.0].mag[s].norm_sqr()).unwrap());
            let mut keep=Vec::with_capacity(cap);
            if kind==0{for index in &rank{let i=index.0;if candidates[i].delta0&bit!=0{keep.push(i);if keep.len()==cap{break}}}}
            for index in &rank{let i=index.0;if keep.len()==cap{break}if kind==0&&candidates[i].delta0&bit!=0{continue}if candidates[i].mag[s].norm_sqr()<min_mag*min_mag{break}keep.push(i);}
            for i in keep{selected[i]|=bit;selected_rank[i][s]=next_rank[s];next_rank[s]+=1;if !listed[i]{order.push(i);listed[i]=true;}}
        }}
        // Union exact physical coordinates across species. Rounded keys are
        // used only inside each species, as in the original prepass. If two
        // species chose different representatives of one rounded cell, split
        // that node rather than silently moving one species' physical k-vector.
        let mut local=vec![[usize::MAX;S];candidates.len()];let new_start=graph.nodes.len();
        for &cidx in &order{let c=&candidates[cidx];let mask=selected[cidx];
            let split=S==2&&mask==3&&c.case_kt[0]!=c.case_kt[1];
            let masks=if split{[1u8,2u8]}else{[mask,0]};
            for node_mask in masks{if node_mask==0{continue}let sid=if node_mask&1!=0{0}else{1};
                let kt=c.case_kt[sid];let mut mag=c.mag;
                let node_index=graph.nodes.len()-new_start;
                for s in 0..S{if node_mask&(1<<s)==0{mag[s]=ZERO;}else{local[cidx][s]=node_index;}}
                let kind=if cidx==z0{2}else{c.kind};let mut node=Node::new(kind,node_mask,kt,mag);node.rank=selected_rank[cidx];graph.nodes.push(node);
            }
        }
        let new_count=graph.nodes.len()-new_start;let mut counts=vec![0usize;new_count];
        for p in &proposals{let mask=selected[p.child]&p.edge.mask;let mut last=usize::MAX;
            for s in 0..S{if mask&(1<<s)==0{continue}let i=local[p.child][s];if i!=last{counts[i]+=1;last=i;}}
        }
        let mut cursor=graph.edges.len();for (i,n) in counts.iter().enumerate(){graph.nodes[new_start+i].edge_start=cursor;cursor+=n;graph.nodes[new_start+i].edge_end=cursor;}
        let edge_base=graph.edges.len();let total=cursor-edge_base;
        graph.edges.resize_with(edge_base+total,||Edge{parent:0,relation:0,tap:0,mask:0,contribution:[0.;S]});
        let mut write:Vec<usize>=graph.nodes[new_start..].iter().map(|n|n.edge_start).collect();
        for p in proposals{let mask=selected[p.child]&p.edge.mask;let mut last=usize::MAX;
            for s in 0..S{if mask&(1<<s)==0{continue}let i=local[p.child][s];if i==last{continue}last=i;
                let mut e=p.edge;e.mask=0;
                for t in 0..S{if mask&(1<<t)!=0&&local[p.child][t]==i{e.mask|=1<<t;}else{e.contribution[t]=0.;}}
                graph.edges[write[i]]=e;write[i]+=1;
            }
        }
        let r2:Vec<[f32;S]>=rep.event_time.iter().map(|dt|std::array::from_fn(|s|(-dt/cases[s].tissue[1]).exp())).collect();
        let work=(graph.nodes.len()-new_start)*rep.event_time.len();
        let advance=|n:&mut Node<S>|{
            if n.kind==0{for e in 0..rep.event_time.len(){
                let k1=n.kt;let dt=rep.event_time[e];for a in 0..3{n.kt[a]+=rep.gradm_event[e][a] as f64*scale[a] as f64;}n.kt[3]+=dt as f64;
                let k2=n.kt;let integral=(0..3).map(|a|k1[a]*k1[a]+k1[a]*k2[a]+k2[a]*k2[a]).sum::<f64>() as f32;
                let b=(1./3.)*std::f32::consts::TAU*std::f32::consts::TAU*dt*integral;
                for s in 0..S{if n.mask&(1<<s)!=0{n.mag[s]*=r2[e][s]*(-b*cases[s].tissue[3]).exp();}}
                if rep.adc_mask[e]{let pass=|nyq:f32,k:f64|1./(1.+(-((nyq-k.abs() as f32+0.5)*100.)).exp());
                    let deph=pass(nyquist[0],n.kt[0])*pass(nyquist[1],n.kt[1]);let klen=(n.kt[..3].iter().map(|x|x*x).sum::<f64>().sqrt()) as f32;
                    let x=n.kt[2] as f32*thickness;let wz=if x.abs()<1e-8{1.}else{(std::f32::consts::PI*x).sin()/(std::f32::consts::PI*x)};
                    let spatial=(1.+klen).powf(-2.2)*wz.abs()*deph;
                    for s in 0..S{if n.mask&(1<<s)!=0{n.signal[s]+=n.mag[s].norm()*(-(n.kt[3] as f32/cases[s].tissue[2]).abs()).exp()*spatial;}}
                }
            }}else{let time: f32=rep.event_time.iter().sum();let k2=n.kt[..3].iter().map(|x|x*x).sum::<f64>() as f32;
                for s in 0..S{if n.mask&(1<<s)!=0{let r1=(-time/cases[s].tissue[0]).exp();n.mag[s]*=r1*if n.kind==2{1.}else{(-cases[s].tissue[3]*time*k2).exp()};if n.kind==2{n.mag[s]+=1.-r1;n.regrown[s]=1.-r1;}}}
            }
        };
        if work>=32_768{graph.nodes[new_start..].par_iter_mut().for_each(advance)}else{graph.nodes[new_start..].iter_mut().for_each(advance)}
        graph.layers.push(graph.nodes.len());
    }
    for ri in 0..seq.len()+1{let a=graph.layers[ri];let b=graph.layers[ri+1];let max: [f32;S]=std::array::from_fn(|s|graph.nodes[a..b].iter().map(|n|n.signal[s]).fold(0.,f32::max));
        for n in &mut graph.nodes[a..b]{for s in 0..S{n.emitted[s]=if max[s]<1e-9{0.}else{n.signal[s]/max[s]};}}}
    for ri in (0..seq.len()+1).rev(){let a=graph.layers[ri];let b=graph.layers[ri+1];
        for i in a..b{let (la,lu,es,ee,maxc)={let n=&mut graph.nodes[i];for s in 0..S{n.latent[s]=n.latent[s].max(n.emitted[s]);n.latent_abs[s]=n.latent_abs[s].max(n.signal[s]);}
            let maxc:[f32;S]=std::array::from_fn(|s|graph.edges[n.edge_start..n.edge_end].iter().map(|e|e.contribution[s]).fold(n.regrown[s],f32::max));
            (n.latent,n.latent_abs,n.edge_start,n.edge_end,maxc)};
            if ri==0{continue}let parent_start=graph.layers[ri-1];
            for e in &graph.edges[es..ee]{let p=&mut graph.nodes[parent_start+e.parent];for s in 0..S{if e.mask&(1<<s)!=0&&maxc[s]>0.{let weight=e.contribution[s]/maxc[s];p.latent[s]=p.latent[s].max(la[s]*weight);p.latent_abs[s]=p.latent_abs[s].max(lu[s]*weight);}}}
        }
    }
    graph
}
