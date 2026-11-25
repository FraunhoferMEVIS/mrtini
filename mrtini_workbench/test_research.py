import numpy as np
import pytest
from types import SimpleNamespace as NS
from research import native_maps,overview,snapshot_graph,compare_nodes,comparison,preview_maps
from test_analysis_views import graph_fixture
from physics import Protocol,phantom

def test_maps_species_and_coordinates():
    d=NS(voxel_pos=np.array([[0.,0,0],[.001,0,0],[0.,0,0],[.001,.001,0]]),PD=np.array([.8,.5,.2,.6]),T1=np.array([1.,2.,.4,1.2]),T2=np.ones(4),T2dash=np.ones(4),D=np.ones(4),B0=np.array([0,1,-440,2]),B1=np.ones((1,4),complex))
    m=native_maps(d,np.array([0,0,1,0]));assert m['PD'].shape==(2,2);np.testing.assert_allclose(m['PD'],[[1,.5],[0,.6]]);assert m['fat_fraction'][0,0]==pytest.approx(.2);assert m['water_T1'][0,0]==pytest.approx(1);assert m['fat_T1'][0,0]==pytest.approx(.4);assert np.isnan(m['water_T1'][1,0]);assert m['fat_index'][0,0]==2

def test_summary_and_snapshot(tmp_path):
    g=graph_fixture();o=overview(g);assert sum(x['total'] for x in o)==7;assert sum(x['edges'] for x in o)==7
    p=tmp_path/'g.npz';snapshot_graph(g,p)
    with np.load(p) as a:
        c=compare_nodes(a,a);assert c['shared']==7 and c['reference_only']==0
        b={k:a[k].copy() for k in a.files};b['kt_end'][2,0]+=1;c=compare_nodes(a,b);assert c['shared']==6 and c['reference_only']==c['current_only']==1

def test_comparison_common_scale_and_trajectory():
    p=vars(Protocol());a={'complex_image':np.ones((2,2),complex),'signal':np.ones((4,1),complex),'kspace':np.zeros((4,2))};b={**a,'complex_image':a['complex_image']*2,'signal':a['signal']*2};c=comparison(a,b,p,p,{},{});assert c['limit']==2 and c['image_l2']==c['adc_l2']==1;assert (c['difference']==1).all();c=comparison(a,{**b,'kspace':np.ones((4,2))},p,{**p,'fov_mm':100},{},{});assert c['image_l2'] is None and c['adc_l2'] is None

def test_projection_ranges_stable():
    from analysis_views import graph_window
    g=graph_fixture();g.emitted[1]=0;g.latent[3]=0;a=graph_window(g,mode='water');b=graph_window(g,mode='fat');assert a['axis_ranges']==b['axis_ranges'];assert not any(n['id']==1 for n in graph_window(g,activity='emitting')['nodes']);assert not any(n['id']==3 for n in graph_window(g,activity='latent')['nodes'])

def test_preview_api(tmp_path,monkeypatch):
    import server,time
    from fastapi.testclient import TestClient
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.setattr(server,'jobs',{});monkeypatch.setattr(server,'active',None)
    with TestClient(server.app) as c:
        ids=[]
        for fa in (15,25):
            p=vars(Protocol(resolution=64,flip_deg=fa));r=c.post('/api/jobs',json={'protocol':p});assert r.status_code==200,r.text;i=r.json()['id'];ids.append(i)
            for _ in range(200):
                j=c.get('/api/jobs/'+i).json()
                if j['status'] in ('done','error'):break
                time.sleep(.025)
            assert j['status']=='done',j
            assert c.get(f'/api/jobs/{i}/maps?key=PD').status_code==200
            assert c.get(f'/api/jobs/{i}/voxel?x=32&y=32').status_code==200
            t=c.get(f'/api/jobs/{i}/trajectory').json();assert len(t['k'])==4096
            assert c.get(f'/api/jobs/{i}/saved').status_code==200
        assert len(c.get('/api/history').json())==2
        out=c.get(f'/api/jobs/{ids[1]}/compare/{ids[0]}');assert out.status_code==200,out.text;assert out.json()['same_adc']
        assert c.post('/api/draft-strip',json={'protocol':vars(Protocol(resolution=64))}).status_code==200

def test_cache_keys(tmp_path):
    from native import NativeRunner
    from dataclasses import replace
    f=tmp_path/'phantom.npz';f.write_bytes(b'x');r=NativeRunner(tmp_path);p=Protocol(backend='native',native_edit=True)
    def rf_key(p):return r.prepared_key(replace(p,max_states=800,min_state_mag=1e-4,workspace_mb=128,kernel_mode='dynamic'),None,f)
    assert rf_key(p)==rf_key(replace(p,max_states=200,workspace_mb=64,kernel_mode='specialized'))
    for q in [replace(p,global_rf=True),replace(p,b0_offset_hz=50),replace(p,b1_scale=.9),replace(p,te_ms=10),replace(p,flip_deg=30)]:assert rf_key(q)!=rf_key(p)

def test_archive_graph_roundtrip(tmp_path,monkeypatch):
    import server
    i='a'*32;d=tmp_path/'runs'/i;d.mkdir(parents=True);g=graph_fixture();fake=NS(engine=NS(graph=g))
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.setattr(server,'runner',fake);monkeypatch.setattr(server,'analysis_job',i);monkeypatch.setattr(server,'active',None);monkeypatch.setattr(server,'jobs',{i:{'status':'done','result':{'preview':False}}})
    out=server.archive_graph(i);assert out['bytes']>0
    monkeypatch.setattr(server,'analysis_job',None);a=server.graph_view(i);assert a['total_nodes']==7 and a['total_edges']==7 and a['case_species']==[0,1]
