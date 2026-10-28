"""Python marshaling contract only: native functions are recorded test doubles."""
from types import SimpleNamespace
import pytest
import torch

NAMES=('pp','pz','mp','mz','zp','zz')
RELS=('++','+z','-+','-z','z+','zz')


def data():
    return SimpleNamespace(T1=torch.tensor([1.,3.]),T2=torch.tensor([.1,.3]),T2dash=torch.tensor([.2,.4]),D=torch.tensor([.5,1.5]),nyquist=torch.tensor([10.,20.,30.]),size=torch.tensor([.1,.2,.3]))


def taps():
    return SimpleNamespace(**{n:torch.tensor([complex(i+j+1,j) for j in range(12)],dtype=torch.complex128)[::2] for i,n in enumerate(NAMES)},deltas=torch.tensor([-3,-2,-1,0,1,2],dtype=torch.int64),dk=torch.tensor(25.))

@pytest.mark.parametrize('as_list',[True,False])
def test_bank_bridge_argument_order_dtype_and_values(modules,as_list):
    b=modules.bridge;native=modules.native
    native.compute_graph.reset_mock();sentinel=object();native.compute_graph.return_value=sentinel
    reps=[object(),object()];seq=reps if as_list else iter(reps)
    t=taps();support={r:torch.tensor([i],dtype=torch.int64) for i,r in enumerate(RELS)}
    pulse_ids = torch.zeros(len(reps), dtype=torch.int32)
    got = b.compute_graph_rust(
        seq,
        data(),
        pulse_id_per_rep=pulse_ids,
        taps_bank=[t],
        support_bank=[support],
        z=torch.zeros(7),
        max_state_count=31,
        min_state_mag=1e-5,
        normalized_grads=True,
        slice_thickness=.005,
        pad=2.,
    )
    assert got is sentinel;native.compute_graph.assert_called_once()
    a=native.compute_graph.call_args.args
    assert a[0]==reps
    assert a[1].dtype==torch.int32 and a[1].device.type=='cpu'
    torch.testing.assert_close(a[1],pulse_ids)
    assert a[2:8]==pytest.approx((2.,.2,.3,1.,31,1e-5))
    assert a[8]==(10.,20.,30.) and a[9]==pytest.approx((.1,.2,.3)) and a[10] is True
    for i,n in enumerate(NAMES):
        x=getattr(a[11][0],n);assert x.dtype==torch.complex64 and x.device.type=='cpu' and x.is_contiguous()
        torch.testing.assert_close(x,getattr(t,n).to(torch.complex64))
    assert a[11][0].deltas.dtype==torch.int32 and a[11][0].dk==25.
    for relation in RELS:
        assert a[12][0][relation].dtype==torch.int32
    assert [a[12][0][relation].item() for relation in RELS]==list(range(6))
    assert a[13:]==(7,.005,2.)


def test_multi_bridge(modules):
    native=modules.native;native.compute_graph.reset_mock()
    t=taps();support={r:torch.tensor([i]) for i,r in enumerate(RELS)}
    reps=[object()];ids=torch.tensor([0],dtype=torch.int64)
    modules.bridge.compute_graph_rust(iter(reps),data(),ids,[t],[support],torch.zeros(9),max_state_count=40)
    a=native.compute_graph.call_args.args
    assert a[0]==reps and a[1].dtype==torch.int32 and a[1].device.type=='cpu'
    assert a[2:8]==pytest.approx((2.,.2,.3,1.,40,1e-8))
    for n in NAMES: assert getattr(a[11][0],n).dtype==torch.complex64
    assert a[12][0]['++'].dtype==torch.int32 and a[13:]==(9,1.,1.)
