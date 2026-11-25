import unittest, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from native import PhantomCache

class TestPhantomCache(unittest.TestCase):
    def test_reuse_invalidate_and_isolate(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'phantom.npz';f.write_bytes(b'x')
            c=PhantomCache(); calls=[]
            def build():
                calls.append(1); return SimpleNamespace(PD=torch.tensor([1.,2.]),size=torch.tensor([.2,.2,.005]),nested={'B0':torch.tensor([3.,4.])})
            k=lambda n=128,i=15,t=.005: c.identity(f,n,i,t)
            a,hit=c.get(k(),build);self.assertFalse(hit)
            a.PD.zero_();a.nested['B0'].zero_();a.size[-1]=2
            b,hit=c.get(k(),build);self.assertTrue(hit);self.assertEqual(len(calls),1)
            torch.testing.assert_close(b.PD,torch.tensor([1.,2.]));torch.testing.assert_close(b.nested['B0'],torch.tensor([3.,4.]));torch.testing.assert_close(b.size,torch.tensor([.2,.2,.005]))
            for key in (k(64),k(64,16),k(64,16,.004)):
                _,hit=c.get(key,build);self.assertFalse(hit)
            f.write_bytes(b'changed')
            _,hit=c.get(k(64,16,.004),build);self.assertFalse(hit);self.assertEqual(len(calls),5)
    def test_noncopyable_fallback(self):
        class Uncopyable:
            def __deepcopy__(self,memo):raise TypeError('unsupported')
        c=PhantomCache();calls=[]
        def build():calls.append(1);return Uncopyable()
        a,hit=c.get('x',build);self.assertFalse(hit)
        b,hit=c.get('x',build);self.assertFalse(hit);self.assertIsNot(a,b);self.assertEqual(len(calls),2)
    def test_failure_not_cached(self):
        c=PhantomCache()
        def fail():raise RuntimeError('load failed')
        with self.assertRaises(RuntimeError):c.get('x',fail)
        self.assertIsNone(c.key);self.assertIsNone(c.data)
        _,hit=c.get('x',lambda:{'PD':torch.ones(2)});self.assertFalse(hit)
if __name__=='__main__':unittest.main()
