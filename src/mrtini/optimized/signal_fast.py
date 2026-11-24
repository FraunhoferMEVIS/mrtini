# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact tissue-wise ADC contraction. No additional pathway/voxel threshold.

Factorize over equal (D, T2') pairs, not equal tissue labels. Other maps may
vary arbitrarily. Zero-PD voxels are omitted from signal only, never RF/state
propagation. The generic fallback bounds the state x ADC x voxel workspace.
"""
import math
import numpy as np
import torch

class SignalPlan:
    def __init__(self, data, D, T2d, max_groups=16):
        self.V = data.PD.numel()
        self.device = data.PD.device
        # Do this once, not once per repetition. Keep differentiable parameters
        # on the generic path: equal values need not have equal derivatives.
        self.groups = None
        self.active = None
        self.active_count = self.V
        if torch.is_grad_enabled() and (D.requires_grad or T2d.requires_grad or data.PD.requires_grad):
            return
        pd = data.PD.detach().reshape(-1).cpu().numpy()
        active = np.flatnonzero(pd != 0)
        self.active_count = len(active)
        if len(active) != self.V:
            self.active = torch.as_tensor(active, device=self.device, dtype=torch.long)
        if not len(active):
            self.groups = []
            return
        d = D.detach().reshape(-1).cpu().numpy()[active]
        t = T2d.detach().reshape(-1).cpu().numpy()[active]
        pairs = np.stack((d,t),axis=1)
        if not np.isfinite(d).all() or np.isnan(t).any() or (t==0).any():
            # Keep original IEEE behavior for nonphysical values.
            return
        unique, inv = np.unique(pairs, axis=0, return_inverse=True)
        if len(unique) > max_groups:
            return
        self.groups = []
        for i in range(len(unique)):
            idx = active[inv == i]
            ix = None if len(idx) == self.V else torch.as_tensor(idx,device=self.device,dtype=torch.long)
            self.groups.append((ix, float(unique[i,0]), float(unique[i,1]),len(idx)))

    @staticmethod
    def _select(x, ix):
        return x if ix is None else x.index_select(-1,ix)

    def contract(self, M, k0, adc, t, cb, deph, allowed, ids, data, D, T2d,
                 coil, *, state_batch, adc_batch, voxel_batch, workspace_bytes):
        """M [S,V], k0 [S,4], adc [A,4], t/cb/deph [S,A]."""
        S,V=M.shape; A=adc.shape[0]; C=coil.shape[1]
        out=M.new_zeros((A,C))
        if not S or not A or not self.active_count:
            return out
        if allowed is not None:
            M=M*allowed[:,ids]
        # Exact phase separation; spatial coordinates and B0 need not be uniform.
        Q=1.41421356237*M*torch.exp(2j*torch.pi*(k0[:,3:]*data.B0+k0[:,:3]@data.voxel_pos.T))
        if self.groups is not None:
            for ix,d,t2d,n in self.groups:
                H=deph*torch.exp(-t.abs()/t2d+d*cb)
                H=H.T.to(M.dtype).contiguous()
                q=self._select(Q,ix)
                b0=self._select(data.B0,ix); t2=self._select(data.T2.abs(),ix)
                pos=data.voxel_pos if ix is None else data.voxel_pos.index_select(0,ix)
                c=coil if ix is None else coil.index_select(0,ix)
                # Bound ADC x voxel intermediates; unlike old code, default
                # covers a 128 x 128 phantom in one voxel tile on typical readouts.
                ab=min(A, adc_batch or (A if M.is_cuda else 64))
                vb=min(n, voxel_batch or (n if M.is_cuda else 2048), max(1,workspace_bytes//(48*ab)))
                for al in range(0,A,ab):
                    ah=min(al+ab,A); a=adc[al:ah]
                    for vl in range(0,n,vb):
                        vh=min(vl+vb,n)
                        event=torch.exp(2j*torch.pi*(a[:,3:]*b0[vl:vh]+a[:,:3]@pos[vl:vh].T))*torch.exp(-a[:,3:]/t2[vl:vh])
                        out[al:ah]+=((H[al:ah]@q[:,vl:vh])*event)@c[vl:vh]
            return out
        # Continuous maps: keep exact voxel-specific damping. Reduce states
        # before coil projection and combine the two damping exponentials.
        ix=self.active
        q=self._select(Q,ix); d=self._select(D,ix); t2d=self._select(T2d,ix)
        b0=self._select(data.B0,ix); t2=self._select(data.T2.abs(),ix)
        pos=data.voxel_pos if ix is None else data.voxel_pos.index_select(0,ix)
        c=coil if ix is None else coil.index_select(0,ix)
        n=q.shape[1]; sb=min(S,state_batch if M.is_cuda else min(state_batch,16)); ab=min(A,adc_batch or (128 if M.is_cuda else 64))
        vb=min(n,voxel_batch or (n if M.is_cuda else 1024),max(1,workspace_bytes//(48*sb*ab)))
        for al in range(0,A,ab):
            ah=min(al+ab,A); a=adc[al:ah]
            for vl in range(0,n,vb):
                vh=min(vl+vb,n)
                event=torch.exp(2j*torch.pi*(a[:,3:]*b0[vl:vh]+a[:,:3]@pos[vl:vh].T))*torch.exp(-a[:,3:]/t2[vl:vh])
                accum=M.new_zeros((ah-al,vh-vl))
                for sl in range(0,S,sb):
                    sh=min(sl+sb,S)
                    damp=torch.exp(-t[sl:sh,al:ah,None].abs()/t2d[vl:vh]+cb[sl:sh,al:ah,None]*d[vl:vh])
                    accum=accum+(q[sl:sh,None,vl:vh]*damp*deph[sl:sh,al:ah,None]).sum(0)
                out[al:ah]+=(accum*event)@c[vl:vh]
        return out
