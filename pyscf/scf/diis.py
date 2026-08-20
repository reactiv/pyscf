#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

from functools import reduce
import numpy
import scipy.linalg
import scipy.optimize
from pyscf import lib
from pyscf.lib import logger

DEBUG = False

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.damp = 0

    # Adaptive-depth residual-norm trust window with angular-coherence
    # eviction for the retained CDIIS subspace.  Stale residuals recorded far
    # from the current iterate violate the local-linearity assumption behind
    # Pulay extrapolation; discarding them by residual-norm ratio keeps the
    # extrapolation conditioned (cf. Chupin, Dupuy, Legendre & Sere,
    # ESAIM M2AN 55 (2021) 2785, adaptive-depth restarted DIIS).
    # Fixed universal constants, applied identically to every system:
    _TRUST_RADIUS = 100.       # radial window ||e_i|| <= R*||e_new||
    _COHERENCE_THRESHOLD = 0.98  # |<e_i/||e_i||, e_j/||e_j||>| eviction bound
    _RETAIN_FLOOR = 3          # most recent pairs always retained

    def _trust_window(self, errvec):
        '''Slot indices of stored residuals inside the radial trust window
        ||e_i|| <= R * ||e_new|| (R = 100).'''
        nd = self.get_num_vec()
        radius = self._TRUST_RADIUS * numpy.linalg.norm(errvec)
        keep = []
        for i in range(nd):
            ei = numpy.asarray(self.get_err_vec(i))
            if numpy.linalg.norm(ei) <= radius:
                keep.append(i)
        return keep

    def _coherence_prune(self, keep):
        '''Among the surviving residuals, evict the larger-norm (staler)
        member of any pair whose normalized overlap is >= 0.98.  The
        min(nd, 3) most recent pairs are always retained, regardless of the
        trust window and coherence rules.'''
        nd = self.get_num_vec()
        protected = set(self._bookkeep[-min(nd, self._RETAIN_FLOOR):])
        survivors = set(keep) | protected
        # chronological order (oldest first) as recorded in _bookkeep
        keep = [i for i in self._bookkeep if i in survivors]
        vecs = [numpy.asarray(self.get_err_vec(i)) for i in keep]
        norms = [numpy.linalg.norm(v) for v in vecs]
        units = [v/n if n != 0 else v for v, n in zip(vecs, norms)]
        evicted = set()
        for a in range(len(keep)):
            if keep[a] in evicted:
                continue
            for b in range(a+1, len(keep)):
                if keep[a] in evicted:
                    break
                if keep[b] in evicted:
                    continue
                if abs(numpy.vdot(units[a], units[b])) >= self._COHERENCE_THRESHOLD:
                    stale = a if norms[a] >= norms[b] else b
                    if keep[stale] not in protected:
                        evicted.add(keep[stale])
        return [i for i in keep if i not in evicted]

    def _compact(self, keep):
        '''Re-store the surviving (Fock, residual) pairs in slots 0..n-1 and
        rebuild _bookkeep, _head and the cached inner-product matrix _H from
        the unmodified stored vectors.'''
        errvecs = [numpy.asarray(self.get_err_vec(i)).copy() for i in keep]
        vecs = [numpy.asarray(self.get_vec(i)).copy() for i in keep]
        for i, (errvec, vec) in enumerate(zip(errvecs, vecs)):
            self._store('e%d' % i, errvec)
            self._store('x%d' % i, vec)
        self._bookkeep = list(range(len(keep)))
        self._head = len(keep)
        if not keep:
            self._H = None
            return
        self._H = numpy.zeros((self.space+1, self.space+1), errvecs[0].dtype)
        self._H[0,1:] = self._H[1:,0] = 1
        for i, xi in enumerate(errvecs):
            for j in range(i+1):
                value = numpy.vdot(xi, errvecs[j])
                self._H[i+1,j+1] = value
                self._H[j+1,i+1] = value.conjugate()

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        if self.rollback == 0 and self.get_num_vec() > 0:
            keep = self._coherence_prune(self._trust_window(errvec))
            if len(keep) < self.get_num_vec():
                self._compact(keep)
        f_prev = kwargs.get('f_prev', None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        else:
            xnew = lib.diis.DIIS.update(self, f*(1-self.damp) + f_prev*self.damp, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec_orig(s, d, f):
    '''error vector = SDF - FDS'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(lib.dot, (s,d,f))
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(lib.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = numpy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orig(s, d[0], f[0]).ravel(),
            get_err_vec_orig(s, d[1], f[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec_orth(s, d, f, Corth):
    '''error vector in orthonormal basis = C.T.conj() (SDF - FDS) C'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(lib.dot, (Corth.conj().T, s, d, f, Corth))
        # Symmetry information to reduce numerical error in DIIS (issue #1524)
        if hasattr(Corth, 'orbsym'):
            orbsym = Corth.orbsym
            sdf[orbsym[:,None] != orbsym] = 0
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(lib.dot, (Corth[i].conj().T, s[i], d[i], f[i], Corth[i]))
            errvec.append((sdf.conj().T - sdf))
        if hasattr(Corth, 'orbsym'):
            orbsym = Corth.orbsym
            sym_forbid = orbsym[:,None] != orbsym
            for sdf in errvec:
                sdf[sym_forbid] = 0
        elif hasattr(Corth[0], 'orbsym'):
            for i, sdf in enumerate(errvec):
                orbsym = Corth[i].orbsym
                sdf[orbsym[:,None] != orbsym] = 0
        errvec = numpy.hstack([x.ravel() for x in errvec])

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orth(s, d[0], f[0], Corth).ravel(),
            get_err_vec_orth(s, d[1], f[1], Corth).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

class EDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    def update(self, s, d, f, mf, h1e, vhf, *args, **kwargs):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = numpy.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es, ds, fs)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        return fock

def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return numpy.einsum('i,i', c, es) - numpy.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIIS(lib.diis.DIIS):
    '''
    Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922
    '''
    def update(self, s, d, f, mf, h1e, vhf, *args, **kwargs):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        fun, c = adiis_minimize(ds, fs, self._head)
        if self.verbose >= logger.DEBUG1:
            etot = mf.energy_elec(d, h1e, vhf)[0] + fun
            logger.debug1(self, 'E %s  diis-c %s ', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

def adiis_minimize(ds, fs, idnewest):
    nx = ds.shape[0]
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (numpy.einsum('i,i', c, dd_fn) * 2 +
                numpy.einsum('i,ij,j', c, df, c))

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc+= numpy.einsum('j,kj->k', c, df)
        fc+= numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()
