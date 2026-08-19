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
        lib.diis.DIIS.__init__(self, mf, None)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.damp = 0

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f, self.Corth)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
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


class HystereticDIIS(lib.diis.DIIS):
    '''Energy--commutator controller for EDIIS, ADIIS, and CDIIS.

    All three extrapolators are updated on every call so that switching does
    not start from an empty subspace.  The controller only chooses which of
    the three extrapolated Fock matrices is returned.
    '''
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.damp = 0
        self.conv_tol = 1e-9
        self.conv_tol_grad = numpy.sqrt(self.conv_tol)

        # Keep the traditional CDIIS restart file at the user-supplied path.
        self._cdiis = CDIIS(mf, filename, Corth)
        self._adiis = ADIIS(mf, _subspace_filename(filename, 'adiis'))
        self._ediis = EDIIS(mf, _subspace_filename(filename, 'ediis'))
        self._phase = 'ediis'
        self._pending_phase = None
        self._pending_count = 0
        self._last_energy = None
        self._last_residual = None

    def update(self, s, d, f, mf, h1e, vhf, *args, **kwargs):
        self._sync_extrapolators()

        energy = mf.energy_elec(d, h1e, vhf)[0]
        errvec = get_err_vec(s, d, f, self.Corth)
        residual = numpy.linalg.norm(errvec) / numpy.sqrt(errvec.size)
        if self._last_residual is None or self._last_residual == 0:
            q = numpy.inf
        else:
            q = residual / self._last_residual

        energy_noise = max(self.conv_tol, 100 * numpy.finfo(float).eps)
        if self._last_energy is None:
            energy_change = 0.
            requested = 'ediis'
        else:
            energy_change = energy - self._last_energy
            if energy_change > energy_noise or q > 1.1:
                requested = 'ediis'
            elif residual <= 10 * self.conv_tol_grad or q <= .7:
                requested = 'cdiis'
            else:
                requested = 'adiis'

        self._select_phase(requested, energy_change, energy_noise, q)

        # Keep independent, warm histories.  No extrapolator's minimization or
        # subspace policy is modified by the controller.
        focks = {
            'ediis': self._ediis.update(s, d, f, mf, h1e, vhf,
                                        *args, **kwargs),
            'adiis': self._adiis.update(s, d, f, mf, h1e, vhf,
                                        *args, **kwargs),
            'cdiis': self._cdiis.update(s, d, f, mf, h1e, vhf,
                                        *args, **kwargs),
        }
        logger.debug1(self, 'hybrid-diis phase=%s dE=%g r=%g q=%g',
                      self._phase, energy_change, residual, q)
        self._last_energy = energy
        self._last_residual = residual
        return focks[self._phase]

    def _select_phase(self, requested, energy_change, energy_noise, q):
        # Move forward as soon as contraction supports it.  Falling back to a
        # more conservative method needs two consecutive samples, and CDIIS
        # uses the wider q=0.8 boundary to avoid phase chatter.
        rank = {'ediis': 0, 'adiis': 1, 'cdiis': 2}
        if rank[requested] >= rank[self._phase]:
            self._phase = requested
            self._pending_phase = None
            self._pending_count = 0
            return

        if (self._phase == 'cdiis' and requested == 'adiis' and
            energy_change <= energy_noise and q <= .8):
            self._pending_phase = None
            self._pending_count = 0
            return

        if requested == self._pending_phase:
            self._pending_count += 1
        else:
            self._pending_phase = requested
            self._pending_count = 1

        if self._pending_count >= 2:
            self._phase = requested
            self._pending_phase = None
            self._pending_count = 0

    def _sync_extrapolators(self):
        for obj in (self._ediis, self._adiis, self._cdiis):
            obj.space = self.space
            obj.verbose = self.verbose
            obj.stdout = self.stdout
        self._cdiis.rollback = self.rollback
        self._cdiis.damp = self.damp
        self._cdiis.Corth = self.Corth

    def get_num_vec(self):
        return self._cdiis.get_num_vec()


def _subspace_filename(filename, suffix):
    if filename is None:
        return None
    return '%s.%s' % (filename, suffix)

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
