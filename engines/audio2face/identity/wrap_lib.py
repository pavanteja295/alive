#!/usr/bin/env python3
"""Geometry primitives for the Beltrami wrap. Each one is here because a simpler
alternative was tried first and measurably failed -- the comments say which."""
import numpy as np
import scipy.sparse as sp


def tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vert_normals(V, F):
    f = tri_normals(V, F); o = np.zeros_like(V)
    for k in range(3):
        np.add.at(o, F[:, k], f)
    return o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)


def dihedral(V, F):
    """Mean angle between adjacent face normals, in degrees.

    This is the smoothness metric, and it has to be this one. `|L @ V|` was used first --
    a ONE-RING quantity -- and it read 1.00x the archetype while the render was visibly
    faceted, because a FLAME facet spans about five MetaHuman vertices and a one-ring
    operator cannot see it. Dihedral angle is what a shaded render actually shows.
    """
    n = tri_normals(V, F)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((e[:, 1], e[:, 0])); e, fid = e[o], fid[o]
    same = np.all(e[1:] == e[:-1], axis=1)
    i, j = fid[:-1][same], fid[1:][same]
    return float(np.degrees(np.arccos(np.clip((n[i] * n[j]).sum(1), -1, 1))).mean())


def umbrella(V, F):
    """Uniform Laplacian, used only as the ICP regulariser (not as the basis)."""
    n = len(V)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.vstack([e, e[:, ::-1]]), axis=0)
    deg = np.bincount(e[:, 0], minlength=n).astype(float)
    A = sp.coo_matrix((-1.0 / np.maximum(deg[e[:, 0]], 1), (e[:, 0], e[:, 1])), shape=(n, n))
    return (sp.identity(n, format="csr") + A.tocsr()).tocsr()


def cot_laplacian(V, F):
    """Cotangent stiffness + barycentric mass, for the generalised problem L u = lambda M u.

    The UNWEIGHTED graph Laplacian was tried first and made things worse -- 1.18x the
    archetype's dihedral at K=100, degrading to 1.59x at K=900. Its eigenvectors are
    smooth with respect to CONNECTIVITY, and MetaHuman's mesh is dense at the eyes and
    lips and sparse over the scalp, so its low modes oscillate wherever the mesh is dense.
    The cotangent operator's modes are geometric and tessellation-invariant, which is the
    property that makes the truncation a real band limit.

    Cotangents are clamped at 0 on obtuse triangles; otherwise L is indefinite and the
    eigensolve is meaningless.
    """
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    e0, e1, e2 = V[i2] - V[i1], V[i0] - V[i2], V[i1] - V[i0]
    a2 = np.maximum(np.linalg.norm(np.cross(e1, -e2), axis=1), 1e-12)
    c0, c1, c2 = (np.maximum(-(x * y).sum(1) / a2, 0.0)
                  for x, y in ((e1, e2), (e2, e0), (e0, e1)))
    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    Wv = np.concatenate([c0, c0, c1, c1, c2, c2]) * 0.5
    n = len(V)
    W = sp.coo_matrix((Wv, (I, J)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    m = np.zeros(n)
    for k in range(3):
        np.add.at(m, F[:, k], a2 / 6.0)
    return L.tocsc(), sp.diags(np.maximum(m, 1e-10)).tocsc()


def loop(V, F):
    """One Loop subdivision step.

    MIDPOINT subdivision was used first. It adds vertices exactly on the existing flat
    triangles and therefore adds no smoothness at all: the target stayed faceted at
    7.42 deg and the wrap faithfully copied FLAME's tessellation. Loop moves the original
    vertices too and converges to a smooth limit surface -- 3.27 deg, smoother than the
    archetype itself.
    """
    nV = len(V)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    ue = np.unique(e, axis=0)
    eid = {tuple(k): nV + i for i, k in enumerate(ue)}
    opp = [[] for _ in range(len(ue))]
    for a, b, c in F:
        for x, y, zc in ((a, b, c), (b, c, a), (c, a, b)):
            opp[eid[(min(x, y), max(x, y))] - nV].append(zc)
    En = np.empty((len(ue), 3))
    for k, (a, b) in enumerate(ue):
        o = opp[k]
        En[k] = ((3 / 8) * (V[a] + V[b]) + (1 / 8) * (V[o[0]] + V[o[1]])
                 if len(o) == 2 else 0.5 * (V[a] + V[b]))
    adj = [[] for _ in range(nV)]
    for a, b in ue:
        adj[a].append(b); adj[b].append(a)
    Vn = V.copy()
    for i, nb in enumerate(adj):
        n = len(nb)
        if n < 3:
            continue
        beta = (5 / 8 - (3 / 8 + 0.25 * np.cos(2 * np.pi / n)) ** 2) / n
        Vn[i] = (1 - n * beta) * V[i] + beta * V[nb].sum(0)
    g = lambda a, b: eid[(min(a, b), max(a, b))]
    Fo = []
    for a, b, c in F:
        ab, bc, ca = g(a, b), g(b, c), g(c, a)
        Fo += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.vstack([Vn, En]), np.array(Fo, np.int64)


def similarity_icp(Vsrc, Vtgt, tree, iters=20):
    """Place the archetype on the FLAME head. Rotation, translation and scale, no
    reflection -- both meshes are X=lateral, Y=up, Z=forward, so a reflection here would
    be a bug, not a fit."""
    s = (np.linalg.norm(Vtgt - Vtgt.mean(0), axis=1).mean()
         / np.linalg.norm(Vsrc - Vsrc.mean(0), axis=1).mean())
    R, t = np.eye(3), Vtgt.mean(0) - s * Vsrc.mean(0)
    for _ in range(iters):
        cur = s * (Vsrc @ R) + t
        _, i = tree.query(cur)
        P, Q = cur, Vtgt[i]
        ca, cb = P.mean(0), Q.mean(0); X, Y = P - ca, Q - cb
        U, S, Vt = np.linalg.svd(X.T @ Y); dR = U @ Vt
        if np.linalg.det(dR) < 0:
            U[:, -1] *= -1; dR = U @ Vt
        ds = S.sum() / (X ** 2).sum()
        s, R, t = s * ds, R @ dR, t @ dR * ds + (cb - ds * (ca @ dR))
    return s * (Vsrc @ R) + t
