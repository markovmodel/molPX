# Copyright (c) 2017, Fabian Paul, Computational Molecular Biology Group, 
# Freie Universitaet Berlin and Max Planck Institute of Colloids and Interfaces
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""PCCA for the TICA space and the splash projection.

.. moduleauthor:: Fabian Paul <fab@zedat.fu-berlin.de>

"""

from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.spatial
import scipy.optimize
import warnings


def _othonormalize(vertices):
    # pick vertex that is closest to zero as the origin
    i0 = np.argmin(np.linalg.norm(vertices, axis=1))
    v0 = vertices[i0, :]

    # for the remaining vertices, subtract v0 and othonormalize
    a = np.concatenate((vertices[0:i0, :], vertices[i0+1:, :])) - v0[np.newaxis, :]
    q, _ = np.linalg.qr(a.T)
    return v0, q.T


def _projector(vectors, min_1=True):
    dim = vectors.shape[1]
    X = np.zeros((dim, dim))
    for i in range(vectors.shape[0]):
        X += np.outer(vectors[i, :], vectors[i, :])
    if min_1:
        X -= np.eye(dim)
    return X


def _order_from_rank(rank):
    # TODO: start from the absolute maximum?
    N = rank.shape[0]
    assert rank.shape[1]==N
    rank = rank.copy()
    order = np.zeros(N, dtype=int)
    for i in range(N):
        j = np.argmax(rank[:, i])
        order[i] = j
        rank[j, :] = -np.inf

    assert len(np.unique(order)) == len(order)
    return order


def _vertex_order(vertices):
    # vertex #0 is the one that is closest to the origin of the coordiante system
    i0 = np.argmin(np.linalg.norm(vertices, axis=1))
    v0 = vertices[i0, :]
    others = np.concatenate((np.arange(0, i0), np.arange(i0+1, vertices.shape[0])))
    other_vertices = vertices[others, :]

    # order the rest by closest canonical (Cartesian) axis
    N = other_vertices.shape[0]

    rank = np.zeros((N, N))
    for i in range(N):
        for j in range(N): 
            rank[i, j] = abs(other_vertices[i, j]) #-(sum(vertices[i, :]**2)-vertices[i, j]**2)

    order = _order_from_rank(rank)

    return np.concatenate(([i0], others[order]))


def core_assignments(input_, vertices, f=0.5):
    r"""Assign every row of input_ to that vertex to which is has the highest membership.

        parameters
        ----------
        input_ : list of np.ndarray((n_time_steps, n_dims))
            the input data
        vertices : np.ndarray((n_dims+1, n_dims))
            coordiantes of the vertices
        f : float, default = 0.5
            Cut-off for the PCCA membership. Frames with a membership lower than f are left unassigned.
            f typically takes a value between 0 and 1.

        returns
        -------
        dtrajs : list of np.ndarray(n_time_steps, dtype=int)
            For every assigned frame, the index of the vertex with highest membership.
            For frames that are unassigned, the value -1.
    """
    if not isinstance(input_, (list, tuple)):
        input_ = [ input_ ]

    M = np.vstack((vertices.T, np.ones(vertices.shape[0])))
    lu_and_piv = sp.linalg.lu_factor(M)

    dtrajs = [ np.zeros(traj.shape[0], dtype=int)-1 for traj in input_ ]

    for traj, dtraj in zip(input_, dtrajs):
        for i, x in enumerate(traj):
            #l = np.linalg.solve(M, np.concatenate((x, [1])))
            l = sp.linalg.lu_solve(lu_and_piv, np.concatenate((x, [1]))) # these are the memberships
            # see https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
            j = np.argmax(l)
            if l[j] > f:
                dtraj[i] = j
            ## old (wrong) algo.
            #dist = np.linalg.norm(vertices-x, axis=1)
            #order = np.argsort(dist)
            #if dist[order[0]]*f < dist[order[1]]:
            #    dtraj[i] = order[0]

    return dtrajs


def find_vertices_inner_simplex(input_, return_means=False, f_centers=float('-inf')):
    r'''Find vertices of the "inner simplex". This is the old PCCA algorithm from Weber & Galliat 2002.

    parameters
    ----------
    input_ : list of np.ndarray((n_time_steps, n_dims))
        The input data. Chunking is not yet implemented but possible in principle.

    returns
    -------
    vertices : np.ndarray((n_dims+1, n_dims))
        Coordinates of the n_dims+1 vertices.
    '''
    # inner simplex algorithm (a.k.a. old PCCA, Weber & Galliat 2002) for large number of data points
    if not isinstance(input_, (list, tuple)):
        input_ = [ input_ ]

    dim = input_[0].shape[1]

    # First find the two most distant vertices. We use the following heuristic:
    # The two points with the largest separation in a simplex should be among those that 
    # lie on the (axes-parallel, Cartesian) bounding box of the points. E.g. in 2-D 
    # a simplex is a triangle. Even if the triangle has an obtuse angle, two of its
    # vertices will lie on the bounding box. In 3-D two (or more) vertices of a 
    # tetrahedron will lie on the bounding box while up to two vertices will dangle 
    # in midair, etc.
    maxima = np.zeros(dim) - np.inf
    minima = np.zeros(dim) + np.inf
    min_pts = np.empty((dim, dim))
    max_pts = np.empty((dim, dim))

    # first pass
    #it = input_.iterator()
    #with it:
    #    for chunk in it:
    print('pass 1')
    for traj in input_:
        for x in traj:
            wh = x < minima
            minima[wh] = x[wh]
            min_pts[wh, :] = x
            wh = x > maxima
            maxima[wh] = x[wh]
            max_pts[wh, :] = x

    # Among all the points on the bounding box, pick the ones with largest separation.
    ext_pts = np.concatenate((max_pts, min_pts))
    d = sp.spatial.distance.squareform(sp.spatial.distance.pdist(ext_pts))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    vertices = np.empty((2, dim))
    vertices[0, :] = ext_pts[i]
    vertices[1, :] = ext_pts[j]

    # further passes, follow the algorithm form Weber & Galliat
    for k in range(2, dim+1): # find dim+1 vertices
        print('pass', k)
        v0, w = _othonormalize(vertices)
        P = _projector(w, min_1=True)
        candidate = vertices[-1, :]
        d = 0.0
        #it = input_.iterator()
        #with it:
        #    for chunk in it:
        for traj in input_:
            for frame in traj:
                d_candidate = np.linalg.norm(P.dot(frame-v0))
                if d_candidate > d:
                    candidate = frame
                    d = d_candidate
        vertices = np.vstack((vertices, candidate))

    order = _vertex_order(vertices)

    if return_means:
        centers = np.zeros((dim+1, dim))
        counts = np.zeros(dim+1, dtype=int)
        dtrajs = core_assignments(input_, vertices, f=f_centers)
        for traj, dtraj in zip(input_, dtrajs):
            for x, d in zip(traj, dtraj):
                counts[d] += 1
                centers[d, :] += x

        centers /= counts[:, np.newaxis]

        return vertices[order, :], centers[order, :]
    else:
        return vertices[order, :]


def mds_projection(vertices, center=None, n_dim_target=2):
    r"""Compute a projection matrix that represents the MSD embedding of the vertices into n_dim dimensions.

        parameters
        ----------
        vertices : np.ndarray((n_dims_source+1, n_dims_source))
            coordiantes of the vertices
        center : int, optional, default = None
            index of the vertex to put at the coordinate origin in the target space
            By default, the vertex closest ot the coodinate origin in the source space is selected.
        n_dim_target : int, default = 2
            dimension of the target space

        returns
        -------
        (P, o)
        P : np.ndarray((n_dims_source, n_dim_target))
            the projection matrix
        o : np.ndarray((n_dims_source))
            the shift vector

        To apply the projection to your data `d`, compute `(d-o).dot(P)`
    """
    # todo: this can be optimized...
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_dim_target, metric=True, dissimilarity='euclidean')
    vertices_low_D = mds.fit_transform(vertices)
    if center is None:
        # pick the one that is closest to the origin in the projection
        center = np.argmin(np.linalg.norm(vertices_low_D, axis=1))
    u = vertices_low_D[center, :]
    L = np.concatenate((vertices_low_D[0:center, :], vertices_low_D[center+1:, :])) - u

    o = vertices[center, :]
    W = np.concatenate((vertices[0:center, :], vertices[center+1:, :])) - o
    return np.linalg.inv(W).dot(L), o


def thomson_problem(dim, n_elec, selftest=True, max_iter=100, return_energy=False):
    r"""Searches a local optimum of the Thomson problem with n_elec electrons in dim dimensions."""
    assert dim>=3

    x0 = np.zeros((n_elec, dim))
    # inital guess with v. Neumann method
    for i in range(n_elec):
        while True:
            r = (np.random.random_sample(dim)*2.0) - 1.0
            n = np.linalg.norm(r)
            if n<1 and n>0:
                x0[i, :] = r/n
                break # while

    X0 = x0.reshape(-1)

    def func(X):
        x = X.reshape((n_elec, dim))
        d = sp.spatial.distance.squareform(sp.spatial.distance.pdist(x))
        np.fill_diagonal(d, np.inf)
        value = 0.5*np.power(d, -(dim-2)).sum()
        return value

    def fprime(X):
        x = X.reshape((n_elec, dim))
        d = sp.spatial.distance.squareform(sp.spatial.distance.pdist(x))
        np.fill_diagonal(d, np.inf)
        m = np.power(d, -dim)
        v = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        return (2-dim)* (m[:, :, np.newaxis]*v).sum(axis=1).reshape(-1)

    def f_eqcons(X):
        x = X.reshape((n_elec, dim))
        return np.linalg.norm(x, axis=1)**2.0 - 1.0

    def fprime_eqcons(X):
        x = X.reshape((n_elec, dim))
        f = np.zeros((n_elec, n_elec*dim))
        for i in range(n_elec):
            indices = i*dim + np.arange(dim) # last=fast index
            f[i, indices] = 2.0 * x[i, :]
        return f

    #print 'initial energy', func(X0)

    if selftest:
        direction = np.random.rand(len(X0)) * 1.0E-5
        f1, grad1 = func(X0), fprime(X0)
        f2 = func(X0 + direction)
        df = np.dot(grad1, direction)
        # in general we would have to use |a-b|/max(|a|,|b|) but since mostly |df|>|f1-f2| we can use error/|df|
        err = np.abs((f2 - f1 - df) / (df))
        assert err < 0.01
        #print 'gradient error', err

        direction = np.random.rand(len(X0)) * 1.0E-5
        f1, grad1 = f_eqcons(X0), fprime_eqcons(X0)
        f2 = f_eqcons(X0 + direction)
        df = np.dot(grad1, direction)
        err = np.abs((f2 - f1 - df) / (df))
        assert  np.all(err < 0.01)
        #print 'constraint gradient error', err

    out, fx, its, imode, smode = sp.optimize.fmin_slsqp(func=func, x0=X0, f_eqcons=f_eqcons,
        fprime=fprime, fprime_eqcons=fprime_eqcons, iter=max_iter, iprint=0, full_output=True)

    if imode!=0:
        warnings.warn(smode)

    x = out.reshape((n_elec, dim))

    n = np.linalg.norm(x, axis=1)
    x /= n[:, np.newaxis]

    # TODO: check results in 3-D against http://www-wales.ch.cam.ac.uk/~wales/CCD/Thomson/table.html

    if return_energy:
        return x, func(x.reshape(-1))
    else:
        return x


def splash_corner_projection(vertices, center=0, n_dim_target=2, max_iter=100):
    r"""Compute a projection matrix that represents an even embedding of the vertices into a target space with dimension n_dim.

        parameters
        ----------
        vertices : np.ndarray((n_dims_source+1, n_dims_source))
            coordiantes of the vertices
        center : int, optional, default = None
            index of the vertex to put at the coordinate origin in the target space
            By default, the vertex closest ot the coodinate origin in the source space is selected.
        n_dim_target : int, default = 2
            dimension of the target space
        max_iter : int, default = 100
            For n_dim_target >= 3, the projection is found by searching for 
            a local optimum of the Thomson problem. max_iter limits the
            number of iterations of the minimizer.

        returns
        -------
        (P, o)
        P : np.ndarray((n_dims_source, n_dim_target))
            the projection matrix
        o : np.ndarray((n_dims_source))
            the shift vector

        To apply the projection to your data `d`, compute `(d-o).dot(P)`
    """
    N = vertices.shape[0] - 1    
    o = vertices[center, :]
    W = np.concatenate((vertices[0:center, :], vertices[center+1:, :])) - o
    if n_dim_target == 2:
        L = np.empty((N, 2))
        for i in range(N):
            L[i, 0] = np.sin(2.0*np.pi*i/float(N))
            L[i, 1] = np.cos(2.0*np.pi*i/float(N))
    elif n_dim>2:
        L = thomson_problem(n_dim_target, N, max_iter=max_iter)
    else:
        raise Exception('n_dim must be an integer > 1')
    return np.linalg.inv(W).dot(L), o


def milestoning_count_matrix(dtrajs, lag=1, n_states=None, return_mass_matrix=False, return_scrapped=False):
    r"""Computed the Milestoning covariance matrix and mass matrix as described in [1]_

        parameters
        ----------
        dtrajs : list of np.ndarray((n_time_steps, ), dtype=int)
            Core label trajectories. Frames that are not assigned to any
            core, should take a strictly neagtive value, e. g. -1.
        lag : int, default = 1
            the lag time
        n_states : int, optional
            determines the shape of the retuned matrices. If None, use
            the maximum core index in dtrajs + 1.
        return_mass_matrix : bool, optional, default = False
            whether to compute and return the mass matrix (a. k. a. the
            instantaneous covariance matrix of the core committors)
        return_scrapped : bool, optional, default = False
            whether to return the number of frames that were scrapped
            in the computation of the matrices. Frames at the beginning
            or the end of a trajectory that are not assigned to any
            core, are scrapped.

        returns
        -------
        Depending on the value of return_mass_matrix, return_scrapped
        *  c
        *  (c, m)
        *  (c, n)
        *  (c, m, n)

        c : np.ndarray((n_states, n_states), dtype=int)
            the time-lagged covartiance matrix of the core committors
        m : np.ndarray((n_states, n_states), dtype=int)
            the mass matrix
        n : number of frames that weren't used in estimating the matrices

    References
    ----------
    .. [1] Ch. Schuette & M. Sarich, Eur. Phys. J. Spec. Top. 244, 245 (2005)
    """
    import warnings
    warnings.warn('Milestoning code is not thoroughly tested, to be safe, please use pyemma.')
    if n_states is None:
        n_states = max([np.max(d) for d in dtrajs]) + 1
        assert n_states >= 1

    c = np.zeros((n_states, n_states), dtype=int)
    if return_mass_matrix:
        m = np.zeros((n_states, n_states), dtype=int)
    n_scrapped = 0

    for d in dtrajs:
        if np.any(d>=0):
            # cut off transition state pieces at the end and beginning
            first_idx = next(i for i, s in enumerate(d) if s>=0)
            #print(first_idx)
            last_idx = len(d) - next(i for i, s in enumerate(d[::-1]) if s>=0)
            #print(last_idx)
            if last_idx - first_idx <= lag:
                n_scrapped += len(d)
                continue
            n_scrapped += first_idx
            #print(first_idx)
            n_scrapped += len(d)-last_idx
            #print( len(d)-last_idx)
            d = d[first_idx:last_idx]
            # generate past and future
            past = np.zeros(len(d), dtype=int)
            last_s = d[0]
            for i, s in enumerate(d):
                if s>=0:
                    last_s = s
                past[i] = last_s
            future = np.zeros(len(d), dtype=int)
            next_s = d[-1]
            for i, s in zip(np.arange(len(d))[::-1], d[::-1]):
                if s>=0:
                    next_s = s
                future[i] = next_s
            # fill count matrix
            for p, f in zip(past[0:-lag], future[lag:]):
                c[p, f] += 1
            if return_mass_matrix:
                for p, f in zip(past[0:-lag], future[0:-lag]):
                    m[p, f] += 1
        else:
            n_scrapped += len(d)

    if return_mass_matrix and return_scrapped:
        return c, m, n_scrapped
    elif return_mass_matrix:
        return c, m
    elif return_scrapped:
        return c, n_scrapped
    else:
        return c


## workflow for visualization:
# vertices = find_vertices_inner_simplex(data)
# P, o = corner_projection(vertices, n_dim=2)
# P, o = mds_projection(vertices, n_dim=2) # alternative
# low_dimensional_data = (data-o).dot(P)
# plt.hist2d(low_dimensional_data[:, 0], low_dimensional_data[:, 1])

## workflow for clustering: replace n_dim=2 with somewhat higher number

#minimal energies of the Thomson problem form website above
#e =  np.array([32.7169494, 40.5964505, 49.165253, 58.8532306,
# 69.3063633, 80.6702441, 92.9116553, 106.0504048, 120.0844674,
# 135.0894675, 150.8815683, 167.6416223, 185.2875361, 203.9301906,
# 223.347074, 243.8127602, 265.1333263, 287.302615, 310.4915423,
# 334.6344399, 359.6039459, 385.530838, 412.2612746, 440.2040574,
# 468.9048532, 498.5698724, 529.1224083, 560.6188877, 593.0385035,
# 626.389009, 660.6752788, 695.9167443, 732.0781075, 769.1908464,
# 807.174263, 846.188401, 886.1671136, 927.0592706, 968.7134553,
# 1011.5571826, 1055.1823147, 1099.8192903, 1145.4189643, 1191.9222904,
# 1239.3614747, 1287.7727207, 1337.0949452, 1387.3832292, 1438.6182506,
# 1490.7733352, 1543.8304009, 1597.9418301, 1652.9094098, 1708.8796815,
# 1765.8025779, 1823.6679602, 1882.4415253, 1942.1227004, 2002.8747017,
# 2064.5334832, 2127.1009015, 2190.6499064, 2255.0011909, 2320.6338837,
# 2387.0729818, 2454.369689, 2522.6748718, 2591.8501523, 2662.0464745,
# 2733.2483574, 2805.3558759, 2878.5228296, 2952.5696752, 3027.5284889,
# 3103.4651244, 3180.3614429, 3258.2116057, 3337.00075, 3416.7201967,
# 3497.4390186, 3579.0912227, 3661.7136993, 3745.2916362, 3829.8443384,
# 3915.3092696, 4001.7716755, 4089.15401, 4177.5335996, 4266.8224641,
# 4357.1391631])
