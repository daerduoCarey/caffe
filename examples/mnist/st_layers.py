import numpy as np

# This file implement the forwarding and backwarding of spatial transformer layer

def bilinear_transform_forward(X, x, y):
	
	H, W = X.shape
	res = 0

	m, n, w = np.floor(x), np.floor(y), 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		res += w * X[m, n]

	m, n, w = np.floor(x) + 1, np.floor(y), 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		res += w * X[m, n]

	m, n, w = np.floor(x), np.floor(y) + 1, 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		res += w * X[m, n]

	m, n, w = np.floor(x) + 1, np.floor(y) + 1, 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		res += w * X[m, n]

	return res


def st_forward(Theta, U, HH, WW):

	'''
		Input:	Theta:	transformation parameter of dimension N * 2 * 3, only allowing affine transformation
				U:		the input maps of dimension N * C * H * W, all channel will be applied with the same transformation
				HH, WW:	the dimension of output 

		Output:	V:		the output maps of dimension N * C * HH * WW
				cache:	(theta, U), information for backward propogation
	'''

	N, C, H, W = U.shape

	V = np.zeros((N, C, HH, WW))

	px = np.zeros((N, HH, WW))
	py = np.zeros((N, HH, WW))

	for item in xrange(N):

		for i in xrange(HH):
			for j in xrange(WW):
				vec = np.array([i*1.0/HH*2-1, j*1.0/WW*2-1, 1])
				sx, sy = np.dot(Theta[item], vec)
				px[item, i, j], py[item, i, j] = (sx+1)/2 * H, (sy+1)/2 * W

		for c in xrange(C):
			for i in xrange(HH):
				for j in xrange(WW):
					ppx, ppy = px[item, i, j], py[item, i, j]
					V[item, c, i, j] = bilinear_transform_forward(U[item, c], ppx, ppy)

	cache = (Theta.copy(), U.copy(), HH, WW, px.copy(), py.copy())
	return V, cache
		

def bilinear_transform_backward(X, x, y, dV, dU):
	
	H, W = X.shape
	delta_dpx = delta_dpy = 0

	m, n, w = np.floor(x), np.floor(y), 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		
		dU[m, n] += w * dV
		
		if abs(x - m) >= 1:
			delta_dpx += 0
		elif m >= x:
			delta_dpx += max(0, 1 - abs(y - n)) * X[m, n]
		else:
			delta_dpx += -max(0, 1 - abs(y - n)) * X[m, n]

		if abs(y - n) >= 1:
			delta_dpy += 0
		elif n >= y:
			delta_dpy += max(0, 1 - abs(x - m)) * X[m, n]
		else:
			delta_dpy += -max(0, 1 - abs(x - m)) * X[m, n]


	m, n, w = np.floor(x) + 1, np.floor(y), 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		
		dU[m, n] += w * dV
		
		if abs(x - m) >= 1:
			delta_dpx += 0
		elif m >= x:
			delta_dpx += max(0, 1 - abs(y - n)) * X[m, n]
		else:
			delta_dpx += -max(0, 1 - abs(y - n)) * X[m, n]

		if abs(y - n) >= 1:
			delta_dpy += 0
		elif n >= y:
			delta_dpy += max(0, 1 - abs(x - m)) * X[m, n]
		else:
			delta_dpy += -max(0, 1 - abs(x - m)) * X[m, n]

	m, n, w = np.floor(x), np.floor(y) + 1, 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		
		dU[m, n] += w * dV
		
		if abs(x - m) >= 1:
			delta_dpx += 0
		elif m >= x:
			delta_dpx += max(0, 1 - abs(y - n)) * X[m, n]
		else:
			delta_dpx += -max(0, 1 - abs(y - n)) * X[m, n]

		if abs(y - n) >= 1:
			delta_dpy += 0
		elif n >= y:
			delta_dpy += max(0, 1 - abs(x - m)) * X[m, n]
		else:
			delta_dpy += -max(0, 1 - abs(x - m)) * X[m, n]

	m, n, w = np.floor(x) + 1, np.floor(y) + 1, 0
	if 0 <= m < H and 0 <= n < W:
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n))
		
		dU[m, n] += w * dV
		
		if abs(x - m) >= 1:
			delta_dpx += 0
		elif m >= x:
			delta_dpx += max(0, 1 - abs(y - n)) * X[m, n]
		else:
			delta_dpx += -max(0, 1 - abs(y - n)) * X[m, n]

		if abs(y - n) >= 1:
			delta_dpy += 0
		elif n >= y:
			delta_dpy += max(0, 1 - abs(x - m)) * X[m, n]
		else:
			delta_dpy += -max(0, 1 - abs(x - m)) * X[m, n]

	return delta_dpx * dV, delta_dpy * dV


def st_backward(dV, cache):

	Theta, U, HH, WW, px, py = cache

	N, C, H, W = U.shape

	dU = np.zeros((N, C, HH, WW))
	dpx = np.zeros((N, HH, WW))
	dpy = np.zeros((N, HH, WW))
	dTheta = np.zeros((N, 2, 3))

	for item in xrange(N):
		for c in xrange(C):
			for i in xrange(HH):
				for j in xrange(WW):
					ppx, ppy = px[item, i, j], py[item, i, j]
					delta_dpx, delta_dpy = bilinear_transform_backward(U[item, c], ppx, ppy, dV[item, c, i, j], dU[item, c])
					dpx[item, i, j] += delta_dpx
					dpy[item, i, j] += delta_dpy

		for i in xrange(HH):
			for j in xrange(WW):
				dTheta[item, 0, 0] += H/2 * dpx[item, i, j] * (i*1.0/HH*2-1)
				dTheta[item, 0, 1] += H/2 * dpx[item, i, j] * (j*1.0/WW*2-1)
				dTheta[item, 0, 2] += H/2 * dpx[item, i, j]
				dTheta[item, 1, 0] += W/2 * dpy[item, i, j] * (i*1.0/HH*2-1)
				dTheta[item, 1, 1] += W/2 * dpy[item, i, j] * (j*1.0/WW*2-1)
				dTheta[item, 1, 2] += W/2 * dpy[item, i, j]


	return dU, dTheta

