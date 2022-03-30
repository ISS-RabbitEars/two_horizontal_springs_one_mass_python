import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m, k1, k2, xeq1, xeq2 = p
	x, v = ic

	print(ti)

	return [v, A.subs({M:m, K1:k1, K2:k2, Xeq1:xeq1, Xeq2:xeq2, X:x, Xdot:v})]


M, K1, K2, Xeq1, Xeq2, t = sp.symbols('M K1 K2 Xeq1 Xeq2 t')
X = dynamicsymbols('X')

Xdot = X.diff(t, 1)

T = sp.Rational(1, 2) * M * Xdot**2
V = sp.Rational(1, 2) * (K1 * (X - Xeq1)**2 + K2 * (X - Xeq2)**2)

L = T - V

dLdX = L.diff(X, 1)
dLdXdot = L.diff(Xdot, 1)
ddtdLdXdot = dLdXdot.diff(t, 1)

dL = ddtdLdXdot - dLdX

sol = sp.solve(dL,X.diff(t, 2))

A = sol[0]

#------------------------------------------------------

m = 1
k1 = 50 
k2 = 50
xeq1 = 2
xeq2 = 8
xo = 2 
vo = 0
tf = 10

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

p = m, k1, k2, xeq1, xeq2
ic = xo, vo

xv = odeint(integrate, ic, ta, args=(p,))

ke = np.asarray([T.subs({M:m, Xdot:i}) for i in xv[:,1]])
pe = np.asarray([V.subs({K1:k1, K2:k2, Xeq1:xeq1, Xeq2:xeq2, X:i}) for i in xv[:,0]])
E = ke + pe

#-------------------------------------------------------

rad = 0.25
post1 = 0
post2 = xeq2 + xeq1
yline = 0
xmax = post2 + rad
xmin = post1 - rad
ymax = yline + 2 * rad
ymin = yline - 2 * rad
nl1 = int(np.ceil((max(xv[:,0]) - post1 - rad)/(2 * rad)))
nl2 = int(np.ceil((post2 - min(xv[:,0]) - rad)/(2 * rad)))
xl1 = np.zeros((nl1,nframes))
yl1 = np.zeros((nl1,nframes))
xl2 = np.zeros((nl2,nframes))
yl2 = np.zeros((nl2,nframes))
for i in range(nframes):
	l1 = (xv[i,0] - post1 - rad)/nl1
	l2 = (post2 - xv[i,0] - rad)/nl2
	xl1[0][i] = xv[i,0] - rad - 0.5 * l1
	xl2[0][i] = xv[i,0] + rad + 0.5 * l2
	for j in range(1,nl1):
		xl1[j][i] = xl1[j-1][i] - l1
	for j in range(nl1):
		yl1[j][i] = yline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l1)**2))
	for j in range(1,nl2):
		xl2[j][i] = xl2[j-1][i] + l2
	for j in range(nl2):
		yl2[j][i] = yline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l2)**2))

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((xv[frame,0],yline),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([post1,post1],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([post2,post2],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([xv[frame,0]-rad,xl1[0][frame]],[yline,yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[nl1-1][frame],post1],[yl1[nl1-1][frame],yline],'xkcd:cerulean')
	for i in range(nl1-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	plt.plot([xv[frame,0]+rad,xl2[0][frame]],[yline,yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[nl2-1][frame],post2],[yl2[nl2-1][frame],yline],'xkcd:cerulean')
	for i in range(nl2-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.title("Two Springs")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
#writervideo = animation.FFMpegWriter(fps=nfps)
#ani.save('two_spring_chain.mp4', writer=writervideo)
plt.show()

















