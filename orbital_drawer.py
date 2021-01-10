# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:06:40 2020

Copyright Jake Skelton
"""


import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as opt


def RDF(r, theta=0.0, psi=0.0):
    """
    Callable for the defining equation of the wavefunction contour (for positive wavefunction
    values).

    Parameters
    ----------
    r : float
        DESCRIPTION.
    theta : float, optional
        DESCRIPTION. The default is 0.0.
    psi : float, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    f : float
        The RHS of the wavefunction contour equation. f=0 defines the contour of the wavefunction
        at value 'phi'.

    """
    if ((3*(np.cos(theta))**2 - 1)/psi).any() < 0:
        print("Error in RDF, log got negative argument")
    f = 2*np.log(r) - r/3 + np.log((3*(np.cos(theta))**2 - 1)/psi)
    return f

def RDF2(r, theta=0.0, psi=0.0):
    """
    Callable for the defining equation of the wavefunction contour (for negative wavefunction
    values).

    Parameters
    ----------
    r : float
        DESCRIPTION.
    theta : float, optional
        DESCRIPTION. The default is 0.0.
    psi : float, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    f : float
        The RHS of the wavefunction contour equation. f=0 defines the contour of the wavefunction
        at value 'phi'.

    """
    if ((1 - 3*np.cos(theta)**2)/psi < 0).any():
        print("Error in RDF2, log got negative argument")
    ## Note difference in argument of 2nd log compared to RDF ##
    f = 2*np.log(r) - r/3 + np.log((1 - 3*(np.cos(theta))**2)/psi)
    return f

def deriv(r, theta=0.0):
    """
    The derivative of the wavefunction contour w.r.t. 'r'. This the same for positive and negative
    contours.

    Parameters
    ----------
    r : float
        DESCRIPTION.
    theta : float, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    df : float
        DESCRIPTION.

    """
    df = 2/r - 1/3
    return df

## contour is the (un-normalised) value of the wavefunction at which to plot
## (sqrt of probability denisty).
contour = 1 
## t_star is the theta beyond which f(r, theta, phi) is insoluble. ##
t_star = np.arccos(np.sqrt((1 + contour*(np.e/6)**2)/3))
t_2star = np.arccos(np.sqrt((1 - contour*(np.e/6)**2)/3))
#print("theta_star = ", t_star)
t_1 = np.linspace(0, t_star, 200)[:-1] ## Nominally 200 points per half-lobe of the wavefunction.
t_1 = np.hstack((t_1, t_1[::-1]))
t_2 = np.linspace(t_2star, np.pi/2, 200)[1:]
t_2 = np.hstack((t_2[::-1], t_2))

## Meat of the program: populates the arrays 'r' of the radial values of the wavefunction contours.
r_1 = np.zeros_like(t_1)
l_1 = t_1.size//2
r_1[:l_1] = opt.newton(partial(RDF, theta=t_1[:l_1], psi=contour) , x0=7*np.ones(l_1), 
                      fprime=partial(deriv, theta=t_1[:l_1]))
r_1[l_1:] = opt.newton(partial(RDF, theta=t_1[l_1:], psi=contour) , x0=0.1*np.ones(l_1), 
                      fprime=partial(deriv, theta=t_1[l_1:]))

r_2 = np.zeros_like(t_2)
l_2 = t_2.size//2
r_2[l_2:] = opt.newton(partial(RDF2, theta=t_2[l_2:], psi=contour), x0=7*np.ones(l_2), 
                          fprime=partial(deriv, theta=t_2[l_2:]))
r_2[:l_2] = opt.newton(partial(RDF2, theta=t_2[:l_2], psi=contour), x0=0.1*np.ones(l_2), 
                          fprime=partial(deriv, theta=t_2[:l_2]))

## Matplotlib stuff ##
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar', frame_on=False)
## Removes grid and axes of plot ##
ax.grid(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

## Root-finding only finds wavefunction contour in the positive quadrant. Rest follows by symmetry.
for n in range(2):
    ax.plot(np.hstack((t_1, -t_1)) + n*np.pi, np.hstack((r_1, r_1[::-1])), 'k-')
    ax.plot(np.hstack((t_2, np.pi - t_2)) + n*np.pi, np.hstack((r_2, r_2[::-1])), 'k-')

## Delete hash to save the image ##
#plt.savefig("C:\\Users\\jakes\\pictures\\3d_orbital.svg", dpi=300)
