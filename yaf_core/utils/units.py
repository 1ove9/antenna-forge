"""Unit conversion utilities for antenna engineering."""
from __future__ import annotations
import math
C0=3e8;EPS0=8.854187817e-12;MU0=4e-7*math.pi;ETA0=math.sqrt(MU0/EPS0)
def wavelength(freq_hz:float)->float:return C0/freq_hz
def freq_to_omega(freq_hz:float)->float:return 2*math.pi*freq_hz
def db_to_linear(db:float)->float:return 10**(db/10)
def linear_to_db(lin:float)->float:
    if lin<=0:return -200
    return 10*math.log10(lin)
def vswr_from_s11(s11_mag:float)->float:
    if s11_mag>=1:return float('inf')
    return (1+s11_mag)/(1-s11_mag)
def impedance_from_s11(s11:complex,z0:float=50)->complex:
    return z0*(1+s11)/(1-s11)
def s11_from_impedance(z:complex,z0:float=50)->complex:
    return (z-z0)/(z+z0)
def dipole_length_estimate(freq_hz:float,factor:float=0.475)->float:
    return C0/freq_hz*factor
def neper_to_db(np:float)->float:return np*8.685889638
