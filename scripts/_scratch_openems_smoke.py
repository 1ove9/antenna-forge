#!/usr/bin/env python3
"""Minimal openEMS smoke test — confirm build -> run -> read a real result.

Not part of the product. This verifies that the installed openEMS Python
bindings can construct a CSX structure, run a real FDTD pass, and read back a
genuine port result (S11 dip + input impedance) and a near-field-to-far-field
gain. It is a shrunk copy of the official Simple_Patch_Antenna tutorial
(openEMS / ContinuousStructure / AddLumpedPort / AddEdges2Grid /
CreateNF2FFBox / Run / CalcPort / CalcNF2FF), which is the exact API path the
real adapter uses.

Run:
    python3 scripts/_scratch_openems_smoke.py
"""

import os
import tempfile

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

sim_path = os.path.join(tempfile.gettempdir(), "openems_smoke_patch")

# Rogers-like patch (mirrors the official tutorial)
patch_width = 32.0      # x, resonant length (mm)
patch_length = 40.0     # y (mm)
sub_epsR = 3.38
sub_kappa = 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * sub_epsR
sub_w, sub_l, sub_t = 60.0, 60.0, 1.524
sub_cells = 4
feed_pos = -6.0
feed_R = 50.0
SimBox = np.array([200.0, 200.0, 150.0])
f0, fc = 2e9, 1e9

FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(["MUR"] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0 / (f0 + fc) / 1e-3 / 20

mesh.AddLine("x", [-SimBox[0] / 2, SimBox[0] / 2])
mesh.AddLine("y", [-SimBox[1] / 2, SimBox[1] / 2])
mesh.AddLine("z", [-SimBox[2] / 3, SimBox[2] * 2 / 3])

patch = CSX.AddMetal("patch")
patch.AddBox(priority=10,
             start=[-patch_width / 2, -patch_length / 2, sub_t],
             stop=[patch_width / 2, patch_length / 2, sub_t])
FDTD.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=mesh_res / 2)

substrate = CSX.AddMaterial("substrate", epsilon=sub_epsR, kappa=sub_kappa)
substrate.AddBox(priority=0,
                 start=[-sub_w / 2, -sub_l / 2, 0],
                 stop=[sub_w / 2, sub_l / 2, sub_t])
mesh.AddLine("z", np.linspace(0, sub_t, sub_cells + 1))

gnd = CSX.AddMetal("gnd")
gnd.AddBox(priority=10,
           start=[-sub_w / 2, -sub_l / 2, 0],
           stop=[sub_w / 2, sub_l / 2, 0])
FDTD.AddEdges2Grid(dirs="xy", properties=gnd)

port = FDTD.AddLumpedPort(1, feed_R, [feed_pos, 0, 0], [feed_pos, 0, sub_t],
                          "z", 1.0, priority=5, edges2grid="xy")

mesh.SmoothMeshLines("all", mesh_res, 1.4)
nf2ff = FDTD.CreateNF2FFBox()

print("running FDTD ...")
FDTD.Run(sim_path, cleanup=True, verbose=0)

f = np.linspace(1e9, 3e9, 401)
port.CalcPort(sim_path, f)
s11 = port.uf_ref / port.uf_inc
zin = port.uf_tot / port.if_tot
s11_db = 20 * np.log10(np.abs(s11))

idx = int(np.argmin(s11_db))
f_res = f[idx]
print(f"read {len(f)} frequency points")
print(f"  S11 min  = {s11_db[idx]:.2f} dB @ {f_res/1e9:.3f} GHz")
print(f"  Zin@res  = {zin[idx].real:.1f} {zin[idx].imag:+.1f}j ohm")

theta = np.arange(-180.0, 182.0, 2.0)
nf = nf2ff.CalcNF2FF(sim_path, f_res, theta, [0.0, 90.0], center=[0, 0, 1e-3])
dmax = float(np.asarray(nf.Dmax).reshape(-1)[0])
print(f"  Dmax     = {dmax:.3f}  ({10*np.log10(dmax):.2f} dBi)")
print("SMOKE OK: openEMS built, ran, returned real S11/Zin and NF2FF gain.")
