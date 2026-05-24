"""Unit tests for physics models."""

import numpy as np
from yaf_core.physics.materials import MaterialLibrary
from yaf_core.physics.metasurface import MetasurfaceModel, UnitCell
from yaf_core.physics.ris import RISModel, RISElement
from yaf_core.physics.oam import OAMModel
from yaf_core.physics.graphene import GrapheneModel


class TestMaterialLibrary:
    def test_materials_seeded(self):
        lib = MaterialLibrary()
        mats = lib.list_all()
        assert len(mats) >= 18  # minimum materials seeded

    def test_copper(self):
        lib = MaterialLibrary()
        cu = lib.get("copper")
        assert cu.sigma == 5.8e7

    def test_graphene_dispersion(self):
        lib = MaterialLibrary()
        eps = lib.get_dispersive_permittivity("graphene", 1e12)
        assert isinstance(eps, complex)

    def test_drude_plasma(self):
        lib = MaterialLibrary()
        eps = lib.get_dispersive_permittivity("plasma_ar", 1e10)
        assert isinstance(eps, complex)

    def test_unknown_material(self):
        lib = MaterialLibrary()
        try:
            lib.get("nonexistent")
            assert False, "Should have raised"
        except KeyError:
            pass


class TestMetasurface:
    def test_unit_cell(self):
        cell = UnitCell(s11_db=-40, s21_db=-0.5, s21_phase_deg=45)
        s11 = cell.s11()
        assert abs(abs(s11) - 0.01) < 0.01
        s21 = cell.s21()
        assert abs(abs(s21) - 10**(-0.5/20)) < 0.05

    def test_array_factor(self):
        model = MetasurfaceModel(nx=8, ny=8, frequency=10e9)
        theta = np.linspace(0, np.pi, 37)
        phi = np.linspace(0, 2*np.pi, 73)
        af = model.array_factor(theta, phi)
        assert af.shape == (37, 73)


class TestRIS:
    def test_ris_element(self):
        elem = RISElement(bits=2)
        assert elem.states == 4
        elem.set_state(0)
        assert elem.phase_deg == 0
        elem.set_state(1)
        assert elem.phase_deg == 90

    def test_ris_codebook(self):
        ris = RISModel(nx=8, ny=8, bits=2)
        assert "broadside" in ris.codebook
        ris.apply_codebook("broadside")
        phases = ris.get_phase_configuration()
        assert phases.shape == (8, 8)


class TestOAM:
    def test_near_field(self):
        oam = OAMModel(topological_charge=1, frequency=10e9)
        rho = np.linspace(0.001, 0.1, 20)
        phi = np.zeros_like(rho)
        field = oam.near_field(rho, phi)
        assert len(field) == 20


class TestGraphene:
    def test_conductivity(self):
        model = GrapheneModel(mu_c_ev=0.2, temperature_k=300)
        sigma = model.surface_conductivity(frequency=1e12)
        assert sigma.real > 0
