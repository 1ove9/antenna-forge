"""Unit tests for geometry kernel and parametric generators."""

import numpy as np
from yaf_core.geometry.kernel import GeometryKernel
from yaf_core.geometry.parametric import ParametricGenerator


class TestGeometryKernel:
    def test_make_box(self):
        kernel = GeometryKernel()
        box = kernel.make_box(0.1, 0.05, 0.02, (0, 0, 0))
        assert box.num_vertices == 8
        assert box.num_faces == 12

    def test_make_cylinder(self):
        kernel = GeometryKernel()
        cyl = kernel.make_cylinder(0.01, 0.05, segments=16)
        assert cyl.num_vertices > 0
        assert cyl.num_faces > 0

    def test_export_stl_manual(self):
        kernel = GeometryKernel()
        box = kernel.make_box(0.01, 0.01, 0.01)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            kernel.export_stl(box, f.name)
            assert os.path.getsize(f.name) > 0
        os.unlink(f.name)


class TestParametricGenerator:
    def test_dipole(self):
        gen = ParametricGenerator()
        dipole = gen.dipole(length=0.0625, radius=0.001)
        assert dipole.num_vertices > 0
        assert dipole.num_faces > 0
        assert dipole.metadata.get("length") == 0.0625

    def test_patch(self):
        gen = ParametricGenerator()
        patch = gen.rectangular_patch(width=0.03, length=0.04)
        assert patch.num_faces > 0

    def test_horn(self):
        gen = ParametricGenerator()
        horn = gen.horn_antenna(0.1, 0.07, 0.15, 0.03, 0.015, 0.05)
        assert horn.num_faces > 0

    def test_spiral(self):
        gen = ParametricGenerator()
        spiral = gen.archimedean_spiral(0.005, 0.03, 2.0, 0.002, segments=50)
        assert spiral.num_vertices > 0

    def test_sierpinski(self):
        gen = ParametricGenerator()
        fractal = gen.sierpinski_gasket(order=2, side_length=0.1, height=0.001)
        assert fractal.num_faces > 0
