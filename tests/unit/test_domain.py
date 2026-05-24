"""Tests for domain models."""

from yaf_core.domain.design import (
    BoundingBox, Design, DesignSpec, DesignState, DesignVersion,
    PatternTarget, Polarization, Region,
)
from yaf_core.domain.geometry import Geometry, Material, MaterialType, Mesh
from yaf_core.domain.simulation import (
    PortType, Port, SimulationSpec, SimulationJob, SParamResult, FarFieldResult, SimulationResult
)
from yaf_core.domain.optimization import (
    OptimizationMethod, OptimizationState, Trial, OptimizationRun
)


class TestDesignSpec:
    def test_create_spec(self):
        spec = DesignSpec(
            name="test_dipole", frequency_range=(2.4e9, 2.5e9),
            size_constraint=BoundingBox(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=-0.1, z_max=0.1),
            polarization=Polarization.LINEAR, material_palette=["copper"]
        )
        assert spec.name == "test_dipole"
        assert spec.frequency_range == (2.4e9, 2.5e9)
        assert spec.polarization == Polarization.LINEAR

    def test_design_lifecycle(self):
        spec = DesignSpec(name="t", frequency_range=(1e9,2e9), size_constraint=BoundingBox(x_min=0,x_max=1,y_min=0,y_max=1,z_min=0,z_max=1))
        design = Design(spec=spec)
        assert design.state == DesignState.DRAFT
        assert not design.state.is_terminal
        design.state = DesignState.SOLVED
        assert design.state.is_terminal

    def test_bounding_box(self):
        bb = BoundingBox(x_min=0, x_max=10, y_min=0, y_max=5, z_min=-1, z_max=1)
        assert bb.dimensions == (10, 5, 2)
        assert bb.volume == 100

    def test_pattern_target(self):
        pt = PatternTarget(main_lobe_direction=(90, 0), beamwidth_3db=60, side_lobe_level_db=-20)
        assert pt.beamwidth_3db == 60
        assert pt.side_lobe_level_db == -20

    def test_design_version(self):
        import uuid
        dv = DesignVersion(design_id=uuid.uuid4(), version=1, geometry_hash="abc123")
        assert dv.version == 1
        assert dv.geometry_hash == "abc123"


class TestGeometry:
    def test_material(self):
        mat = Material(id="cu", name="Copper", type=MaterialType.CONDUCTOR, sigma=5.8e7)
        assert mat.sigma == 5.8e7
        assert mat.type == MaterialType.CONDUCTOR

    def test_geometry_serialization(self):
        geom = Geometry(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
        assert geom.num_vertices == 3
        assert geom.num_faces == 1
        data = geom.serialize()
        geom2 = Geometry.deserialize(data)
        assert geom2.num_vertices == 3


class TestSimulation:
    def test_spec(self):
        spec = SimulationSpec(frequency_range=(1e9, 2e9), frequency_points=51)
        assert spec.frequency_range == (1e9, 2e9)

    def test_job(self):
        import uuid
        spec = SimulationSpec(frequency_range=(1e9, 2e9))
        job = SimulationJob(design_id=uuid.uuid4(), design_version=1, spec=spec, solver_name="nec2")
        assert job.solver_name == "nec2"
        assert job.state == "pending"

    def test_s_param_result(self):
        result = SParamResult(frequency=[1e9], s_matrix=[[[complex(0.1, 0)]]])
        assert len(result.frequency) == 1

    def test_far_field(self):
        ff = FarFieldResult(theta=[0, 90], phi=[0], e_theta=[[1j,0j],[0j,0j]], e_phi=[[0j,0j],[0j,0j]], frequency=1e9)
        gain = ff.gain_dbi()
        assert len(gain) == 2
        assert len(gain[0]) == 1


class TestOptimization:
    def test_trial(self):
        import uuid
        t = Trial(run_id=uuid.uuid4(), trial_number=0, parameters={"len": 0.03})
        assert t.trial_number == 0

    def test_run(self):
        import uuid
        run = OptimizationRun(design_id=uuid.uuid4(), method=OptimizationMethod.BAYESIAN)
        assert run.method == OptimizationMethod.BAYESIAN
        assert run.state == OptimizationState.PENDING
