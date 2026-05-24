"""YAF geometry kernel — OpenCASCADE, parametric generators, implicit representations."""

from yaf_core.geometry.kernel import GeometryKernel
from yaf_core.geometry.parametric import ParametricGenerator
from yaf_core.geometry.implicit import SIRENGeometry
from yaf_core.geometry.topology import TopologyField

__all__ = ["GeometryKernel", "ParametricGenerator", "SIRENGeometry", "TopologyField"]
