import math

import numpy as np
import pytest

pytest.importorskip('trimesh')
pytest.importorskip('h5py')
import trimesh

from scripts.build_maniskill_goal_markers import annulus, check_mesh, hollow_goal


def test_open_sphere_has_real_36mm_inner_mouth_and_16mm_wall():
    mesh, dims = hollow_goal()
    check_mesh(mesh)
    assert np.allclose(mesh.extents[:2], [50,50])
    assert dims['rim_z_mm'] < 50
    top = mesh.vertices[np.isclose(mesh.vertices[:,2], dims['rim_z_mm'])]
    radii = np.linalg.norm(top[:,:2],axis=1)
    assert min(radii) == pytest.approx(18)
    assert max(radii) == pytest.approx(dims['rim_outer_diameter_mm']/2)
    assert mesh.bounds[0,2] == pytest.approx(0)
    def below(radius,height):
        return math.pi*(radius*radius*height-height**3/3+2*radius**3/3)
    h = dims['rim_z_mm']-25
    expected=below(25,h)-below(23.4,h)
    assert mesh.volume == pytest.approx(expected,rel=.001)


def test_push_plate_is_thin_and_color_rings_do_not_overlap():
    plate=annulus(0,100,.6)
    check_mesh(plate)
    assert np.allclose(plate.extents,[200,200,.6])
    rings=[annulus(a,b,.6) for a,b in [(0,20),(20,40),(40,60),(60,80),(80,100)]]
    assert sum(m.volume for m in rings)==pytest.approx(plate.volume)
    stencil=annulus(100,105,1.2)
    check_mesh(stencil)
    assert np.allclose(stencil.extents,[210,210,1.2])
    section=trimesh.intersections.mesh_plane(stencil,[0,0,1],[0,0,.6])
    circumference=np.linalg.norm(section[:,1]-section[:,0],axis=1).sum()
    assert circumference==pytest.approx(2*math.pi*(100+105),rel=.001)


@pytest.mark.parametrize('opening,wall', [(0,1.6),(50,1.6),(36,0),(36,26)])
def test_impossible_shell_dimensions_rejected(opening,wall):
    with pytest.raises(ValueError):
        hollow_goal(opening,wall)
