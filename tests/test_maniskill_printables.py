import math
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pytest

pytest.importorskip('trimesh')
pytest.importorskip('h5py')
import trimesh

from scripts.build_maniskill_printables import ball, box, tray, validate, write_3mf


def test_tray_has_source_sized_open_cavity_not_a_solid_block():
    shape = tray()
    validate(shape, [50,50,15])
    assert shape.volume == pytest.approx(21500)
    segments = trimesh.intersections.mesh_plane(shape, plane_normal=[0,0,1], plane_origin=[0,0,10])
    # Outer 50mm square and inner 40mm square must both be present in the wall cross-section.
    assert np.linalg.norm(segments[:,1]-segments[:,0], axis=1).sum() == pytest.approx(360)
    for segment in segments:
        assert any(np.allclose(segment[:,axis], edge) for axis in (0,1) for edge in (0,5,45,50))


@pytest.mark.parametrize('hemisphere', [False, True])
def test_sphere_is_closed_correctly_sized_without_artificial_flattening(hemisphere):
    shape = ball(hemisphere=hemisphere)
    validate(shape, [40,40,20 if hemisphere else 40])
    volume = 4/3*math.pi*20**3/(2 if hemisphere else 1)
    assert shape.volume == pytest.approx(volume, rel=.001)


def test_3mf_retains_mm_and_single_assembly_with_two_color_parts(tmp_path):
    lower, upper = box([50,50,120]), box([50,50,120])
    upper.apply_translation([0,0,120])
    path = tmp_path/'peg.3mf'
    write_3mf(path,[('red',lower,'#B00E0EFF'),('blue',upper,'#0C2AA0FF')])
    ns={'m':'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        model = ET.fromstring(z.read('3D/3dmodel.model'))
    assert model.attrib['unit']=='millimeter'
    assert len(model.findall('m:build/m:item',ns))==1
    assert len(model.findall('m:resources/m:object/m:components/m:component',ns))==2
    coords = [float(v.attrib['z']) for v in model.findall('.//m:vertex',ns)]
    assert min(coords)==0 and max(coords)==240
    assert len(model.findall('m:resources/m:basematerials/m:base',ns))==2
