#!/usr/bin/env python3
"""Export archived ManiSkill task props in millimeters; no simulator/model changes."""
import argparse
import csv
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'datasets/multisim_sidecar_archive_v2_20261025'


def box(size):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(np.asarray(size)/2)
    return mesh


def ball(radius=20, hemisphere=False, segments=192, rings=127):
    """Analytic latitude mesh with explicit poles; hemisphere has a closed equatorial disk."""
    max_theta = math.pi/2 if hemisphere else math.pi
    vertices = [[0, 0, radius]]
    for j in range(1, rings+1):
        theta = max_theta*j/(rings if hemisphere else rings+1)
        for i in range(segments):
            phi = 2*math.pi*i/segments
            vertices.append([radius*math.sin(theta)*math.cos(phi),
                             radius*math.sin(theta)*math.sin(phi), radius*math.cos(theta)])
    end = len(vertices)
    vertices.append([0, 0, 0 if hemisphere else -radius])
    faces = []
    for i in range(segments):
        n = (i+1) % segments
        faces.append([0, 1+i, 1+n])
        for j in range(rings-1):
            a, b = 1+j*segments+i, 1+j*segments+n
            faces.extend([[a, a+segments, b+segments], [a, b+segments, b]])
        faces.append([end, 1+(rings-1)*segments+n, 1+(rings-1)*segments+i])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.apply_translation([0, 0, 0 if hemisphere else radius])
    return mesh


def tray():
    """Union boundary of a 50mm base and four walls, without internal overlapping faces."""
    axes = [[0, 5, 45, 50], [0, 5, 45, 50], [0, 5, 15]]
    cells = {(i, j, k) for i in range(3) for j in range(3) for k in range(2)
             if k == 0 or i in (0, 2) or j in (0, 2)}
    vertices, faces = [], []
    for cell in sorted(cells):
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for direction in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += direction
                if tuple(neighbor) in cells:
                    continue
                corners = []
                for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                    index = list(cell)
                    index[axis] += int(direction > 0)
                    index[others[0]] += a
                    index[others[1]] += b
                    corners.append([axes[d][index[d]] for d in range(3)])
                corners = np.asarray(corners, dtype=float)
                if np.cross(corners[1]-corners[0], corners[2]-corners[0])[axis]*direction < 0:
                    corners = corners[::-1]
                base = len(vertices)
                vertices.extend(corners)
                faces.extend([[base, base+1, base+2], [base, base+2, base+3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def write_3mf(path, parts):
    """Generic geometry-only 3MF, explicit millimeter unit and optional part colors."""
    namespace = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    ET.register_namespace('', namespace)
    def tag(name):
        return '{'+namespace+'}'+name
    model = ET.Element(tag('model'), unit='millimeter')
    ET.SubElement(model, tag('metadata'), name='Title').text = path.stem
    ET.SubElement(model, tag('metadata'), name='Description').text = 'Geometry only; select P2S/nozzle/filament and slice locally.'
    resources = ET.SubElement(model, tag('resources'))
    materials = ET.SubElement(resources, tag('basematerials'), id='1')
    for name, _, color in parts:
        ET.SubElement(materials, tag('base'), name=name, displaycolor=color)
    for i, (name, shape, _) in enumerate(parts):
        obj = ET.SubElement(resources, tag('object'), id=str(i+2), type='model', name=name,
                            pid='1', pindex=str(i))
        mesh = ET.SubElement(obj, tag('mesh'))
        vertices = ET.SubElement(mesh, tag('vertices'))
        for v in shape.vertices:
            ET.SubElement(vertices, tag('vertex'), x=f'{v[0]:.7f}', y=f'{v[1]:.7f}', z=f'{v[2]:.7f}')
        triangles = ET.SubElement(mesh, tag('triangles'))
        for f in shape.faces:
            ET.SubElement(triangles, tag('triangle'), v1=str(f[0]), v2=str(f[1]), v3=str(f[2]))
    parent_id = len(parts)+2
    parent = ET.SubElement(resources, tag('object'), id=str(parent_id), type='model', name=path.stem)
    components = ET.SubElement(parent, tag('components'))
    for i in range(len(parts)):
        ET.SubElement(components, tag('component'), objectid=str(i+2))
    build = ET.SubElement(model, tag('build'))
    ET.SubElement(build, tag('item'), objectid=str(parent_id))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr('3D/3dmodel.model', ET.tostring(model, encoding='utf-8', xml_declaration=True))


def visual_seed(seed, domain, component):
    return int.from_bytes(hashlib.sha256(f'franka20k-v1/{seed}/{domain}/{component}'.encode()).digest()[:8], 'little')


def validate(mesh, expected):
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 1
    assert np.isfinite(mesh.vertices).all()
    assert np.allclose(mesh.extents, expected, atol=.002), (mesh.extents, expected)
    assert abs(mesh.bounds[0, 2]) < 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/printable_maniskill_p2s_20261103')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for sub in ['stl', '3mf', 'markers', 'evidence']:
        (out/sub).mkdir(exist_ok=True)
    sources = [ARCHIVE/'replay/DiT4DiT/examples/ManiSkill3/franka20k_envs.py',
               ARCHIVE/'replay/DiT4DiT/examples/ManiSkill3/franka_hard10k_envs.py']
    sim = ARCHIVE/'replay/ManiSkill/mani_skill'
    sources.extend(sim/'envs/tasks/tabletop'/name for name in
                   ['eort_task_suite.py', 'place_sphere.py', 'lift_peg_upright.py', 'pull_cube.py', 'push_cube.py', 'pick_cube_cfgs.py'])
    sources.append(sim/'utils/building/actors/common.py')
    sources.append(ARCHIVE/'replay/DiT4DiT/examples/ManiSkill3/plan_franka20k.py')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    data = json.loads((ARCHIVE/'manifest.ready.json').read_text())
    measurements = {}
    for task in ['lift_peg_upright', 'push_cube', 'pull_cube', 'pick_cube', 'pick_cuboid', 'pick_sphere', 'place_sphere', 'stack_cube']:
        row = next(r for r in data['episodes'] if r['group']=='maniskill_franka' and r['task']==task
                   and r['split']=='test' and r['collection']=='maniskill_base' and r['domain']=='Clean')
        with h5py.File(ARCHIVE/row['media'], 'r') as f:
            extra = f['traj_0/obs/extra']
            measurements[task] = dict(id=row['id'], media=row['media'],
                extents_mm=(extra['obj_extent'][0]*1000).tolist(), mass_g=float(extra['obj_mass'][0,0])*1000,
                friction=extra['obj_friction'][0].tolist())
    assert np.allclose(measurements['lift_peg_upright']['extents_mm'], [240,50,50])
    assert np.allclose(measurements['pick_cuboid']['extents_mm'], [50,30,36])
    for task in ['push_cube','pull_cube','pick_cube','pick_sphere','place_sphere','stack_cube']:
        assert np.allclose(measurements[task]['extents_mm'], [40,40,40])
    items = []
    meshes = {}
    def export(name, shape, size, role, color, note):
        validate(shape, size)
        target = out/'stl'/(name+'.stl')
        shape.export(target)
        reloaded = trimesh.load(target, force='mesh')
        validate(reloaded, size)
        write_3mf(out/'3mf'/(name+'.3mf'), [(name, shape, color)])
        meshes[name] = shape
        items.append(dict(name=name, dimensions_mm=list(map(float, size)), role=role,
                          suggested_color=color, volume_cm3=round(shape.volume/1000, 5),
                          solid_PLA_mass_g_at_1p24=round(shape.volume/1000*1.24, 3),
                          mesh_watertight=True, note=note))
    export('cube_40', box([40,40,40]), [40,40,40], 'push/pull/pick/stack target', '#E61919FF',
           'Reuse in chosen target color; no added bevel changes contact geometry.')
    export('stack_base_cube_40_gray', box([40,40,40]), [40,40,40], 'stack_cube goal block', '#595959FF',
           'Print a second independent cube; do not glue it to the target cube.')
    export('cuboid_50x30x36', box([50,30,36]), [50,30,36], 'pick_cuboid target', '#E61919FF', '50x30 bed face, 36mm tall.')
    export('sphere_d40', ball(), [40,40,40], 'pick_sphere/place_sphere target', '#E61919FF',
           'Exact full sphere; requires slicer support. Alternative: two closed hemispheres.')
    export('sphere_hemisphere_d40_h20_print_twice', ball(hemisphere=True), [40,40,20], 'sphere manufacturing alternative', '#E61919FF',
           'Print two, equator on bed; glue flat faces and finish seam to recover sphere. No flange or flattened pole.')
    export('place_sphere_tray_50x50x15', tray(), [50,50,15], 'place_sphere goal bin', '#CCCCCCFF',
           '40x40 opening; 5mm floor, 5mm walls, 10mm cavity depth. Original simulation geometry.')
    assert abs(meshes['place_sphere_tray_50x50x15'].volume - (50*50*5+(50*50-40*40)*10)) < 1e-5
    export('peg_240x50x50_flat', box([240,50,50]), [240,50,50], 'lift_peg_upright target', '#B00E0EFF',
           'Square bar, not a cylinder; paint first/second 120mm halves red/blue.')
    export('peg_50x50x240_upright', box([50,50,240]), [50,50,240], 'same peg, print orientation alternative', '#B00E0EFF',
           'Same geometry rotated; one red-to-blue filament change at height120mm, brim suggested.')
    export('peg_half_120x50x50_print_twice', box([120,50,50]), [120,50,50], 'peg manufacturing alternative', '#B00E0EFF',
           'Print one red, one blue; end-glue to 240x50x50; joint strength unverified. Prefer whole peg.')
    lower, upper = box([50,50,120]), box([50,50,120])
    upper.apply_translation([0,0,120])
    write_3mf(out/'3mf/peg_upright_two_color_assembled.3mf',
              [('red_lower_half', lower, '#B00E0EFF'), ('blue_upper_half', upper, '#0C2AA0FF')])
    # Pick an actual accepted hard scene with all four primitive clutter shapes.
    chosen = None
    for row in data['episodes']:
        if row['collection']!='maniskill_hard' or row['split']!='test':
            continue
        metadata = json.loads((ARCHIVE/row['media']).with_suffix('.json').read_text())
        index = metadata['env_info']['env_kwargs']['hard_index']
        if index % 5 in (2,4):
            chosen = (row, metadata)
            break
    row, metadata = chosen
    seed = metadata['episodes'][0]['episode_seed']
    rng = np.random.default_rng(visual_seed(seed, 'hard-physical', 'geometry'))
    clutter = []
    for i, kind in enumerate(['sphere','box','cylinder','cuboid']):
        ext = rng.uniform(.03,.06,3).astype(np.float32)
        if kind=='sphere':
            radius = float(rng.uniform(.018,.028))*1000
            size = [2*radius]*3
            shape = ball(radius)
        elif kind=='cylinder':
            radius = float(rng.uniform(.016,.025))*1000
            half_h = float(rng.uniform(.018,.035))*1000
            size = [2*radius, 2*radius, 2*half_h]
            shape = trimesh.creation.cylinder(radius=radius, height=2*half_h, sections=192)
            shape.apply_translation([0,0,half_h])
        else:
            if kind=='box':
                ext[:] = ext[0]
            else:
                ext[:] = [rng.uniform(.045,.065),rng.uniform(.025,.035),rng.uniform(.03,.05)]
            size = (ext*1000).tolist()
            shape = box(size)
        name = f'hard_clutter_{i+1}_{kind}'
        export(name, shape, size, 'optional physical distractor', ['#194DE6FF','#19B233FF','#F2BF19FF','#8C33BFFF'][i],
               'Geometry sampled exactly from one archived hard test scene; generic solid color, not all randomized appearances.')
        clutter.append(dict(name=name, kind=kind, dimensions_mm=size))
    with h5py.File(ARCHIVE/row['media'], 'r') as f:
        saved_extents_mm = np.asarray(f['traj_0/obs/extra/hard10k_clutter_extents'][0])*1000
    assert np.allclose(saved_extents_mm, [c['dimensions_mm'] for c in clutter], atol=.002), 'Hard clutter does not match saved geometry'
    colors = ['#C21316','#FFFFFF','#C21316','#FFFFFF','#C21316']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="210mm" viewBox="0 0 210 210">']
    for radius, color in zip([100,80,60,40,20], colors):
        svg.append(f'<circle cx="105" cy="105" r="{radius}" fill="{color}"/>')
    svg.append('</svg>')
    (out/'markers/push_pull_goal_d200_print_at_100percent.svg').write_text('\n'.join(svg))
    (out/'markers/pick_goal_virtual_d50_reference_only.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="60mm" viewBox="0 0 60 60"><circle cx="30" cy="30" r="25" fill="#00C840"/><text x="30" y="33" text-anchor="middle" font-size="5">VIRTUAL</text></svg>')
    provenance = dict(utc=datetime.now(timezone.utc).isoformat(), units='millimeter',
        printer_assumption='User said P2SC; interpreted as P2S Combo; nozzle0.4mm/PLA assumed',
        source_hashes=hashes, measured_task_objects=measurements,
        clutter_source=dict(id=row['id'], media=row['media'], seed=seed,
                            hard_index=metadata['env_info']['env_kwargs']['hard_index'], shapes=clutter,
                            actual_HDF5_clutter_extents_match=True),
        validation='STL reloaded: closed connected orientable positive-volume meshes, bounds and dimensions checked',
        not_validated=['Bambu Studio import/slice', 'physical print accuracy', 'mass/friction match', 'robot trials'])
    (out/'evidence/source_geometry.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'parts.json').write_text(json.dumps(items,indent=2)+'\n')
    with (out/'parts.csv').open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    preview(out, meshes)
    for path in (out/'3mf').glob('*.3mf'):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            xml = ET.fromstring(z.read('3D/3dmodel.model'))
            assert xml.attrib['unit']=='millimeter'
    sha = '\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))
                    for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(sha+'\n')
    print(json.dumps(dict(output=str(out), printable_STL_parts=len(items),
                         models_3mf=len(list((out/'3mf').glob('*.3mf'))),all_mesh_checks_passed=True),indent=2))


def preview(out, meshes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    names = ['cube_40','stack_base_cube_40_gray','cuboid_50x30x36','sphere_d40',
             'place_sphere_tray_50x50x15','peg_240x50x50_flat','sphere_hemisphere_d40_h20_print_twice',
             'hard_clutter_3_cylinder']
    titles = ['Target cube | 40 x 40 x 40 mm','Stack goal cube | 40 x 40 x 40 mm',
              'Cuboid | 50 x 30 x 36 mm','Sphere | diameter 40 mm',
              'Tray | 50 x 50 x 15 mm\n40 mm opening / 10 mm cavity',
              'Square peg | 240 x 50 x 50 mm\nred + blue, each 120 mm',
              'Hemisphere | D40 x H20 mm\nprint two, join on flat faces',
              'Example hard-scene cylinder\nexact dimensions in parts.csv']
    fig = plt.figure(figsize=(16,9),facecolor='#f5f7fa')
    for index, (name,title) in enumerate(zip(names,titles)):
        ax=fig.add_subplot(2,4,index+1,projection='3d',computed_zorder=False)
        mesh=meshes[name]
        display = ball(hemisphere=index==6, segments=64, rings=31) if index in (3,6) else mesh
        polygons=display.triangles
        color='#CE3C3C' if index not in [1,4,7] else ['#CE3C3C','#707780','#CE3C3C','#CE3C3C','#B5BCC4','#CE3C3C','#CE3C3C','#3F81C5'][index]
        if index==5:
            for x,c in [(0,'#B00E0E'),(120,'#0C2AA0')]:
                piece=box([120,50,50]);piece.apply_translation([x,0,0])
                shade = np.clip(.7+.3*piece.face_normals@np.array([-.3,-.5,1]),.35,1)
                ax.add_collection3d(Poly3DCollection(piece.triangles,facecolor=shade[:,None]*to_rgb(c),edgecolor='none'))
        else:
            shade = np.clip(.7+.3*display.face_normals@np.array([-.3,-.5,1]),.35,1)
            ax.add_collection3d(Poly3DCollection(polygons,facecolor=shade[:,None]*to_rgb(color),
                                                 edgecolor='#89929D' if index==4 else 'none',
                                                 linewidth=.3 if index==4 else 0))
        low,high=mesh.bounds
        ax.set_xlim(low[0]-3,high[0]+3);ax.set_ylim(low[1]-3,high[1]+3);ax.set_zlim(0,high[2]+3)
        ax.set_box_aspect(mesh.extents+6);ax.view_init(elev=27,azim=-57)
        ax.set_title(title,fontsize=10,pad=5);ax.set_axis_off()
    fig.suptitle('ManiSkill task props — source-sized geometry for Bambu P2S Combo',fontsize=17,y=.98)
    fig.text(.5,.025,'All dimensions in mm. Geometry-only STL/3MF; select printer and filament in Bambu Studio. Renderings are not to a shared scale.',ha='center',fontsize=10)
    fig.subplots_adjust(wspace=.04,hspace=.14,top=.9,bottom=.07)
    fig.savefig(out/'overview.png',dpi=180);fig.savefig(out/'overview.pdf');plt.close(fig)


if __name__=='__main__':
    main()
