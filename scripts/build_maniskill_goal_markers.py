#!/usr/bin/env python3
"""Printable hollow visual goal and thin/stencil push goal; no task or policy changes."""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from scripts.build_maniskill_printables import ROOT, write_3mf


def hollow_goal(opening=36, wall=1.6):
    radius, inner = 25., 25.-wall
    if wall <= 0 or inner <= 0 or not 0 < opening < 2*inner:
        raise ValueError('Need positive wall and an opening smaller than the inner sphere diameter')
    height = math.sqrt(inner*inner-(opening/2)**2)
    outer_angle = np.sort(np.r_[np.linspace(-math.pi/2,math.asin(height/radius),129),0])
    inner_angle = np.linspace(math.asin(height/inner),-math.pi/2,129)
    outer = np.c_[radius*np.cos(outer_angle),radius*np.sin(outer_angle)]
    inside = np.c_[inner*np.cos(inner_angle),inner*np.sin(inner_angle)]
    outer[0,0], inside[-1,0] = 0,0
    profile = np.vstack([outer,inside,outer[:1]])
    mesh = trimesh.creation.revolve(profile,sections=192)
    mesh.apply_translation([0,0,25])
    return mesh, dict(sphere_outer_diameter_mm=50, wall_mm=wall, opening_inner_diameter_mm=opening,
                       rim_z_mm=height+25, sphere_center_z_mm=25,
                       rim_outer_diameter_mm=2*math.sqrt(radius*radius-height*height))


def annulus(inner, outer, thickness):
    if inner < 0 or not inner < outer or thickness <= 0:
        raise ValueError('Invalid annular plate dimensions')
    profile = [[inner,0],[outer,0],[outer,thickness],[inner,thickness],[inner,0]]
    return trimesh.creation.revolve(profile,sections=256)


def check_mesh(shape):
    assert shape.is_watertight and shape.is_winding_consistent and shape.volume>0
    assert len(shape.split(only_watertight=False))==1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opening',type=float,default=36)
    p.add_argument('--output',type=Path,default=ROOT/'assets/printable_goal_markers_v2_20261103')
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    shape, dimensions=hollow_goal(args.opening)
    shapes={'pick_green_hollow_D50_open'+f'{args.opening:g}'+'_wall1p6':(shape,'#00FF00FF')}
    flipped=shape.copy();flipped.apply_transform(trimesh.transformations.rotation_matrix(math.pi,[1,0,0]))
    flipped.apply_translation([0,0,-flipped.bounds[0,2]])
    shapes['pick_green_hollow_print_opening_down']=(flipped,'#00FF00FF')
    shapes['push_goal_plain_disc_D200_H0p6']=(annulus(0,100,.6),'#C21316FF')
    shapes['push_goal_removable_stencil_ID200_OD210_H1p2']=(annulus(100,105,1.2),'#CCCCCCFF')
    for name,(mesh,color) in shapes.items():
        check_mesh(mesh);mesh.export(out/(name+'.stl'))
        check_mesh(trimesh.load(out/(name+'.stl'),force='mesh'))
        write_3mf(out/(name+'.3mf'),[(name,mesh,color)])
    parts=[]
    for i,(r0,r1) in enumerate([(80,100),(60,80),(40,60),(20,40),(0,20)]):
        mesh=annulus(r0,r1,.6);check_mesh(mesh)
        parts.append((f'ring_{r0:g}_{r1:g}mm',mesh,'#C21316FF' if i%2==0 else '#FFFFFFFF'))
    write_3mf(out/'push_goal_red_white_D200_H0p6_multicolor.3mf',parts)
    assert abs(sum(mesh.volume for _,mesh,_ in parts)-shapes['push_goal_plain_disc_D200_H0p6'][0].volume)<1e-5
    for f in out.glob('*.3mf'):
        loaded=trimesh.load(f,force='scene');assert loaded.bounds is not None
        with zipfile.ZipFile(f) as z:assert z.testzip() is None
    dimensions.update(gripper_fit_verified=False,preferred_orientation_for_observation='opening away from viewing camera',
                      full_spherical_cap_removed=True,physical_task_equivalence=False)
    (out/'dimensions_and_checks.json').write_text(json.dumps(dimensions,indent=2)+'\n')
    preview(out,shape,parts)
    print(json.dumps(dict(output=str(out),hollow_ball=dimensions,files=[f.name for f in sorted(out.iterdir())]),indent=2))


def preview(out,shell,parts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,5))
    for i in range(2):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        selected=[('shell',shell,'#30C34B')] if i==0 else parts
        for _,mesh,color in selected:
            shade=np.clip(.7+.3*mesh.face_normals@np.array([-.3,-.5,1]),.3,1)
            ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor=shade[:,None]*to_rgb(color),edgecolor='none'))
        if i==0:
            ax.set_xlim(-26,26);ax.set_ylim(-26,26);ax.set_zlim(0,50);ax.set_box_aspect([1,1,1])
            ax.set_title('Green hollow visual goal\nouter sphere D50 / wall1.6 / opening36 mm')
            ax.view_init(elev=40,azim=-55)
        else:
            ax.set_xlim(-106,106);ax.set_ylim(-106,106);ax.set_zlim(0,1)
            ax.set_box_aspect([1,1,.03]);ax.view_init(elev=70,azim=-75)
            ax.set_title('Push/Pull red-white target\nD200 x H0.6 mm; flush colored rings')
        ax.set_axis_off()
    fig.tight_layout();fig.savefig(out/'preview.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
