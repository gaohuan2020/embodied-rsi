"""Panda presentation: only modify the render scene, never collision geometry."""
from itertools import pairwise

import numpy as np


def decorate(scene, world, trail=()):
    import mujoco

    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0:
            continue
        name = world.model.geom(geom.objid).name or ""
        body = world.model.body(int(world.model.geom_bodyid[geom.objid])).name or ""
        if name == "table":
            geom.rgba[:] = [.105, .15, .205, 1.]
            geom.specular, geom.shininess = .35, .5
        elif body.startswith("link") or body in {"hand", "left_finger", "right_finger"}:
            if max(geom.rgba[:3]) > .45:
                geom.rgba[:] = [.78, .86, .94, 1.]
            else:
                geom.rgba[:] = [.055, .10, .145, 1.]
            geom.specular, geom.shininess = .6, .7

    def line(a, b, color, width=.001):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array(color))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array(a), np.array(b))
        geom.emission = .35
        scene.ngeom += 1

    def box(position, size, color, emission=0.):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.array(size), np.array(position),
                           np.eye(3).ravel(), np.array(color))
        geom.emission, geom.specular = emission, .2
        scene.ngeom += 1

    # Render-only laboratory backdrop and workstation; these have no physical contacts.
    box([.32, 0., -.068], [.57, .45, .018], [.13, .21, .28, 1.])
    for x in (-.16, .8):
        for y in (-.35, .35):
            box([x, y, -.26], [.022, .022, .18], [.11, .19, .25, 1.])
    box([.1, 1.2, .55], [1.8, .025, 1.02], [.12, .2, .28, 1.], .06)
    box([-.9, .15, .55], [.025, 1.45, 1.02], [.105, .17, .24, 1.], .06)
    for x in np.arange(-1.3, 1.51, .4):
        line([x, 1.168, -.4], [x, 1.168, .93], [.19, .34, .43, 1.], .002)
    line([-1.5, 1.16, .85], [1.5, 1.16, .85], [.27, .72, .85, 1.], .007)
    line([-.868, -1.3, .85], [-.868, 1.16, .85], [.27, .72, .85, 1.], .007)
    for x in (-.8, -.1, .6):
        box([x, 1.163, .45], [.27, .005, .15], [.055, .12, .18, 1.])
        for z, width in ((.51, .21), (.43, .14), (.36, .18)):
            line([x - .21, 1.152, z], [x - .21 + width * 2, 1.152, z], [.22, .57, .68, 1.], .003)
    for value in np.arange(-1.5, 1.76, .25):
        line([value, -1.25, -.457], [value, 1.5, -.457], [.12, .23, .3, .7], .001)
        line([-1.5, value, -.457], [1.55, value, -.457], [.12, .23, .3, .7], .001)

    # Subtle table grid, base halo, tray outline, and recent tool path.
    for x in np.arange(-.2, .86, .1):
        line([x, -.42, .001], [x, .42, .001], [.13, .26, .34, .5], .0004)
    for y in np.arange(-.4, .41, .1):
        line([-.22, y, .001], [.86, y, .001], [.13, .26, .34, .5], .0004)
    angles = np.linspace(0, 2 * np.pi, 49)
    ring = np.c_[.115 * np.cos(angles), .115 * np.sin(angles), np.full(len(angles), .004)]
    for a, b in pairwise(ring):
        line(a, b, [.17, .82, .91, 1.], .0016)
    target = world.target
    radius = .088 if world.task != "stack" else .037
    corners = [[target[0] + x, target[1] + y, .002] for x, y in
               [(-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius)]]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        line(a, b, [.28, .94, .72, .85], .0015)
    points = list(trail)
    for a, b in pairwise(points):
        if np.linalg.norm(np.asarray(a) - b) > .001:
            line(a, b, [.24, .78, 1., .55], .0012)
