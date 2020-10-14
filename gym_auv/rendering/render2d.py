"""
2D rendering framework.
Modified version of the classical control module in OpenAI's gym.

Changes:
    - Added an 'origin' argument to the draw_circle() and make_circle() functions to allow drawing of circles anywhere.
    - Added an 'outline' argument to the draw_circle() function, allows a more stylised render

Created by Haakon Robinson, based on OpenAI's gym.base_env.classical.rendering.py
"""

import os
import six
import sys
import pyglet
from pyglet import gl
import numpy as np
import math
from numpy import sin, cos, arctan2
from gym import error
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import CircularObstacle, PolygonObstacle, VesselObstacle

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

STATE_W = 96 
STATE_H = 96
VIDEO_W = 720
VIDEO_H = 600
WINDOW_W = VIDEO_W
WINDOW_H = VIDEO_H

SCALE       = 5.0        # Track scale
PLAYFIELD   = 5000   # Game over boundary
FPS         = 50
ZOOM        = 2       # Camera ZOOM
DYNAMIC_ZOOM = False
CAMERA_ROTATION_SPEED = 0.02
env_bg_h = int(2*PLAYFIELD)
env_bg_w = int(2*PLAYFIELD)

RAD2DEG = 57.29577951308232

env_bg = None
bg = None
rot_angle = None

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))


class Viewer2D(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.fixed_geoms = []
        self.transform = Transform()
        self.camera_zoom = 1.5

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_fixed(self, geom):
        self.fixed_geoms.append(geom)

    def render(self, return_rgb_array=False):
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        for geom in self.fixed_geoms:
            geom.render()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    def draw_circle(self, origin=(0,0), radius=10, res=30, filled=True, outline=True, start_angle=0, end_angle=2*np.pi, **attrs):
        geom = make_circle(origin=origin, radius=radius, res=res, filled=filled, start_angle=start_angle, end_angle=end_angle)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        if filled and outline:
            outl = make_circle(origin=origin, radius=radius, res=res, filled=False)
            _add_attrs(outl, {'color': (0,0,0), 'linewidth': 1})
            self.add_onetime(outl)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)

        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

    def transform_vertices(self, points, translation, rotation, scale=1):
        res = []
        for p in points:
            res.append((
                cos(rotation) * p[0] * scale - sin(rotation) * p[1] * scale + translation[0],
                sin(rotation) * p[0] * scale + cos(rotation) * p[1] * scale + translation[1]))
        return res

    def draw_arrow(self, base, angle, length, **attrs):
        TRIANGLE_POLY = ((-1, -1), (1, -1), (0, 1))
        head = (base[0] + length * cos(angle), base[1] + length * sin(angle))
        tri = self.transform_vertices(TRIANGLE_POLY, head, angle - np.pi / 2, scale=0.7)
        self.draw_polyline([base, head], linewidth=2, **attrs)
        self.draw_polygon(tri, **attrs)

    def draw_shape(self, vertices, position=None, angle=None, color=(1, 1, 1), filled=True, border=True):
        if (position is not None):
            poly_path = self.transform_vertices(vertices, position, angle)
        else:
            poly_path = vertices
        if (filled):
            self.draw_polygon(poly_path + [poly_path[0]], color=color)
        if (border):
            border_color = (0, 0, 0) if type(border) == bool else border
            self.draw_polyline(poly_path + [poly_path[0]], linewidth=1, color=border_color if filled else color)

    def __del__(self):
        self.close()


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        gl.glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        gl.glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        gl.glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        gl.glBegin(gl.GL_POINTS) # draw point
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if   len(self.v) == 4 : gl.glBegin(gl.GL_QUADS)
        elif len(self.v)  > 4 : gl.glBegin(gl.GL_POLYGON)
        else: gl.glBegin(gl.GL_TRIANGLES)
        for p in self.v:
            gl.glVertex3f(p[0], p[1],0)  # draw each vertex
        gl.glEnd()


def make_circle(origin=(0,0), radius=10, res=30, filled=True, start_angle=0, end_angle=2*np.pi, return_points=False):
    points = []
    for i in range(res+1):
        ang = start_angle + i*(end_angle - start_angle)/res
        points.append((math.cos(ang)*radius + origin[0], math.sin(ang)*radius + origin[1]))
    if (return_points):
        return points
    else:
        if filled:
            return FilledPolygon(points)
        else:
            return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1],0)  # draw each vertex
        gl.glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), linewidth=1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(linewidth)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)


def _render_path(env):
    env._viewer2d.draw_polyline(env.path._points, linewidth=1, color=(0.3, 1.0, 0.3))

def _render_vessel(env):
    env._viewer2d.draw_polyline(env.vessel.path_taken, linewidth=1, color=(0.8, 0, 0))  # previous positions
    vertices = [
        (-env.vessel.width/2, -env.vessel.width/2),
        (-env.vessel.width/2, env.vessel.width/2),
        (env.vessel.width/2, env.vessel.width/2),
        (3/2*env.vessel.width, 0),
        (env.vessel.width/2, -env.vessel.width/2),
    ]

    env._viewer2d.draw_shape(vertices, env.vessel.position, env.vessel.heading, color=(0, 0, 0.8))

def _render_interceptions(env):
    for t, obst_intercept_array in enumerate(env.sensor_obst_intercepts_transformed_hist):
        for obst_intercept in obst_intercept_array:
            env._viewer2d.draw_circle(origin=obst_intercept, radius=1.0 - t/len(env.sensor_obst_intercepts_transformed_hist), res=30, color=(0.3, 1.0 - t/len(env.sensor_obst_intercepts_transformed_hist), 0.3))

def _render_sensors(env):
    for isensor, sensor_angle in enumerate(env.vessel._sensor_angles):
        isector = env.config["sector_partition_fun"](env, isensor) # isensor // env.config["n_sensors_per_sector"]
        distance = env.vessel._last_sensor_dist_measurements[isensor]
        p0 = env.vessel.position
        p1 = (
            p0[0] + np.cos(sensor_angle+env.vessel.heading)*distance,
            p0[1] + np.sin(sensor_angle+env.vessel.heading)*distance
        )
        
        closeness = env.vessel._last_sector_dist_measurements[isector]
        redness = 0.5 + 0.5*max(0, closeness)
        greenness = 1 - max(0, closeness)
        blueness = 0.5 if abs(isector - int(np.floor(env.config["n_sectors"]/2) + 1))  % 2 == 0 and not env.config["sensor_rotation"] else 1
        alpha = 0.5
        env._viewer2d.draw_line(p0, p1, color=(redness, greenness, blueness, alpha))

def _render_progress(env):
    ref_point = env.path(env.vessel._last_navi_state_dict['vessel_arclength']).flatten()
    env._viewer2d.draw_circle(origin=ref_point, radius=1, res=30, color=(0.8, 0.3, 0.3))
    
    target_point = env.path(env.vessel._last_navi_state_dict['target_arclength']).flatten()
    env._viewer2d.draw_circle(origin=target_point, radius=1, res=30, color=(0.3, 0.8, 0.3))

def _render_obstacles(env):
    for i, obst in enumerate(env.obstacles):
        c = (0.8, 0.8, 0.8)

        if isinstance(obst, CircularObstacle):
            env._viewer2d.draw_circle(obst.position, obst.radius, color=c)

        elif isinstance(obst, PolygonObstacle):
            env._viewer2d.draw_shape(obst.points, color=c)
        
        elif isinstance(obst, VesselObstacle):
            env._viewer2d.draw_shape(list(obst.boundary.exterior.coords), color=c)
        

def _render_tiles(env, win):
    global env_bg
    global bg

    if env_bg is None:
        # Initialise background
        from pyglet.gl.gl import GLubyte
        data = np.zeros((env_bg_h, env_bg_w, 3))
        k = env_bg_h//100
        for x in range(0, data.shape[0], k):
            for y in range(0, data.shape[1], k):
                data[x:x+k, y:y+k, :] = np.array((
                    int(255*min(1.0, 0.3 + 0.025 * (np.random.random() - 0.5))),
                    int(255*min(1.0, 0.7 + 0.025 * (np.random.random() - 0.5))),
                    int(255*min(1.0, 0.8 + 0.025 * (np.random.random() - 0.5)))
                ))

        pixels = data.flatten().astype('int').tolist()
        raw_data = (GLubyte * len(pixels))(*pixels)
        bg = pyglet.image.ImageData(width=env_bg_w, height=env_bg_h, format='RGB', data=raw_data)
        if not os.path.exists('./resources'):
            os.mkdir('./resources')
        bg.save('./resources/bg.png')
        env_bg = pyglet.sprite.Sprite(bg, x=env.vessel.position[0] - env_bg_w/2, y=env.vessel.position[1] - env_bg_h/2)
        env_bg.scale = 1

    if env.t_step % 250 == 0:
        env_bg = pyglet.sprite.Sprite(bg, x=env.vessel.position[0] - env_bg_w/2, y=env.vessel.position[1] - env_bg_h/2)
        env_bg.scale = 1

    env_bg.draw()

def _render_indicators(env, W, H):

    prog = W/40.0
    h = H/40.0
    gl.glBegin(gl.GL_QUADS)
    gl.glColor4f(0,0,0,1)
    gl.glVertex3f(W, 0, 0)
    gl.glVertex3f(W, 5*h, 0)
    gl.glVertex3f(0, 5*h, 0)
    gl.glVertex3f(0, 0, 0)
    gl.glEnd()
    
    env._viewer2d.reward_text_field.text = "Current Reward:"
    env._viewer2d.reward_text_field.draw()
    env._viewer2d.reward_value_field.text = "{:2.3f}".format(env.last_reward)
    env._viewer2d.reward_value_field.draw()

    env._viewer2d.cum_reward_text_field.text = "Cumulative Reward:"
    env._viewer2d.cum_reward_text_field.draw()
    env._viewer2d.cum_reward_value_field.text = "{:2.3f}".format(env.cumulative_reward)
    env._viewer2d.cum_reward_value_field.draw()

    env._viewer2d.time_step_text_field.text = "Time Step:"
    env._viewer2d.time_step_text_field.draw()
    env._viewer2d.time_step_value_field.text = str(env.t_step)
    env._viewer2d.time_step_value_field.draw()

    env._viewer2d.episode_text_field.text = "Episode:"
    env._viewer2d.episode_text_field.draw()
    env._viewer2d.episode_value_field.text = str(env.episode)
    env._viewer2d.episode_value_field.draw()

    env._viewer2d.lambda_text_field.text = "Log10 Lambda:"
    env._viewer2d.lambda_text_field.draw()
    env._viewer2d.lambda_value_field.text = "{:2.2f}".format(np.log10(env.rewarder.params["lambda"]))
    env._viewer2d.lambda_value_field.draw()

    env._viewer2d.eta_text_field.text = "Eta:"
    env._viewer2d.eta_text_field.draw()
    env._viewer2d.eta_value_field.text = "{:2.2f}".format(env.rewarder.params["eta"])
    env._viewer2d.eta_value_field.draw()

def render_env(env, mode):
    global rot_angle

    def render_objects():
        t = env._viewer2d.transform
        t.enable()
        _render_sensors(env)
        #_render_interceptions(env)
        if env.path is not None:
            _render_path(env)
        _render_vessel(env)
        _render_tiles(env, win)
        _render_obstacles(env)
        if env.path is not None:
            _render_progress(env)
        #_render_interceptions(env)

        # Visualise path error (DEBUGGING)
        # p = np.array(env.vessel.position)
        # dir = rotate(env.past_obs[-1][0:2], env.vessel.heading)
        # env._viewer2d.draw_line(p, p + 10*np.array(dir), color=(0.8, 0.3, 0.3))

        for geom in env._viewer2d.onetime_geoms:
           geom.render()

        t.disable()

        if env.config["show_indicators"]:
            _render_indicators(env, WINDOW_W, WINDOW_H)
            
    scroll_x = env.vessel.position[0]
    scroll_y = env.vessel.position[1]
    ship_angle = -env.vessel.heading + np.pi/2
    if (rot_angle is None):
        rot_angle = ship_angle
    else:
        rot_angle += CAMERA_ROTATION_SPEED * geom.princip(ship_angle - rot_angle)

    if DYNAMIC_ZOOM:
        if (int(env.t_step/1000) % 2 == 0):
            env._viewer2d.camera_zoom = 0.999*env._viewer2d.camera_zoom + 0.001*(ZOOM - env._viewer2d.camera_zoom)
        else:
            env._viewer2d.camera_zoom = 0.999*env._viewer2d.camera_zoom + 0.001*(1 - env._viewer2d.camera_zoom)

    env._viewer2d.transform.set_scale(env._viewer2d.camera_zoom, env._viewer2d.camera_zoom)
    env._viewer2d.transform.set_translation(
        WINDOW_W/2 - (scroll_x*env._viewer2d.camera_zoom*cos(rot_angle) - scroll_y*env._viewer2d.camera_zoom*sin(rot_angle)),
        WINDOW_H/2 - (scroll_x*env._viewer2d.camera_zoom*sin(rot_angle) + scroll_y*env._viewer2d.camera_zoom*cos(rot_angle))
    )
    env._viewer2d.transform.set_rotation(rot_angle)

    win = env._viewer2d.window
    win.switch_to()
    x = win.dispatch_events()
    win.clear()
    gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
    render_objects()
    arr = None

    if mode == 'rgb_array':
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(WINDOW_H, WINDOW_W, 4)
        arr = arr[::-1, :, 0:3]

    win.flip()

    env._viewer2d.onetime_geoms = []

    return arr

def init_env_viewer(env):
    env._viewer2d = Viewer2D(WINDOW_W, WINDOW_H)

    env._viewer2d.reward_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 30.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.reward_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 30.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.cum_reward_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 50.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.cum_reward_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 50.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.time_step_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 70.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.time_step_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 70.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.episode_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 90.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.episode_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 90.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.lambda_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 110.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.lambda_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 110.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.eta_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 130.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.eta_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 130.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    print('Initialized 2D viewer')
