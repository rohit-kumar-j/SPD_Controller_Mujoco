import mujoco
import glfw
import sys
import numpy as np
import time
from callbacks import CallBacks


class MujocoViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self._time_per_render = 1 / 60.0
        self._loop_count = 0
        self._fastmode = False
        # Graphing
        self._num_pnts = 100

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig)

        for n in range(0, len(self.model.sensor_adr) * 3):
            for i in range(0, 300):
                self.fig.linedata[n][2 * i] = float(-i)

        # get callbacks
        self.callbacks = CallBacks(
            model=self.model,
            data=self.data,
            vopt=self.vopt,
            scn=self.scn,
            cam=self.cam,
            pert=self.pert,
        )

        # Adjust placement and size of graph
        width, height = self.callbacks.get_viewport_width_height()
        width_adjustment = width % 4
        self.pid_viewport = mujoco.MjrRect(
            int(3 * width / 4) + width_adjustment,
            0,
            int(width / 4),
            int(height / 4),
        )
        mujoco.mjr_figure(self.pid_viewport, self.fig, self.callbacks.ctx)

        self.fig.flg_extend = 1
        self.fig.flg_symmetric = 0

        # overlay, markers
        self._overlay = {}
        self._markers = []
        self._data_graph_line_names = []
        self._line_datas = []
        self._sensor_data = "degree"

    def prepare(self):
        self.callbacks._joints = True
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.callbacks._joints
        self.callbacks._convex_hull_rendering = True
        self.vopt.flags[
            mujoco.mjtVisFlag.mjVIS_CONVEXHULL
        ] = self.callbacks._convex_hull_rendering
        self.callbacks._wire_frame = True
        self.scn.flags[
            mujoco.mjtRndFlag.mjRND_WIREFRAME
        ] = self.callbacks._wire_frame
        self.callbacks._shadows = False
        self.scn.flags[
            mujoco.mjtRndFlag.mjRND_SHADOW
        ] = self.callbacks._shadows

    def fastmode(self, fast: bool = True) -> bool:
        if fast is not True:
            self._fastmode = False
        else:
            self._fastmode = True
        return self._fastmode

    def get_arrow_state(self):
        return (
            self.callbacks._key_up,
            self.callbacks._key_down,
            self.callbacks._key_right,
            self.callbacks._key_left,
        )

    def clean_screen(self):
        self.callbacks._hide_menu = True
        self.callbacks._hide_graph = True

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                "Ran out of geoms. maxgeom: %d" % self.scn.maxgeom
            )

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        if self.callbacks._render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self.callbacks._run_speed,
                "[S]lower, [F]aster",
            )
        add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if self.callbacks._render_every_frame else "Off",
        )
        add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        add_overlay(
            topleft,
            "[C]ontact forces",
            "On" if self.callbacks._contacts else "Off",
        )
        add_overlay(
            topleft, "[J]oints", "On" if self.callbacks._joints else "Off"
        )
        add_overlay(
            topleft, "[I]nertia", "On" if self.callbacks._inertias else "Off"
        )
        add_overlay(
            topleft,
            "Toggle [G]raph overlay",
            "On" if self.callbacks._hide_graph else "Off",
        )
        add_overlay(
            topleft, "Center of [M]ass", "On" if self.callbacks._com else "Off"
        )
        add_overlay(
            topleft, "Shad[O]ws", "On" if self.callbacks._shadows else "Off"
        )
        add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if self.callbacks._transparent else "Off",
        )
        add_overlay(
            topleft,
            "[W]ireframe",
            "On" if self.callbacks._wire_frame else "Off",
        )
        add_overlay(
            topleft,
            "Con[V]ex Hull Rendering",
            "On" if self.callbacks._convex_hull_rendering else "Off",
        )
        if self.callbacks._paused is not None:
            if not self.callbacks._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(
                    topleft, "Advance simulation by one step", "[right arrow]"
                )
        add_overlay(
            topleft,
            "Referenc[e] frames",
            "On" if self.vopt.frame == 1 else "Off",
        )
        add_overlay(topleft, "[H]ide Menu", "")
        if self.callbacks._image_idx > 0:
            fname = self.callbacks._image_path % (
                self.callbacks._image_idx - 1
            )
            add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")
        add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        add_overlay(
            bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, "")
        )
        add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
        )
        add_overlay(
            bottomleft,
            "Step",
            str(round(self.data.time / self.model.opt.timestep)),
        )
        add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)
        # CUSTOM
        add_overlay(
            topright,
            "Arrow_up: Previous joint",
            "On" if self.callbacks._key_up is True else "Off",
        )
        add_overlay(
            topright,
            "Arrow_down: Previous joint",
            "On" if self.callbacks._key_down is True else "Off",
        )
        add_overlay(
            topright,
            "Arrow_left: Previous joint",
            "On" if self.callbacks._key_left is True else "Off",
        )
        add_overlay(
            topright,
            "Arrow_right: Previous joint",
            "On" if self.callbacks._key_right is True else "Off",
        )

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def add_sensor(self, sensor):
        """
        sensor: "all" or 0,1,2...
        """
        if sensor == "all":
            self.adrs = self.model.sensor_adr
        else:
            self.adrs = []
            for i in range(len(sensor)):
                try:
                    self.adrs.append(self.model.sensor_adr[sensor[i]])
                except Exception as e:
                    raise ValueError("Sensor is not found in XML file")
        # print(self.adrs)

    def axis_autorange(self):
        self.set_axis_range(x_range=[1.0, -1.0], y_range=[1.0, -1.0])

    def set_axis_range(self, x_range, y_range):
        """
        x_range : list [x_min, x_max]
        y_range : list [y_min, y_max]
        """
        assert (
            type(x_range) == list
        ), "x_range is not a list with x_min and x_max"
        assert (
            type(y_range) == list
        ), "y_range is not a list with y_min and y_max"
        if x_range[0] < x_range[1]:
            self.fig.range[0][0] = x_range[0]  # x_min
            self.fig.range[0][1] = x_range[1]  # x_max
        elif x_range[0] > x_range[1]:
            # limits set to auto range since {x_min > x_max}
            self.fig.range[0][0] = x_range[1]
            self.fig.range[0][1] = x_range[0]

        elif y_range[0] < y_range[1]:
            self.fig.range[1][0] = y_range[0]  # y_min
            self.fig.range[1][1] = y_range[1]  # y_max
        elif y_range[0] > y_range[1]:
            # limits set to auto range since {y_min > y_max}
            self.fig.range[1][0] = y_range[1]
            self.fig.range[1][1] = y_range[0]

    def set_grid_divisions(
        self, x_div: int, y_div: int, x_axis_time: float = 0.0
    ):
        self.fig.gridsize[0] = x_div + 1
        self.fig.gridsize[1] = y_div + 1
        if x_axis_time is not 0.0:
            self._num_pnts = x_axis_time / self.model.opt.timestep
            print("self._num_pnts: ", self._num_pnts)
            if self._num_pnts > 300:
                self._num_pnts = 300
                new_x_axis_time = self.model.opt.timestep * self._num_pnts
                print(
                    f"Minimum x_axis_time is: {new_x_axis_time}"
                    + " reduce the x_axis_time"
                    f" OR Maximum time_step is: "
                    + f"{self.model.opt.timestep*self._num_pnts}"
                    + " increase the timestep"
                )
                # assert x_axis_time ==
            assert 1 <= self._num_pnts <= 300, (
                "num_pnts should be [10,300], it is currently:",
                f"{self._num_pnts}",
            )
            # self._num_pnts = num_pnts
            self._time_per_div = (self.model.opt.timestep * self._num_pnts) / (
                x_div
            )
            self.set_x_label(
                xname=f"time/div: {self._time_per_div}s"
                + f" total: {self.model.opt.timestep * self._num_pnts}"
            )

    def set_graph_name(self, name: str):
        assert type(name) == str, "name is not a string"
        self.fig.title = name

    def show_graph_legend(self, show_legend: bool = True):
        if show_legend is True:
            for i in range(0, len(self._data_graph_line_names)):
                self.fig.linename[i] = self._data_graph_line_names[i]
            self.fig.flg_legend = True

    def set_x_label(self, xname: str):
        assert type(xname) == str, "xname is not a string"
        self.fig.xlabel = xname

    def add_graph_line(self, line_name, line_data):
        assert (
            type(line_name) == str
        ), f"Line_name is not a string: {type(line_name)}"
        if line_name in self._data_graph_line_names:
            print("line name already exists")
        else:
            self._data_graph_line_names.append(line_name)
            self._line_datas.append(line_data)

    def update_graph_line(self, line_name, line_data):
        if line_name in self._data_graph_line_names:
            idx = self._data_graph_line_names.index(line_name)
            self._line_datas[idx] = line_data
        else:
            raise NameError(
                "line name is not valid, add it to list before calling update"
            )

    def sensorupdate(self):
        # print(self._line_datas)
        pnt = int(mujoco.mju_min(self._num_pnts, self.fig.linepnt[0] + 1))
        # print(self.fig.linepnt[0] + 1)
        for n in range(0, len(self._line_datas)):
            for i in range(pnt - 1, 0, -1):
                self.fig.linedata[n][2 * i + 1] = self.fig.linedata[n][
                    2 * i - 1
                ]
            self.fig.linepnt[n] = pnt
            self.fig.linedata[n][1] = self._line_datas[n]

    def set_graph_units(self, type):
        if type == "radian":
            self._sensor_data == "radian"
        if type == "degree":
            self._sensor_data == "degree"

    def update_graph_size(self, size_div_x=None, size_div_y=None):
        if size_div_x is None and size_div_y is None:
            width, height = self.callbacks.get_viewport_width_height()
            width_adjustment = width % 3
            self.pid_viewport.left = int(2 * width / 3) + width_adjustment
            self.pid_viewport.width = int(width / 3)
            self.pid_viewport.height = int(height / 3)

        else:
            assert size_div_x is not None and size_div_y is None, ""
            width, height = self.callbacks.get_viewport_width_height()
            width_adjustment = width % size_div_x
            self.pid_viewport.left = (
                int((size_div_x - 1) * width / size_div_x) + width_adjustment
            )
            self.pid_viewport.width = int(width / size_div_x)
            self.pid_viewport.height = int(height / size_div_x)

    def render(self):
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.callbacks.window is None:
                return
            elif glfw.window_should_close(self.callbacks.window):
                glfw.terminate()
                sys.exit(0)
            (
                self.callbacks.viewport.width,
                self.callbacks.viewport.height,
            ) = glfw.get_framebuffer_size(self.callbacks.window)
            with self.callbacks._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(
                    self.callbacks.viewport, self.scn, self.callbacks.ctx
                )
                # overlay items
                if not self.callbacks._hide_menu:
                    for gridpos, [t1, t2] in self._overlay.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.callbacks.viewport,
                            t1,
                            t2,
                            self.callbacks.ctx,
                        )
                # Handle graph and pausing interactions
                if (
                    not self.callbacks._hide_graph
                    and not self.callbacks._paused
                ):
                    self.sensorupdate()
                    self.update_graph_size()
                    mujoco.mjr_figure(
                        self.pid_viewport, self.fig, self.callbacks.ctx
                    )
                elif self.callbacks._hide_graph and self.callbacks._paused:
                    self.update_graph_size()
                elif not self.callbacks._hide_graph and self.callbacks._paused:
                    mujoco.mjr_figure(
                        self.pid_viewport, self.fig, self.callbacks.ctx
                    )
                elif self.callbacks._hide_graph and not self.callbacks._paused:
                    self.sensorupdate()
                    self.update_graph_size()

                self.callbacks._swap_buffers()
            self.callbacks._poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            # clear overlay
            self._overlay.clear()

        if self.callbacks._paused:
            while self.callbacks._paused:
                update()
                if self.callbacks._advance_by_one_step:
                    self.callbacks._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self.callbacks._run_speed
            )
            if self._fastmode is True:
                if not self.callbacks._render_every_frame:
                    self._loop_count = 1
                while self._loop_count > 0:
                    update()
                    self._loop_count -= 1
            else:
                if self.callbacks._render_every_frame:
                    self._loop_count = 1
                while self._loop_count > 0:
                    update()
                    self._loop_count -= 1

        # clear markers
        self._markers[:] = []

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def close(self):
        self.callbacks.close()
