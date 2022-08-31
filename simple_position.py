import math
import mujoco
from mujoco_viewer import MujocoViewer
from spd_utils import populate_show_actuator_forces, show_actuator_forces

model = mujoco.MjModel.from_xml_path("pendulum_simple_position.xml")
data = mujoco.MjData(model)
viewer = MujocoViewer(model, data)

viewer.add_graph_line(line_name="sine_wave", line_data=1.0)
viewer.add_graph_line(line_name="position_sensor", line_data=1.0)
viewer.add_graph_line(line_name="force", line_data=0.0)
viewer.add_graph_line(line_name="joint_pos_error", line_data=0.0)
x_div = 10
y_div = 10
viewer.set_grid_divisions(x_div=x_div, y_div=y_div, x_axis_time=0.5)
viewer.show_graph_legend()

viewer.callbacks._paused = True
t = 0

general_kp = 10000 * 1
general_kd = general_kp * model.opt.timestep * 1.9

rendered_axes, f_render, f_list = populate_show_actuator_forces(
    model=model,
    to_be_rendered_axes=[
        "hinge_1",
        "hinge_2",
        "hinge_3",
    ],
)

while True:
    target_pos = math.sin(t * 0.0001 * (180 / math.pi))
    next_pos = math.sin((t + 1) * 0.0001 * (180 / math.pi))
    target_vel = next_pos - target_pos / model.opt.timestep
    error = target_pos - data.qpos[0].tolist()

    show_actuator_forces(
        viewer=viewer,
        data=data,
        rendered_axes=rendered_axes,
        f_render=f_render,
        f_list=f_list,
    )

    t = t + 1
    data.ctrl = [target_pos, -target_pos, target_pos]

    mujoco.mj_step(model, data)
    viewer.render()

    viewer.update_graph_line(
        line_name="sine_wave",
        line_data=target_pos,
    )
    viewer.update_graph_line(
        line_name="position_sensor",
        line_data=data.qpos[0],
    )
    viewer.update_graph_line(
        line_name="force",
        line_data=data.actuator_force[0],
    )
    viewer.update_graph_line(
        line_name="joint_pos_error",
        line_data=error,
    )
