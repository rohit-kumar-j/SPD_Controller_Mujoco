from asyncore import read
import math
from xml.dom.minidom import Element
import mujoco
import numpy as np
from mujoco_viewer import MujocoViewer

# pybullet gui as a replacement for Mj_UI Elements // uncomment for tuning
# import pybullet as p

from final_spd_utils import (
    computePD,
    populate_show_actuator_forces,
    show_actuator_forces,
)

# client = p.connect(p.GUI)

model = mujoco.MjModel.from_xml_path("pendulum_free_spd.xml")
data = mujoco.MjData(model)
viewer = MujocoViewer(model, data)

viewer.add_graph_line(line_name="force", line_data=0.0)
viewer.add_graph_line(line_name="joint_sensor", line_data=0.0)
viewer.add_graph_line(line_name="sine", line_data=0.0)
x_div = 10
y_div = 10
viewer.set_grid_divisions(x_div=x_div, y_div=y_div, x_axis_time=1)
viewer.show_graph_legend()

viewer.callbacks._paused = True

general_kp = 10e2
general_kd = general_kp * model.opt.timestep * 1.2

rendered_axes, f_render, f_list = populate_show_actuator_forces(
    model=model,
    to_be_rendered_axes=[
        "hinge_1",
        "hinge_2",
        "hinge_3",
        "hinge_3_1",
        "hinge_2_1",
    ],
)

h1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_1")
h2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_2")
h3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_3")

hz = 3  # sine wave oscillation frequency

time_period = 1.0 / hz

time_steps_per_second = 1 / model.opt.timestep
divisions_per_sine_wave = time_steps_per_second / hz

print(f"divisions_per_sine_wave: {divisions_per_sine_wave}")
print(f"time_period: {time_period}")

# read_hz = p.addUserDebugParameter("hz", 0, 10, hz)
# read_kp = p.addUserDebugParameter("Kp", 0, 10e4, 10e3)
# read_kd = p.addUserDebugParameter(
#     "Kd", 0, 10e2, 200
# )
# read_kp2 = p.addUserDebugParameter("Kp2", 0, 200, 100)
# read_kd2 = p.addUserDebugParameter("Kd2", 0, 20, 10)

old_time = 0
t = 0
while True:
    # write_hz = p.readUserDebugParameter(read_hz)
    # write_kp = p.readUserDebugParameter(read_kp)
    # write_kd = p.readUserDebugParameter(read_kd)
    # write_kp2 = p.readUserDebugParameter(read_kp2)
    # write_kd2 = p.readUserDebugParameter(read_kd2)
    target_pos = math.sin(math.pi * t * hz * model.opt.timestep)
    next_pos = math.sin(math.pi * (t + 1) * hz * model.opt.timestep)
    target_vel = next_pos - target_pos / model.opt.timestep
    error = target_pos - data.joint("hinge_1").qpos

    spd_forces = computePD(
        model=model,
        data=data,
        controlled_joint_ids=[
            "hinge_1",
            "hinge_2",
            "hinge_3",
            "hinge_2_1",
            "hinge_3_1",
        ],
        desiredPositions=[
            target_pos,
            -target_pos,
            target_pos,
            target_pos,
            -target_pos,
        ],
        desiredVelocities=[
            target_vel,
            -target_vel,
            target_vel,
            target_vel,
            -target_vel,
        ],
        kps=[
            10e3,  # ,write_kp,
            10e3,  # ,write_kp,
            10e3,  # ,write_kp,
            100,  # write_kp2,
            100,  # write_kp2,
        ],  # * model.nu,
        kds=[
            200,  # write_kd,
            200,  # write_kd,
            200,  # write_kd,
            10,  # write_kd2,
            10,  # write_kd2,
        ],  # * model.nu,
        maxForces=[10e4] * model.nu,
        timeStep=model.opt.timestep,
    )

    show_actuator_forces(
        viewer=viewer,
        data=data,
        rendered_axes=rendered_axes,
        f_render=f_render,
        f_list=f_list,
        show_force_labels=True,
    )

    t = t + 1
    # print(
    #     "data.data.actuator_force: ",
    #     data.actuator_force,
    #     "spd_forces:",
    #     spd_forces,
    # )
    data.ctrl = spd_forces

    viewer.update_graph_line(
        line_name="force",
        line_data=spd_forces[4] / 100,
    )
    viewer.update_graph_line(
        line_name="joint_sensor",
        line_data=-data.joint("hinge_3_1").qpos,
    )
    viewer.update_graph_line(
        line_name="sine",
        line_data=target_pos,
    )

    # curr_time = data.time
    # if curr_time - old_time >= 1:
    #     old_time = curr_time
    #     viewer.callbacks._paused = True

    mujoco.mj_step(model, data)
    viewer.render()
