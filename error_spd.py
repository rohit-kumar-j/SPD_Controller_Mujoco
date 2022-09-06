import math
import mujoco
from mujoco_viewer import MujocoViewer
from error_spd_utils import (
    computePD,
    populate_show_actuator_forces,
    show_actuator_forces,
)

model = mujoco.MjModel.from_xml_path("pendulum_simple_position.xml")
data = mujoco.MjData(model)
viewer = MujocoViewer(model, data)

# viewer.add_graph_line(line_name="sine_wave", line_data=0.0)
# viewer.add_graph_line(line_name="position_sensor", line_data=0.0)
viewer.add_graph_line(line_name="force", line_data=0.0)
viewer.add_graph_line(line_name="joint_pos_error", line_data=0.0)
x_div = 10
y_div = 10
viewer.set_grid_divisions(x_div=x_div, y_div=y_div, x_axis_time=0.5)
viewer.show_graph_legend()

viewer.callbacks._paused = True
t = 0

general_kp = 10000 * 2
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

    l1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
    b_jadr_l1 = model.body_jntadr[l1]
    qposadr = model.jnt_qposadr[b_jadr_l1]
    qveladr = model.jnt_dofadr[b_jadr_l1]
    print(f"l1_bodyid: {l1}")
    print(f"l1_model.body_jntadr: {model.body_jntadr}")
    print(f"l1_b_jadr: {b_jadr_l1}")
    print(f"qposadr: {qposadr}")
    print(f"qveladr: {qveladr}\n")

    l2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_2")
    b_jadr_l2 = model.body_jntadr[l2]
    qposadr = model.jnt_qposadr[b_jadr_l2]
    qveladr = model.jnt_dofadr[b_jadr_l2]
    print(f"l2_bodyid: {l2}")
    print(f"l2_model.body_jntadr: {model.body_jntadr}")
    print(f"l2_b_jadr: {b_jadr_l2}")
    print(f"qposadr: {qposadr}")
    print(f"qveladr: {qveladr}\n")

    l3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_3")
    b_jadr_l3 = model.body_jntadr[l3]
    qposadr = model.jnt_qposadr[b_jadr_l3]
    qveladr = model.jnt_dofadr[b_jadr_l3]
    print(f"l3_bodyid: {l3}")
    print(f"l3_model.body_jntadr: {model.body_jntadr}")
    print(f"l3_b_jadr: {b_jadr_l3}")
    print(f"qposadr: {qposadr}")
    print(f"qveladr: {qveladr}\n")

    fb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "my_floating_body")
    b_jadr_fb = model.body_jntadr[fb]
    fb_qposadr = model.jnt_qposadr[b_jadr_fb]
    fb_qveladr = model.jnt_dofadr[b_jadr_fb]
    print(f"fb_bodyid: {fb}")
    print(f"fb_model.body_jntadr: {model.body_jntadr}")
    print(f"fb_b_jadr: {b_jadr_fb}")
    print(f"fb_qposadr: {fb_qposadr}")
    print(f"fb_qveladr: {fb_qveladr}\n")

    print(f"data.qpos: {data.qpos}")
    print(f"data.qpos[fb_qposadr]: {data.qpos[fb_qposadr]}")
    print(f"data.qvel: {data.qvel}")
    print(f"data.qvel[qposadr]: {data.qvel[qposadr]}\n")

    spd_forces = computePD(
        model=model,
        data=data,
        desiredPositions=[target_pos, -target_pos, target_pos],
        desiredVelocities=[0] * model.nu,
        kps=[general_kp] * model.nu,
        kds=[general_kd] * model.nu,
        maxForces=[1000] * model.nu,
        timeStep=model.opt.timestep,
        body_name="link_1",  # ,"my_floating_body",
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
    print("data.ctrl: ", data.ctrl, "spd_forces:", spd_forces)
    data.ctrl = spd_forces

    mujoco.mj_step(model, data)
    viewer.render()

    # viewer.update_graph_line(
    #     line_name="sine_wave",
    #     line_data=target_pos,
    # )
    # viewer.update_graph_line(
    #     line_name="position_sensor",
    #     line_data=data.qpos[0],
    # )
    viewer.update_graph_line(
        line_name="force",
        line_data=spd_forces[0],
    )
    viewer.update_graph_line(
        line_name="joint_pos_error",
        line_data=error,
    )
