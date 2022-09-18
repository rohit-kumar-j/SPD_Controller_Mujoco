import mujoco
import numpy as np


def get_all_joint_indices(model, jnt_name):
    jnt_id = model.joint(jnt_name).dofadr[0]
    # jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    # print(f"jnt_id: {jnt_id}")

    adr_start = model.jnt_dofadr[jnt_id]
    vel_adr_start = model.jnt_qposadr[jnt_id]
    # print(f"adr_start: {adr_start}")

    # print(f"model.jnt_type: {model.jnt_type}")

    # assuming only ball, hinge or slide is used
    # Types of free, ball, slide, hinge: 0, 1, 2, 3
    if model.jnt_type[jnt_id] == 0:
        n_dofs = 6
        # print("free")
    elif model.jnt_type[jnt_id] == 1:
        n_dofs = 3
        # print("ball")
    elif model.jnt_type[jnt_id] == 2:
        n_dofs = 1
        # print("slide")
    elif model.jnt_type[jnt_id] == 3:
        n_dofs = 1
        # print("hinge")
    # n_dofs = 1 if model.jnt_type[jnt_id] > 1 else 3
    return np.arange(adr_start, adr_start + n_dofs)


def computePD(
    model,
    data,
    controlled_joint_ids,
    desiredPositions,
    desiredVelocities,
    kps,
    kds,
    maxForces,
    timeStep,
):
    # decide length of q and qdot
    q = np.empty(
        [
            len(desiredPositions),
        ]
    )
    qdot = np.empty(
        [
            len(desiredVelocities),
        ]
    )
    t = 0
    for i in controlled_joint_ids:
        jnt_id = model.joint(i).dofadr[0]

        # free joint
        if model.jnt_type[jnt_id] == 0:
            q[t] = data.joint(jnt_id).qpos[0]
            q[t + 1] = data.joint(jnt_id).qpos[1]
            q[t + 2] = data.joint(jnt_id).qpos[2]
            q[t + 3] = data.joint(jnt_id).qpos[3]
            q[t + 4] = data.joint(jnt_id).qpos[4]
            q[t + 5] = data.joint(jnt_id).qpos[5]
            t = t + 6

        # ball joint
        elif model.jnt_type[jnt_id] == 1:
            q[t] = data.joint(jnt_id).qpos[0]
            q[t + 1] = data.joint(jnt_id).qpos[1]
            q[t + 2] = data.joint(jnt_id).qpos[2]
            t = t + 3

        # slider joint
        elif model.jnt_type[jnt_id] == 2:
            q[t] = data.joint(jnt_id).qpos
            t = t + 1

        # hinge joint
        elif model.jnt_type[jnt_id] == 3:
            q[t] = data.joint(jnt_id).qpos
            t = t + 1

    print(f"q: {q}")

    for i in range(0, len(controlled_joint_ids)):
        qdot[i] = data.joint(controlled_joint_ids[i]).qvel

    q_des = np.array(desiredPositions)
    qdot_des = np.array(desiredVelocities)

    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)

    # Create empty mass matrix
    MassMatrix = np.empty(
        shape=(model.nv, model.nv),
        dtype=np.float64,
    )

    # Creates in self._data.crb
    mujoco.mj_crb(model, data)

    mujoco.mj_fullM(
        model,
        MassMatrix,
        data.qM,
    )

    Bias_Forces = data.qfrc_bias

    dof_indices = []
    for i in controlled_joint_ids:
        jnt_id = model.joint(i).dofadr[0]
        adr_start = model.jnt_dofadr[jnt_id]

        if model.jnt_type[jnt_id] == 0:
            n_dofs = 6
            print("free")
        elif model.jnt_type[jnt_id] == 1:
            n_dofs = 3
            print("ball")
        elif model.jnt_type[jnt_id] == 2:
            n_dofs = 1
            print("slide")
        elif model.jnt_type[jnt_id] == 3:
            n_dofs = 1
            print("hinge")

        np.arange(adr_start, adr_start + n_dofs)
        dof_indices.append(model.joint(i).dofadr[0])

        print(f"dof_indices: {dof_indices}")

    qError = q_des - q
    qdotError = qdot_des - qdot

    # Compute -Kp(q + qdot - qdes)
    p_term = Kp.dot(qError - qdot * timeStep)

    # Compute -Kd(qdot - qdotdes)
    d_term = Kd.dot(qdotError)

    qddot = np.linalg.solve(
        a=(MassMatrix[dof_indices, :][:, dof_indices] + Kd * timeStep),
        b=(-Bias_Forces[dof_indices] + p_term + d_term),
    )

    tau = (
        Kp.dot(qError - qdot * timeStep)
        + Kd.dot(qdotError)
        - (Kd.dot(qddot) * timeStep)
    )

    # Clip generalized forces to actuator limits
    maxF = np.array(maxForces)
    generalized_forces = np.clip(tau, -maxF, maxF)
    return generalized_forces


def show_actuator_forces(
    viewer,
    data,
    rendered_axes,
    f_render,
    f_list,
    show_force_labels=False,
) -> None:
    if show_force_labels is False:
        label = ""
        for i in rendered_axes:
            viewer.add_marker(
                pos=data.site(f_render[i][1]).xpos,
                mat=data.site(f_render[i][1]).xmat,
                size=[
                    0.02,
                    0.02,
                    (data.actuator_force[f_list[i]] / f_render[i][2]),
                ],
                rgba=f_render[i][3],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                label=label,
            )
    else:
        for i in rendered_axes:
            viewer.add_marker(
                pos=data.site(f_render[i][1]).xpos,
                mat=data.site(f_render[i][1]).xmat,
                size=[
                    0.02,
                    0.02,
                    (data.actuator_force[f_list[i]] / f_render[i][2]),
                ],
                rgba=f_render[i][3],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                label=str(data.actuator_force[f_list[i]]),
            )


def populate_show_actuator_forces(model, to_be_rendered_axes) -> None:
    """
    format :
        self._f_render = {
            "axis_name": ["act_name","geom_for_force_render","scaling"]
        }

        self._f_list = {
            "axis_name": ["actuator_index"], # internally generated
        }
    """
    rendered_axes = to_be_rendered_axes

    f_render = {
        "hinge_1": [
            "pos_servo_1",
            "site_1",
            20.0,
            [1, 0, 1, 0.2],
        ],
        "hinge_2": [
            "pos_servo_2",
            "site_2",
            20.0,
            [1, 0, 1, 0.2],
        ],
        "hinge_3": [
            "pos_servo_3",
            "site_3",
            20.0,
            [1, 0, 1, 0.2],
        ],
    }
    f_list_keys = []
    f_list_values = []
    for key in rendered_axes:
        values = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f_render[key][0],
        )
        # print("values:", values)
        f_list_keys.append(key)
        f_list_values.append(values)
    f_list = dict(zip(f_list_keys, f_list_values))
    # print("self._f_list:", f_list)

    return rendered_axes, f_render, f_list
