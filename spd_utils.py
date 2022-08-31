import mujoco
import numpy as np


def computePD(
    model,
    data,
    desiredPositions,
    desiredVelocities,
    kps,
    kds,
    maxForces,
    timeStep,
):
    q = data.qpos
    qdot = data.qvel
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

    # print(f"q: {q},np.shape(q):{np.shape(q)}")
    # print(f"q_des: {q_des},np.shape(q_des):{np.shape(q_des)}")
    # print(f"qdot: {qdot},np.shape(qdot):{np.shape(qdot)}")
    # print(f"qdot_des: {qdot_des},np.shape(qdot_des):{np.shape(qdot_des)}")
    # print(f"Kp: {Kp},np.shape(Kp):{np.shape(Kp)}")
    # print(f"Kd: {Kd},np.shape(Kd):{np.shape(Kd)}")
    # print(
    #     f"MassMatrix: {MassMatrix},np.shape(MassMatrix):{np.shape(MassMatrix)}"
    # )
    # print(
    #     f"Bias_Forces: {Bias_Forces},np.shape(Bias_Forces):{np.shape(Bias_Forces)}"
    # )

    qError = q_des - q
    qdotError = qdot_des - qdot

    # Compute -Kp(q + qdot - qdes)
    # p_term = Kp.dot(qError - qdot * timeStep)

    # Compute -Kd(qdot - qdotdes)
    # d_term = Kd.dot(qdotError)

    qddot = np.linalg.solve(
        a=(MassMatrix + Kd * timeStep),
        b=(
            -Bias_Forces + Kp.dot(qError - qdot * timeStep) + Kd.dot(qdotError)
        ),
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
