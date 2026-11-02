"""Public controller proprioception only; no object state or teleportation."""
import numpy as np
def pose(controller):
    # Read only the robot control site's public proprioceptive pose. Calling
    # controller.update(force=True) here would additionally execute sim.forward
    # and could change simulator/controller warm-start state in a paired test.
    return (np.asarray(controller.sim.data.get_site_xpos(controller.eef_name)).copy(),
            np.asarray(controller.sim.data.get_site_xmat(controller.eef_name)).reshape(3,3).copy())
def servo(controller,target_position,target_rotation,details=False):
    from robosuite.utils.control_utils import orientation_error
    position,rotation=pose(controller)
    output=np.concatenate([.5*(np.asarray(target_position)-position),.5*orientation_error(np.asarray(target_rotation),rotation)])
    imin=np.asarray(controller.input_min);imax=np.asarray(controller.input_max);omin=np.asarray(controller.output_min);omax=np.asarray(controller.output_max)
    assert imin.shape==imax.shape==omin.shape==omax.shape==(6,)
    scale=np.abs(omax-omin)/np.abs(imax-imin);assert np.all(scale>0)
    action=(np.clip(output,omin,omax)-(omax+omin)/2)/scale+(imax+imin)/2
    action=np.concatenate([np.clip(action,imin,imax),[-1.]]).astype(np.float32)
    if details:
        return action,dict(public_robot_site_position=position.tolist(),public_robot_site_rotation=rotation.tolist(),target_position=np.asarray(target_position).tolist(),target_rotation=np.asarray(target_rotation).tolist(),input_min=imin.tolist(),input_max=imax.tolist(),output_min=omin.tolist(),output_max=omax.tolist())
    return action
def home_target(home_position,trigger_position,tick):
    assert 0<=tick<60
    safe_z=max(float(home_position[2]),float(trigger_position[2])+.12)
    if tick<20:return np.asarray([trigger_position[0],trigger_position[1],safe_z])
    if tick<40:return np.asarray([home_position[0],home_position[1],safe_z])
    return np.asarray(home_position).copy()
