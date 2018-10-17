#!/usr/bin/env python
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def Rot2Angle(R):
    assert(isRotationMatrix(R))
    
    angle_axis= numpy.zeros(3)
    angle_axis[0] = R[2,1] - R[1,2]
    angle_axis[1] = R[0,2] - R[2,0]
    angle_axis[2] = R[1,0] - R[0,1]
    
    costheta = min(max((R[0,0] + R[1, 1] + R[2,2] - 1.0) / 2.0,  -1.0), 1.0)
    sintheta = min(sqrt(angle_axis[0] * angle_axis[0] + angle_axis[1] * angle_axis[1] + angle_axis[2] * angle_axis[2] )/ 2.0, 1.0)
    theta = atan2(sintheta, costheta)
    
    kThreshold = 1e-12
    if ((sintheta > kThreshold) or (sintheta < -kThreshold)):
	r = theta / (2.0 * sintheta)
	angle_axis = angle_axis * r
	return angle_axis

    if (costheta > 0.0):
	angle_axis = angle_axis * 0.5
	return angle_axis

    inv_one_minus_costheta = 1.0 / (1.0 - costheta)

    for i in range(0,3):
	angle_axis[i] = theta * sqrt((R[i, i] - costheta) * inv_one_minus_costheta)
        if (((sintheta < 0.0) and (angle_axis[i] > 0.0)) or ((sintheta > 0.0) and (angle_axis[i] < 0.0))):
	    angle_axis[i] = -angle_axis[i]

    return angle_axis
