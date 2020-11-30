import numpy as np

class StreamLine:
    """Class used to represent a stream line and compute its caracteristics.

    The stream line is represented by a serie of points, using spherical
    coordinates (longitude, latitude). All the caracteristics of this stream
    line are computed from this list.

    Args:
        coord_list (array_like) : List of coordinates representing the stream
            line.
        delta_time (array_like or float) : Delta time between 2 consecutive
            points. If dt is a constant delta_time should be a float and an
            array_like with len = n-2 otherwise, n beeing the number of points.

    Attributes:
        coord_list (array_like) : List of coordinates representing the stream
            line. 
        nb_points (int) : Number of points in coord_list.
        length (float) : Total length of the line.
        mean_pos (ndarray (2,)) : Average of the coordinates. If the stream
            line is in a eddy, this is an approximation of the eddy center
            coordinates. 
        winding_angle (float) : Sum of the oriented angles formed by 3
            consecutive points in the line. Unit is the radian
        angular_velocities (np.array) : List of the oriented angular velocities
            between 3 consecutive points in the line. Unit is the rad.s**-1

    """
    R = 6378.137e3

    def __init__(self, coord_list, delta_time):
        self.coord_list = np.array(coord_list,dtype=float)
        self.nb_points = len(coord_list)
        self._set_length()
        self.mean_pos = np.mean(coord_list,axis=0)
        self.delta_time = delta_time
        self._set_winding_angle_and_angular_velocity(delta_time)

    def _set_length(self):
        """ Compute the total length of the line. """
        if self.nb_points<=1:
            self.length = 0
        else:
            ldiff_degree     = self.coord_list[1:]-self.coord_list[:-1]
            ldiff_meter      = ldiff_degree
            ldiff_meter     *= np.pi*self.R/180
            ldiff_meter[:,0]*= np.sin(self.coord_list[:-1,1]*np.pi/180)
            self.length      = np.sum(np.sqrt(ldiff_meter[:,0]**2+
                                              ldiff_meter[:,1]**2))

    def _set_winding_angle_and_angular_velocity(self,delta_time):
        """ Compute the sum of the oriented angles in the line.

        The angle at a given point X_k is the angle between
        vect(X_k-1, X_k) and vect(X_k, X_k+1).

        Args
            delta_time (array_like or float) : Delta time between 2 consecutive
                points. If dt is a constant delta_time should be a float and an
                array_like with len = n- otherwise, n beeing the number of points.

        """
        if self.nb_points<=2:
            self.winding_angle = 0
        else:
            vectors = self.coord_list[1:]-self.coord_list[:-1]
            not_null_vect = (vectors[:,0] != 0) + (vectors[:,1]!= 0)
            # remove the null vectors
            vectors = vectors[not_null_vect,:] 
            cross_product=vectors[:-1,0]*vectors[1:,1]-vectors[:-1,1]*vectors[1:,0]
            norms = np.sqrt(np.sum(vectors**2,axis=1))
            angles = np.arcsin(cross_product/(norms[1:]*norms[:-1]))
            # self.angle_list    = angles
            self.winding_angle = np.sum(angles)
            if type(delta_time*1.) == float:
                self.angular_velocities = angles/delta_time
            else:
                delta_time=delta_time[not_null_vect]
                self.angular_velocities=2*angles/(delta_time[1:]+delta_time[:-1])

    def get_mean_radius(self):
        """  Compute the mean radius of the stream line.

        Compute the mean distance between the stream line mean pos and the list
        of coordinates. This value does not have any real meaning if the stream
        line winding is low.

        Returns:
            mean_radius (float) : The mean radius of the stream line.

        """

        radius = np.array(self.coord_list)
        radius[:,0] -= self.mean_pos[0]
        radius[:,1] -= self.mean_pos[1]
        radius = np.sqrt(np.sum(radius**2,axis=1))
        mean_radius = np.mean(radius)
        return mean_radius
    
    def get_sub_streamline(self,i,j):
        sub_streamline = StreamLine(self.coord_list[i:j], self.delta_time)
        return sub_streamline


class Eddy:
    """ Class used to represent a eddy and compute its caracteristics.

    Args:
        sl_list (list of StreamLine) : List of stream line representing the
            eddy.

    Attributes:
        sl_list (list of StreamLine) : List of stream line representing the
            eddy.
        nb_sl (int) : Number of stream lines in sl_list.
        center (array_like(2)) : Mean of stream lines center weigthed by
            the stream line length. This represent the stream line center.
        cov_matrix (np.array(2,2)) : Covariance of all the points on all the
            streamlines.
        axis_len (np.array(2) : Length of the eddy axis.
        axis_dir (np.array(2,2)) : Direction of the eddy axis.
        angular_velocity (float) : Mean angular velocity of the eddy.

    """
    def __init__(self, sl_list):
        self.sl_list       = list(sl_list)
        self.nb_sl         = len(self.sl_list)
        self._set_center()
        self._set_cov_matrix()
        self._set_axis_len_and_dir()
        self._set_angular_velocity()

    def _set_center(self):
        """ Compute the eddy center.

        The center is the mean of stream lines center weigthed by the stream
        line length.

        """
        sl_center=np.array([self.sl_list[k].mean_pos  for k in range(self.nb_sl)])
        sl_nb_pts=np.array([self.sl_list[k].nb_points for k in range(self.nb_sl)])
        sl_wcenter = [sl_center[k]*sl_nb_pts[k] for k in range(self.nb_sl)]
        self.center= np.sum(sl_wcenter,axis=0)/np.sum(sl_nb_pts)

    def _set_cov_matrix(self):
        """ Compute the covariance of all the points on all the streamlines.

        The center is the mean of stream lines center weigthed by the stream
        line length.

        """
        nb_coord=np.sum([self.sl_list[k].nb_points for k in range(self.nb_sl)])
        merged_coord_list = np.zeros((nb_coord,2))
        index = 0
        for sl_id in range(self.nb_sl):
            sl = self.sl_list[sl_id]
            merged_coord_list[index:index+sl.nb_points] = np.array(sl.coord_list)
        self.cov_matrix = np.cov(merged_coord_list.T)

    def _set_axis_len_and_dir(self):
        """  Compute the eigen values and the eigen vectors of the cov matrix.

        The eigen values are the axis lengths and the eigen vectors are the
        directions of the axis

        """
        eig_val,eig_vec = np.linalg.eig(self.cov_matrix)
        if eig_val[0]<eig_val[1]:
            self.axis_len = np.array([eig_val[1],eig_val[0]])
            self.axis_dir = np.array([eig_vec[1],eig_vec[0]])
        else:
            self.axis_len = eig_val
            self.axis_dir = eig_vec

    def _set_angular_velocity(self):
        """  Compute the angular velocity of the eddy.

        The angular velocity is the angle variation per time units.

        """
        nb_angular_velocities  = 0
        sum_angular_velocities = 0
        for sl_id in range(self.nb_sl):
            w_list = self.sl_list[sl_id].angular_velocities
            nb_angular_velocities += len(w_list)
            sum_angular_velocities    += np.sum(w_list)
        self.angular_velocity = sum_angular_velocities/nb_angular_velocities