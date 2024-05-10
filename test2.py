import numpy as np
import csv

def sph2cart(az, el, r):
    x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    z = r * np.sin(np.radians(el))
    return x, y, z

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            # Convert spherical to Cartesian coordinates
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

class KalmanFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Size of Sf matrix is 6x1
        self.Filtered_Time = 0  # Initialize with zero or any appropriate value
        self.R = np.zeros((6, 6))  # Size of R matrix is 6x6
        self.pf = np.zeros((6, 6))  # Size of pf matrix is 6x6
        self.Phi = np.eye(6)  # Size of Phi matrix is 6x6
        self.Sp = np.zeros((6, 1))  # Size of Sp matrix is 6x1
        self.Q = np.zeros((6, 6))  # Size of Q matrix is 6x6
        self.Pp = np.zeros((6, 6))  # Size of Pp matrix is 6x6
        self.predicted_Time = 0  # Initialize with zero or any appropriate value

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time, sig_r, sig_a, sig_e_sqr):
        # Initialize state and covariance matrices
        self.Sf[0] = x
        self.Sf[1] = y
        self.Sf[2] = z
        self.Sf[3] = vx
        self.Sf[4] = vy
        self.Sf[5] = vz
        self.Filtered_Time = time

        # Initialize covariance matrix R
        self.R[0, 0] = sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.sin(sig_a)**2 * sig_e_sqr
        self.R[1, 1] = sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.cos(sig_a)**2 * sig_e_sqr
        self.R[2, 2] = sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.sin(sig_a)**2 * sig_e_sqr

        # Initialize pf matrix
        for i in range(6):
            for j in range(6):
                k = i % 3
                l = j % 3
                self.pf[i, j] = self.R[k, l]

    def predict_state_covariance(self, delt, plant_noise):
        # Predict state covariance
        self.Phi[0, 3] = delt
        self.Phi[1, 4] = delt
        self.Phi[2, 5] = delt
        self.Sp = np.dot(self.Phi, self.Sf)
        self.predicted_Time = self.Filtered_Time + delt

        T_3 = (delt * delt * delt) / 3.0
        T_2 = (delt * delt) / 2.0
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = delt
        self.Q[4, 4] = delt
        self.Q[5, 5] = delt
        self.Q = np.dot(self.Q, plant_noise)
        self.Pp = np.dot(np.dot(self.Phi, self.pf), self.Phi.T) + self.Q
        return self.Sp, self.Pp

    def predict_next_state(self, m2, velocity, delt):
        # Predict next state X3^ based on m2, velocity, and time difference delt
        x_pred = m2[0] + velocity[0] * delt
        y_pred = m2[1] + velocity[1] * delt
        z_pred = m2[2] + velocity[2] * delt
        vx_pred = velocity[0]
        vy_pred = velocity[1]
        vz_pred = velocity[2]
        return np.array([[x_pred], [y_pred], [z_pred], [vx_pred], [vy_pred], [vz_pred]])

# Example usage
if __name__ == "__main__":
    measurements = read_measurements_from_csv("data_57.csv")
    kalman_filter = KalmanFilter()
    kalman_filter.Initialize_Filter_state_covariance(0, 0, 0, 0, 0, 0, 0, 1, 1, 1)
    # Initializing three measurements m1, m2, and m3
    m1 = measurements[0]
    m2 = measurements[1]
    m3 = measurements[2]
    
    # Predicting the next state X3^ using m2, velocity, and time difference delt
    velocity = [(m2[i] - m1[i]) / (m2[3] - m1[3]) for i in range(3)]  # Calculating velocity from m1 to m2
    delt = m3[3] - m2[3]  # Time difference between m2 and m3
    X3_predicted = kalman_filter.predict_next_state(m2[:3], velocity, delt)

    # Printing the initial predicted state and covariance
    print("Initial Predicted State (Sp):")
    print(kalman_filter.Sp)
    print("\nInitial Predicted Covariance (Pp):")
    print(kalman_filter.Pp)

    # Updating the state and covariance with measurements
    kalman_filter.predict_state_covariance(delt, np.eye(6))

    # Printing the updated predicted state and covariance
    print("\nUpdated Predicted State (Sp):")
    print(kalman_filter.Sp)
    print("\nUpdated Predicted Covariance (Pp):")
    print(kalman_filter.Pp)
