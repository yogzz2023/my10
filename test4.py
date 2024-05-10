import numpy as np
import csv

def sph2cart(az, el, r):
    x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    z = r * np.sin(np.radians(el))
    return x, y, z

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

    def jpda_gating(self, measurements):
        # Perform joint probabilistic data association (JPDA) gating
        gating_threshold = 1.0  # Adjust gating threshold as needed

        # Loop through each measurement and associate with predicted tracks within the gating region
        associations = []
        for measurement in measurements:
            range_meas, az_meas, el_meas = measurement
            likelihoods = []
            for i in range(len(self.Sp[0])):
                state_vector = self.Sp[:, i]
                if state_vector[0] == 0 and state_vector[1] == 0 and state_vector[2] == 0:
                    # Skip the zero state vector
                    continue
                x_pred, y_pred, z_pred = state_vector[:3]
                dist = np.sqrt((x_pred - range_meas)**2 + (y_pred - az_meas)**2 + (z_pred - el_meas)**2)
                likelihood = np.exp(-0.5 * dist**2 / gating_threshold**2) / (gating_threshold * np.sqrt(2 * np.pi))
                likelihoods.append(likelihood)
            if likelihoods:
                max_likelihood_idx = np.argmax(likelihoods)
                associations.append((max_likelihood_idx, measurement))
        return associations

    def jpda_update(self, measurements, delt):
        # Update step with JPDA
        Z = np.array(measurements)
        H = np.eye(3, 6)
        Inn = Z[:, :, np.newaxis] - np.dot(H, self.Sf)
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.sum(K[:, :, np.newaxis] * Inn, axis=1)
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

# Reading measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                r = float(row[10])  # Range
                az = float(row[11])  # Azimuth
                el = float(row[12])  # Elevation
                measurements.append((r, az, el))
            except ValueError:
                print("Skipping row with non-numeric data:", row)
    return measurements

if __name__ == "__main__":
    measurements = read_measurements_from_csv("data_57.csv")
    kalman_filter = KalmanFilter()
    kalman_filter.Initialize_Filter_state_covariance(0, 0, 0, 0, 0, 0, 0, 1, 1, 1)

    # Performing JPDA gating
    associations = kalman_filter.jpda_gating(measurements)
    
    if associations:
        print("Associations after JPDA gating:")
        for association in associations:
            print(f"Measurement: {association[1]}, Associated Track: {association[0]}")

        # Calculate the time difference between measurements m2 and m1
        m1 = associations[0][1]
        m2 = associations[1][1]
        delt = m2[-1] - m1[-1]

        # Update step with JPDA associations and time difference
        kalman_filter.jpda_update([assoc[1] for assoc in associations], delt)

        # Predicting the state covariance
        kalman_filter.predict_state_covariance(delt, np.eye(6))

        # Printing the updated predicted state and covariance
        print("\nUpdated Predicted State (Sp):")
        print(kalman_filter.Sp)
        print("\nUpdated Predicted Covariance (Pp):")
        print(kalman_filter.Pp)

        # Calculate and print the difference between measurements m2 and m1
        print("\nDifference between measurements m2 and m1:")
        print(f"Range difference: {m2[0] - m1[0]}")
        print(f"Azimuth difference: {m2[1] - m1[1]}")
        print(f"Elevation difference: {m2[2] - m1[2]}")

        # Calculate and print the velocity components
        velocity = [(m2[i] - m1[i]) / delt for i in range(3)]
        print("\nVelocity components:")
        print(f"Vx: {velocity[0]}")
        print(f"Vy: {velocity[1]}")
        print(f"Vz: {velocity[2]}")

        # Predict next state based on m2 and velocity
        next_state = kalman_filter.predict_next_state(m2, velocity, delt)
        print("\nPredicted next state (x, y, z, vx, vy, vz):")
        print(next_state.flatten())
    else:
        print("No associations found after JPDA gating.")
