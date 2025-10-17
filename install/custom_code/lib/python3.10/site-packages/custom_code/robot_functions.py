import numpy as np
import random
import copy

class particle():

    def __init__(self):
        self.x = (random.random() - 0.5) * 2
        self.y = (random.random() - 0.5) * 2
        self.orientation = random.uniform(-np.pi, np.pi)
        self.weight = 1.0

    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, alpha):
        delta_rot1 = odom['r1']
        delta_trans = odom['t']
        delta_rot2 = odom['r2']

        var_rot1 = alpha[0] * abs(delta_rot1) + alpha[1] * delta_trans
        var_trans = alpha[2] * delta_trans + alpha[3] * (abs(delta_rot1) + abs(delta_rot2))
        var_rot2 = alpha[0] * abs(delta_rot2) + alpha[1] * delta_trans

        var_rot1 = max(0, var_rot1)
        var_trans = max(0, var_trans)
        var_rot2 = max(0, var_rot2)

        delta_rot1_hat = delta_rot1 - np.random.normal(0.0, np.sqrt(var_rot1))
        delta_trans_hat = delta_trans - np.random.normal(0.0, np.sqrt(var_trans))
        delta_rot2_hat = delta_rot2 - np.random.normal(0.0, np.sqrt(var_rot2))

        self.x += delta_trans_hat * np.cos(self.orientation + delta_rot1_hat)
        self.y += delta_trans_hat * np.sin(self.orientation + delta_rot1_hat)
        self.orientation += delta_rot1_hat + delta_rot2_hat
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))

class RobotFunctions:

    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = [particle() for _ in range(self.num_particles)]
        self.motion_noise = [10.0, 10.0, 0.1, 0.1] 

    def get_particle_states(self):
        return np.array([[p.x, p.y, p.orientation] for p in self.particles])

    def get_weights(self):
        """Devuelve una lista con los pesos actuales de todas las partículas."""
        return [p.weight for p in self.particles]

    def move_particles(self, deltas):
        for part in self.particles:
            part.move_odom(deltas, self.motion_noise)
    
    def get_selected_state(self):
        weights = self.get_weights()
        if np.sum(weights) == 0:
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights /= np.sum(weights)

        mean_x = np.average([p.x for p in self.particles], weights=weights)
        mean_y = np.average([p.y for p in self.particles], weights=weights)

        sin_sum = np.sum(weights * np.sin([p.orientation for p in self.particles]))
        cos_sum = np.sum(weights * np.cos([p.orientation for p in self.particles]))
        mean_orientation = np.arctan2(sin_sum, cos_sum)
        return [mean_x, mean_y, mean_orientation]

    def update_particles(self, data, map_data, grid):
        ranges = np.array(data.ranges)
        angles = data.angle_min + np.arange(len(ranges)) * data.angle_increment
        
        resolution = map_data.info.resolution
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y
        map_height, map_width = grid.shape

        log_weights = np.zeros(self.num_particles)

        for i, p in enumerate(self.particles):
            global_angles = p.orientation + angles
            global_x = p.x + ranges * np.cos(global_angles)
            global_y = p.y + ranges * np.sin(global_angles)
            
            map_x = ((global_x - origin_x) / resolution).astype(int)
            map_y = ((global_y - origin_y) / resolution).astype(int)
            
            valid_indices = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)

            if not np.any(valid_indices):
                log_weights[i] = -np.inf
                continue
            
            likelihoods = grid[map_y[valid_indices], map_x[valid_indices]] / 100.0
            log_likelihoods = np.log(np.clip(likelihoods, 1e-6, 1.0))
            log_weights[i] = np.sum(log_likelihoods)

        max_log_weight = np.max(log_weights)
        
 
        weights_exp = np.exp(log_weights - max_log_weight)

        sum_weights = np.sum(weights_exp)
        if sum_weights == 0:
            normalized_weights = np.ones(self.num_particles) / self.num_particles
        else:
            normalized_weights = weights_exp / sum_weights

        for i, p in enumerate(self.particles):
            p.weight = normalized_weights[i]

        neff = 1.0 / np.sum(normalized_weights**2) if np.sum(normalized_weights**2) > 0 else 0
        if neff < self.num_particles / 2.0:
            self.resample(normalized_weights)

    def resample(self, weights):
        new_particles = []
        N = self.num_particles
        positions = (np.random.random() + np.arange(N)) / N
        
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                new_part = copy.deepcopy(self.particles[j])
                new_part.x += np.random.normal(0, 0.02)
                new_part.y += np.random.normal(0, 0.02)
                new_part.orientation += np.random.normal(0, 0.05)
                new_particles.append(new_part)
                i += 1
            else:
                j += 1
        
        self.particles = new_particles
        for p in self.particles:
            p.weight = 1.0 / self.num_particles