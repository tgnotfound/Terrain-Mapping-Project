# slam_toolbox.py
# Complete SLAM implementation with graph-based SLAM

import numpy as np
import pickle
from datetime import datetime
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt

class PoseNode:
    """Node in the pose graph."""
    def __init__(self, node_id, pose, scan):
        self.id = node_id
        self.pose = pose.copy()
        self.scan = scan.copy() if isinstance(scan, np.ndarray) else scan[:]
        self.constraints = []  # List of (node_id, relative_pose, information_matrix)
    
    def add_constraint(self, other_id, relative_pose, information):
        self.constraints.append((other_id, relative_pose, information))

class SlamToolbox:
    """Graph-based SLAM system."""
    
    def __init__(self, map_size_pixels=800, map_size_meters=20):
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
        self.resolution = map_size_meters / map_size_pixels
        
        # Map storage (occupancy grid)
        self.map = np.ones((map_size_pixels, map_size_pixels), dtype=np.float32) * 0.5
        self.map_origin_x = -map_size_meters / 2
        self.map_origin_y = -map_size_meters / 2
        
        # Pose graph
        self.nodes = []
        self.node_spacing = 0.3  # meters - minimum distance between nodes
        
        # State tracking
        self.last_pose = None
        self.current_pose = None
        self.last_timestamp = None
        self.velocity_estimate = np.array([0.0, 0.0, 0.0])
        self.last_node_pose = None
        
        # Scan matching parameters
        self.max_iterations = 50
        self.convergence_threshold = 0.001
        self.search_window = 0.5  # meters
        self.search_angle = np.deg2rad(15)  # radians
        
        # Loop closure
        self.loop_closure_distance = 1.5  # meters
        self.loop_closure_score_threshold = 0.6
        self.optimization_frequency = 5  # Optimize every N nodes
        
        print(f"✓ SLAM initialized: {map_size_pixels}px, {map_size_meters}m")
    
    def update(self, scan, pose, timestamp):
        """Main update function."""
        # Initialize dt
        dt = 0.033  # Default ~30Hz
        
        # Calculate actual dt if possible
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt <= 0:
                dt = 0.033
        
        self.last_timestamp = timestamp
        
        # Store current pose
        self.current_pose = pose.copy()
        
        # Calculate velocity estimate
        if self.last_pose is not None and dt > 0:
            self.velocity_estimate = (self.current_pose - self.last_pose) / dt
        else:
            self.velocity_estimate = np.array([0.0, 0.0, 0.0])
        
        # Perform scan matching
        corrected_pose, match_quality = self.scan_match(scan, pose)
        
        # Update map with scan
        self.update_map(scan, corrected_pose)
        
        # Add to pose graph if moved enough
        self.add_pose_node(corrected_pose, scan)
        
        # Check for loop closures
        if len(self.nodes) > 10:
            self.detect_loop_closures()
        
        # Optimize pose graph periodically
        if len(self.nodes) % self.optimization_frequency == 0 and len(self.nodes) > 5:
            self.optimize_pose_graph()
        
        # Update last pose
        self.last_pose = corrected_pose.copy()
        
        return corrected_pose, match_quality
    
    def scan_match(self, scan, initial_pose):
        """ICP-like scan matching against map."""
        if len(self.nodes) == 0:
            # First scan - no matching needed
            return initial_pose, 0.0
        
        # Convert scan to cartesian points
        points = self.scan_to_points(scan, initial_pose)
        
        if len(points) < 10:
            return initial_pose, 0.0
        
        # Create distance field from map for faster matching
        distance_field = self.create_distance_field()
        
        best_pose = initial_pose.copy()
        best_score = self.evaluate_pose(points, initial_pose, distance_field)
        
        # Simple grid search around initial pose
        search_steps_xy = 5
        search_steps_theta = 5
        
        for dx in np.linspace(-self.search_window, self.search_window, search_steps_xy):
            for dy in np.linspace(-self.search_window, self.search_window, search_steps_xy):
                for dtheta in np.linspace(-self.search_angle, self.search_angle, search_steps_theta):
                    test_pose = initial_pose + np.array([dx, dy, dtheta])
                    test_points = self.scan_to_points(scan, test_pose)
                    score = self.evaluate_pose(test_points, test_pose, distance_field)
                    
                    if score > best_score:
                        best_score = score
                        best_pose = test_pose.copy()
        
        # Normalize score to 0-1 range
        match_quality = min(best_score / len(points), 1.0)
        
        return best_pose, match_quality
    
    def scan_to_points(self, scan, pose):
        """Convert scan to world coordinates."""
        points = []
        for distance, angle in scan:
            if distance < 0.1 or distance > 10.0:
                continue
            
            # Point in robot frame
            x_robot = distance * np.cos(angle)
            y_robot = distance * np.sin(angle)
            
            # Transform to world frame
            cos_theta = np.cos(pose[2])
            sin_theta = np.sin(pose[2])
            
            x_world = pose[0] + x_robot * cos_theta - y_robot * sin_theta
            y_world = pose[1] + x_robot * sin_theta + y_robot * cos_theta
            
            points.append([x_world, y_world])
        
        return np.array(points)
    
    def create_distance_field(self):
        """Create distance field from occupancy map."""
        occupied = (self.map < 0.3).astype(np.uint8)
        distance_field = distance_transform_edt(1 - occupied)
        return distance_field
    
    def evaluate_pose(self, points, pose, distance_field):
        """Evaluate how well points match the map."""
        score = 0.0
        
        for point in points:
            # Convert to map coordinates
            mx = int((point[0] - self.map_origin_x) / self.resolution)
            my = int((point[1] - self.map_origin_y) / self.resolution)
            
            if 0 <= mx < self.map_size_pixels and 0 <= my < self.map_size_pixels:
                # Check if point is near occupied cell
                if self.map[my, mx] < 0.4:
                    score += 1.0
                elif distance_field[my, mx] < 3:
                    score += 0.5
        
        return score
    
    def update_map(self, scan, pose):
        """Update occupancy grid with new scan."""
        robot_x = pose[0]
        robot_y = pose[1]
        robot_theta = pose[2]
        
        # Robot position in map coordinates
        robot_mx = int((robot_x - self.map_origin_x) / self.resolution)
        robot_my = int((robot_y - self.map_origin_y) / self.resolution)
        
        if not (0 <= robot_mx < self.map_size_pixels and 0 <= robot_my < self.map_size_pixels):
            return
        
        for distance, angle in scan:
            if distance < 0.1 or distance > 10.0:
                continue
            
            # End point in world coordinates
            world_angle = robot_theta + angle
            end_x = robot_x + distance * np.cos(world_angle)
            end_y = robot_y + distance * np.sin(world_angle)
            
            # End point in map coordinates
            end_mx = int((end_x - self.map_origin_x) / self.resolution)
            end_my = int((end_y - self.map_origin_y) / self.resolution)
            
            # Ray tracing - mark cells along ray as free
            points = self.bresenham_line(robot_mx, robot_my, end_mx, end_my)
            
            for i, (mx, my) in enumerate(points):
                if 0 <= mx < self.map_size_pixels and 0 <= my < self.map_size_pixels:
                    if i < len(points) - 1:
                        # Free space
                        self.map[my, mx] = self.map[my, mx] * 0.7 + 0.3 * 0.9
                    else:
                        # Obstacle
                        self.map[my, mx] = self.map[my, mx] * 0.7 + 0.3 * 0.1
    
    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        for _ in range(max(dx, dy) + 1):
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def add_pose_node(self, pose, scan):
        """Add node to pose graph if moved enough."""
        # Check if we should add a new node
        if self.last_node_pose is None:
            should_add = True
        else:
            distance = np.linalg.norm(pose[:2] - self.last_node_pose[:2])
            angle_diff = abs(self.normalize_angle(pose[2] - self.last_node_pose[2]))
            should_add = distance > self.node_spacing or angle_diff > np.deg2rad(15)
        
        if should_add:
            node = PoseNode(len(self.nodes), pose, scan)
            
            # Add odometry constraint to previous node
            if len(self.nodes) > 0:
                prev_node = self.nodes[-1]
                relative_pose = self.compute_relative_pose(prev_node.pose, pose)
                information = np.eye(3) * 10.0  # High confidence in odometry
                node.add_constraint(prev_node.id, relative_pose, information)
            
            self.nodes.append(node)
            self.last_node_pose = pose.copy()
    
    def detect_loop_closures(self):
        """Detect and add loop closure constraints."""
        if len(self.nodes) < 10:
            return
        
        current_node = self.nodes[-1]
        
        # Check against nodes that are far in time but close in space
        for i, other_node in enumerate(self.nodes[:-10]):
            distance = np.linalg.norm(current_node.pose[:2] - other_node.pose[:2])
            
            if distance < self.loop_closure_distance:
                # Try to match scans
                score = self.match_scans(current_node.scan, current_node.pose,
                                        other_node.scan, other_node.pose)
                
                if score > self.loop_closure_score_threshold:
                    # Add loop closure constraint
                    relative_pose = self.compute_relative_pose(other_node.pose, current_node.pose)
                    information = np.eye(3) * 5.0  # Lower confidence than odometry
                    current_node.add_constraint(other_node.id, relative_pose, information)
                    print(f"  Loop closure: {current_node.id} -> {other_node.id} (score: {score:.2f})")
    
    def match_scans(self, scan1, pose1, scan2, pose2):
        """Match two scans and return similarity score."""
        points1 = self.scan_to_points(scan1, pose1)
        points2 = self.scan_to_points(scan2, pose2)
        
        if len(points1) < 10 or len(points2) < 10:
            return 0.0
        
        # Simple nearest neighbor matching
        matches = 0
        threshold = 0.1  # 10cm
        
        for p1 in points1:
            distances = np.linalg.norm(points2 - p1, axis=1)
            if np.min(distances) < threshold:
                matches += 1
        
        score = matches / len(points1)
        return score
    
    def optimize_pose_graph(self):
        """Optimize pose graph using least squares."""
        if len(self.nodes) < 3:
            return
        
        # Prepare optimization
        n_nodes = len(self.nodes)
        x0 = np.zeros(n_nodes * 3)
        
        for i, node in enumerate(self.nodes):
            x0[i*3:(i+1)*3] = node.pose
        
        # Fix first pose
        def residuals(x):
            res = []
            
            # First pose constraint (fixed)
            res.extend(x[0:3] - self.nodes[0].pose)
            
            # Constraint residuals
            for node in self.nodes:
                pose_i = x[node.id*3:(node.id+1)*3]
                
                for other_id, relative_pose, information in node.constraints:
                    if other_id < len(self.nodes):
                        pose_j = x[other_id*3:(other_id+1)*3]
                        
                        # Compute expected pose_i based on pose_j and relative_pose
                        expected = self.apply_relative_pose(pose_j, relative_pose)
                        
                        # Residual weighted by information
                        diff = pose_i - expected
                        diff[2] = self.normalize_angle(diff[2])
                        
                        res.extend(diff * np.sqrt(np.diag(information)))
            
            return np.array(res)
        
        # Optimize
        result = least_squares(residuals, x0, method='lm', max_nfev=100)
        
        # Update node poses
        for i, node in enumerate(self.nodes):
            node.pose = result.x[i*3:(i+1)*3]
        
        # Rebuild map with optimized poses
        self.rebuild_map()
    
    def rebuild_map(self):
        """Rebuild map from optimized pose graph."""
        self.map = np.ones((self.map_size_pixels, self.map_size_pixels), dtype=np.float32) * 0.5
        
        for node in self.nodes:
            self.update_map(node.scan, node.pose)
    
    def compute_relative_pose(self, pose_from, pose_to):
        """Compute relative pose between two poses."""
        dx = pose_to[0] - pose_from[0]
        dy = pose_to[1] - pose_from[1]
        dtheta = self.normalize_angle(pose_to[2] - pose_from[2])
        
        # Rotate to local frame
        cos_theta = np.cos(-pose_from[2])
        sin_theta = np.sin(-pose_from[2])
        
        local_dx = dx * cos_theta - dy * sin_theta
        local_dy = dx * sin_theta + dy * cos_theta
        
        return np.array([local_dx, local_dy, dtheta])
    
    def apply_relative_pose(self, pose_from, relative_pose):
        """Apply relative pose to get absolute pose."""
        # Rotate relative translation to world frame
        cos_theta = np.cos(pose_from[2])
        sin_theta = np.sin(pose_from[2])
        
        dx = relative_pose[0] * cos_theta - relative_pose[1] * sin_theta
        dy = relative_pose[0] * sin_theta + relative_pose[1] * cos_theta
        
        return np.array([
            pose_from[0] + dx,
            pose_from[1] + dy,
            self.normalize_angle(pose_from[2] + relative_pose[2])
        ])
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_map_bytes(self):
        """Get map as bytes for visualization."""
        map_uint8 = (self.map * 255).astype(np.uint8)
        return map_uint8.tobytes()
    
    def get_map_dimensions(self):
        """Get map dimensions."""
        return (self.map_size_pixels, self.map_size_meters, 
                self.map_origin_x, self.map_origin_y)
    
    def save_map(self, filename=None, pose=None):
        """Save map and pose graph to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"map_{timestamp}"
        
        data = {
            'map': self.map,
            'map_size_pixels': self.map_size_pixels,
            'map_size_meters': self.map_size_meters,
            'map_origin_x': self.map_origin_x,
            'map_origin_y': self.map_origin_y,
            'resolution': self.resolution,
            'nodes': [(n.id, n.pose, n.scan, n.constraints) for n in self.nodes],
            'pose': pose
        }
        
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Save as image too
        try:
            from PIL import Image
            img = Image.fromarray(self.map * 255).convert('L')
            img.save(f"{filename}.png")
        except:
            pass
        
        print(f"✓ Saved: {filename}")
    
    def load_map(self, filename):
        """Load map and pose graph from file."""
        try:
            with open(f"{filename}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.map = data['map']
            self.map_size_pixels = data['map_size_pixels']
            self.map_size_meters = data['map_size_meters']
            self.map_origin_x = data['map_origin_x']
            self.map_origin_y = data['map_origin_y']
            self.resolution = data['resolution']
            
            # Reconstruct nodes
            self.nodes = []
            for node_id, pose, scan, constraints in data['nodes']:
                node = PoseNode(node_id, pose, scan)
                node.constraints = constraints
                self.nodes.append(node)
            
            print(f"✓ Loaded: {filename} ({len(self.nodes)} nodes)")
            return data.get('pose', np.array([0.0, 0.0, 0.0]))
        
        except Exception as e:
            print(f"✗ Load error: {e}")
            return None