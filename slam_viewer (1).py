# slam_viewer.py
# Main application with zoom, pan, and dynamic map display

import sys
import math
import time
import pygame
import numpy as np

from slam_toolbox import SlamToolbox
from rplidar import RPLidar as LidarModel

# --- Constants ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

MAP_SIZE_METERS = 20
MAP_SIZE_PIXELS = 800

LIDAR_DEVICE = 'COM12'  # <-- CHANGE THIS

class SlamViewer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption('SLAM Toolbox - Graph SLAM with Zoom/Pan')
        
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 22)
        
        # SLAM system
        self.slam = SlamToolbox(MAP_SIZE_PIXELS, MAP_SIZE_METERS)
        
        # Lidar
        try:
            self.lidar = LidarModel(LIDAR_DEVICE)
            self.iterator = self.lidar.iter_scans(max_buf_meas=3000)
            print("✓ RPLidar connected")
        except Exception as e:
            print(f"✗ Lidar error: {e}")
            pygame.quit()
            sys.exit()
        
        # Camera/view control
        self.camera_x = 0.0  # meters
        self.camera_y = 0.0
        self.zoom = 40.0  # pixels per meter
        self.zoom_min = 10.0
        self.zoom_max = 200.0
        
        # Mouse control
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # State
        self.pose = np.array([0.0, 0.0, 0.0])
        self.match_quality = 0.0
        self.status_message = ""
        self.status_timer = 0
        
        # Performance
        self.clock = pygame.time.Clock()
        self.fps = 30
        
        # Follow robot mode
        self.follow_robot = True
        
        print("\n🎮 CONTROLS:")
        print("   Mouse Wheel: Zoom in/out")
        print("   Right Click + Drag: Pan map")
        print("   F: Toggle follow robot")
        print("   S: Save map")
        print("   L: Load map")
        print("   R: Reset camera to robot")
        print("   Q/ESC: Quit")
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen pixels."""
        screen_x = (world_x - self.camera_x) * self.zoom + WINDOW_WIDTH / 2
        screen_y = (world_y - self.camera_y) * self.zoom + WINDOW_HEIGHT / 2
        return int(screen_x), int(screen_y)
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen pixels to world coordinates."""
        world_x = (screen_x - WINDOW_WIDTH / 2) / self.zoom + self.camera_x
        world_y = (screen_y - WINDOW_HEIGHT / 2) / self.zoom + self.camera_y
        return world_x, world_y
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.VIDEORESIZE:
                global WINDOW_WIDTH, WINDOW_HEIGHT
                WINDOW_WIDTH, WINDOW_HEIGHT = event.size
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                old_zoom = self.zoom
                if event.y > 0:  # Scroll up
                    self.zoom = min(self.zoom * 1.2, self.zoom_max)
                else:  # Scroll down
                    self.zoom = max(self.zoom / 1.2, self.zoom_min)
                
                # Zoom towards mouse cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
                # Adjust camera to keep point under mouse stationary
                self.camera_x += world_x * (old_zoom / self.zoom - 1)
                self.camera_y += world_y * (old_zoom / self.zoom - 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right click
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                    self.follow_robot = False
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    self.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    current_pos = pygame.mouse.get_pos()
                    dx = (current_pos[0] - self.last_mouse_pos[0]) / self.zoom
                    dy = (current_pos[1] - self.last_mouse_pos[1]) / self.zoom
                    self.camera_x -= dx
                    self.camera_y -= dy
                    self.last_mouse_pos = current_pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return False
                
                elif event.key == pygame.K_f:
                    self.follow_robot = not self.follow_robot
                    self.status_message = f"Follow robot: {'ON' if self.follow_robot else 'OFF'}"
                    self.status_timer = 90
                
                elif event.key == pygame.K_r:
                    self.camera_x = self.pose[0]
                    self.camera_y = self.pose[1]
                    self.follow_robot = True
                    self.status_message = "Camera reset to robot"
                    self.status_timer = 90
                
                elif event.key == pygame.K_s:
                    try:
                        self.slam.save_map(pose=self.pose)
                        self.status_message = "✓ Map saved!"
                        self.status_timer = 180
                    except Exception as e:
                        self.status_message = f"✗ Save failed: {e}"
                        self.status_timer = 180
                
                elif event.key == pygame.K_l:
                    import glob, os
                    pkl_files = glob.glob("map_*.pkl")
                    if pkl_files:
                        latest = max(pkl_files, key=os.path.getctime)
                        filename_base = latest.replace('.pkl', '')
                        loaded_pose = self.slam.load_map(filename_base)
                        if loaded_pose is not None:
                            self.pose = loaded_pose
                            self.camera_x = self.pose[0]
                            self.camera_y = self.pose[1]
                            self.status_message = f"✓ Loaded: {filename_base}"
                        else:
                            self.status_message = "✗ Load failed"
                        self.status_timer = 180
                    else:
                        self.status_message = "✗ No saved maps"
                        self.status_timer = 180
        
        return True
    
    def update_slam(self):
        """Get scan and update SLAM."""
        try:
            scan_data = next(self.iterator)
        except Exception as e:
            print(f"Lidar read error: {e}")
            return
        
        scan = [(item[2] / 1000.0, math.radians(item[1])) 
                for item in scan_data if item[2] > 0]
        
        if len(scan) < 20:
            return
        
        # Update SLAM
        timestamp = time.time()
        self.pose, self.match_quality = self.slam.update(scan, self.pose, timestamp)
        
        # Follow robot if enabled
        if self.follow_robot:
            # Smooth camera follow
            self.camera_x = self.camera_x * 0.8 + self.pose[0] * 0.2
            self.camera_y = self.camera_y * 0.8 + self.pose[1] * 0.2
    
    def render(self):
        """Render everything."""
        self.screen.fill((30, 30, 35))
        
        # Get map
        map_bytes = self.slam.get_map_bytes()
        map_size_px, map_size_m, map_origin_x, map_origin_y = self.slam.get_map_dimensions()
        
        if len(map_bytes) == map_size_px * map_size_px:
            # Create surface from map
            map_surf = pygame.image.frombuffer(map_bytes, (map_size_px, map_size_px), 'P')
            map_surf.set_palette([(i, i, i) for i in range(256)])
            
            # Calculate where to draw the map
            # Map spans from map_origin to map_origin + map_size_m
            top_left_screen = self.world_to_screen(map_origin_x, map_origin_y)
            map_display_size = int(map_size_m * self.zoom)
            
            # Scale map to fit zoom
            if map_display_size > 0:
                scaled_map = pygame.transform.scale(map_surf, (map_display_size, map_display_size))
                self.screen.blit(scaled_map, top_left_screen)
        
        # Draw grid
        self.draw_grid()
        
        # Draw robot
        self.draw_robot()
        
        # Draw pose graph
        self.draw_pose_graph()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def draw_grid(self):
        """Draw coordinate grid."""
        # Determine grid spacing based on zoom
        if self.zoom > 100:
            grid_spacing = 0.5
        elif self.zoom > 50:
            grid_spacing = 1.0
        elif self.zoom > 20:
            grid_spacing = 2.0
        else:
            grid_spacing = 5.0
        
        # Visible world bounds
        top_left_world = self.screen_to_world(0, 0)
        bottom_right_world = self.screen_to_world(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Vertical lines
        start_x = int(top_left_world[0] / grid_spacing) * grid_spacing
        x = start_x
        while x < bottom_right_world[0]:
            screen_x, _ = self.world_to_screen(x, 0)
            color = (80, 80, 80) if abs(x) < 0.01 else (50, 50, 55)
            pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, WINDOW_HEIGHT), 1)
            x += grid_spacing
        
        # Horizontal lines
        start_y = int(top_left_world[1] / grid_spacing) * grid_spacing
        y = start_y
        while y < bottom_right_world[1]:
            _, screen_y = self.world_to_screen(0, y)
            color = (80, 80, 80) if abs(y) < 0.01 else (50, 50, 55)
            pygame.draw.line(self.screen, color, (0, screen_y), (WINDOW_WIDTH, screen_y), 1)
            y += grid_spacing
    
    def draw_robot(self):
        """Draw robot pose."""
        robot_screen = self.world_to_screen(self.pose[0], self.pose[1])
        
        # Robot body
        robot_radius = max(5, int(0.15 * self.zoom))
        pygame.draw.circle(self.screen, (255, 50, 50), robot_screen, robot_radius)
        pygame.draw.circle(self.screen, (255, 150, 150), robot_screen, robot_radius, 2)
        
        # Orientation
        line_length = max(15, int(0.3 * self.zoom))
        end_x = robot_screen[0] + line_length * math.cos(self.pose[2])
        end_y = robot_screen[1] + line_length * math.sin(self.pose[2])
        pygame.draw.line(self.screen, (100, 255, 100), robot_screen, (int(end_x), int(end_y)), 3)
    
    def draw_pose_graph(self):
        """Draw pose graph nodes and edges."""
        if not hasattr(self.slam, 'nodes') or len(self.slam.nodes) < 2:
            return
        
        # Draw edges
        for node in self.slam.nodes:
            node_screen = self.world_to_screen(node.pose[0], node.pose[1])
            
            for other_id, _, _ in node.constraints:
                if other_id < len(self.slam.nodes):
                    other = self.slam.nodes[other_id]
                    other_screen = self.world_to_screen(other.pose[0], other.pose[1])
                    
                    # Different colors for odometry vs loop closures
                    if abs(node.id - other_id) == 1:
                        color = (100, 100, 255)  # Blue for odometry
                    else:
                        color = (255, 100, 255)  # Magenta for loop closures
                    
                    pygame.draw.line(self.screen, color, node_screen, other_screen, 1)
        
        # Draw nodes
        for node in self.slam.nodes:
            node_screen = self.world_to_screen(node.pose[0], node.pose[1])
            pygame.draw.circle(self.screen, (150, 150, 255), node_screen, 3)
    
    def draw_ui(self):
        """Draw UI overlay."""
        # Background panel
        pygame.draw.rect(self.screen, (20, 20, 25, 200), (0, 0, 450, 150))
        
        # Pose info
        pose_text = f"Pose: ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {np.rad2deg(self.pose[2]):.1f}°)"
        self.blit_text(pose_text, (10, 10), self.small_font, (255, 255, 255))
        
        # Match quality
        color = (0, 255, 0) if self.match_quality > 0.5 else (255, 165, 0)
        quality_text = f"Match Quality: {self.match_quality:.3f}"
        self.blit_text(quality_text, (10, 35), self.small_font, color)
        
        # Camera info
        cam_text = f"Camera: ({self.camera_x:.2f}, {self.camera_y:.2f}) Zoom: {self.zoom:.1f}x"
        self.blit_text(cam_text, (10, 60), self.small_font, (200, 200, 200))
        
        # Node count
        if hasattr(self.slam, 'nodes'):
            nodes_text = f"Graph: {len(self.slam.nodes)} nodes"
            self.blit_text(nodes_text, (10, 85), self.small_font, (200, 200, 200))
        
        # Follow mode
        follow_text = f"Follow: {'ON' if self.follow_robot else 'OFF'}"
        follow_color = (100, 255, 100) if self.follow_robot else (150, 150, 150)
        self.blit_text(follow_text, (10, 110), self.small_font, follow_color)
        
        # Controls hint
        controls = "Wheel=Zoom | RClick+Drag=Pan | F=Follow | S=Save | L=Load | R=Reset"
        self.blit_text(controls, (10, WINDOW_HEIGHT - 30), self.small_font, (150, 150, 150))
        
        # Status message
        if self.status_timer > 0:
            status_surf = self.small_font.render(self.status_message, True, (100, 255, 100))
            status_rect = status_surf.get_rect(center=(WINDOW_WIDTH // 2, 30))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), status_rect.inflate(20, 10))
            self.screen.blit(status_surf, status_rect)
            self.status_timer -= 1
        
        # FPS
        fps_text = f"{int(self.clock.get_fps())} FPS"
        self.blit_text(fps_text, (WINDOW_WIDTH - 80, 10), self.small_font, (150, 150, 150))
    
    def blit_text(self, text, pos, font, color):
        """Helper to render text."""
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)
    
    def run(self):
        """Main loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update_slam()
            self.render()
        
        # Cleanup
        print("\n🛑 Shutting down...")
        self.lidar.stop()
        self.lidar.disconnect()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    viewer = SlamViewer()
    viewer.run()