"""
Visualization Module
Creates real-time UI for traffic management system
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
import config


class TrafficVisualizer:
    """
    Creates comprehensive visualization for traffic management system
    Shows video feed, detections, zones, signals, and statistics
    """
    
    def __init__(self, display_size=(1280, 720)):
        """
        Initialize visualizer
        
        Args:
            display_size: Tuple of (width, height) for display window
        """
        self.display_width, self.display_height = display_size
        self.window_name = "AI Traffic Management System"
        
        # Layout dimensions
        self.video_width = int(self.display_width * 0.7)
        self.video_height = self.display_height
        self.panel_width = self.display_width - self.video_width
        
        print("[INFO] Traffic visualizer initialized")
    
    def create_frame(self, video_frame, detections, traffic_analysis, signal_status, stats):
        """
        Create complete visualization frame
        
        Args:
            video_frame: Original video frame
            detections: Vehicle detections from detector
            traffic_analysis: Traffic analysis results
            signal_status: Signal controller status
            stats: System statistics
            
        Returns:
            np.array: Visualization frame
        """
        # Resize video frame
        video_display = cv2.resize(video_frame, (self.video_width, self.video_height))
        
        # Create info panel
        info_panel = self._create_info_panel(traffic_analysis, signal_status, stats)
        
        # Combine video and panel
        display_frame = np.hstack([video_display, info_panel])
        
        return display_frame
    
    def draw_detections_and_zones(self, frame, detections, traffic_analysis):
        """
        Draw vehicle detections and traffic zones on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            traffic_analysis: Traffic analysis results
            
        Returns:
            frame: Frame with overlays
        """
        # Draw detection zones first (semi-transparent)
        overlay = frame.copy()
        
        for zone_name, polygon in traffic_analysis.items():
            if 'color' in traffic_analysis[zone_name]:
                color = traffic_analysis[zone_name]['color']
                
                # Get zone polygon from config (need to scale to frame size)
                h, w = frame.shape[:2]
                normalized_coords = config.DETECTION_ZONES[zone_name]
                pixel_coords = []
                for x, y in normalized_coords:
                    pixel_coords.append([int(x * w), int(y * h)])
                polygon_pts = np.array(pixel_coords, dtype=np.int32)
                
                # Draw filled polygon
                cv2.fillPoly(overlay, [polygon_pts], color)
                cv2.polylines(frame, [polygon_pts], True, color, 2)
                
                # Add zone label
                moments = cv2.moments(polygon_pts)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    # Zone info
                    count = traffic_analysis[zone_name]['count']
                    density = traffic_analysis[zone_name]['density']
                    
                    # Background for text
                    text = f"{zone_name}"
                    (tw, th), _ = cv2.getTextSize(text, config.FONT, 0.6, 2)
                    cv2.rectangle(frame, (cx-tw//2-5, cy-th-25), (cx+tw//2+5, cy-5), (0,0,0), -1)
                    
                    cv2.putText(frame, text, (cx-tw//2, cy-10), 
                              config.FONT, 0.6, config.COLORS['WHITE'], 2)
                    
                    text2 = f"{count} | {density}"
                    (tw2, th2), _ = cv2.getTextSize(text2, config.FONT, 0.5, 1)
                    cv2.rectangle(frame, (cx-tw2//2-5, cy+5), (cx+tw2//2+5, cy+th2+10), (0,0,0), -1)
                    cv2.putText(frame, text2, (cx-tw2//2, cy+th2+5), 
                              config.FONT, 0.5, config.COLORS['WHITE'], 1)
        
        # Blend overlay
        alpha = 0.25
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw vehicle bounding boxes
        for detection in detections:
            bbox = detection['bbox']
            vehicle_class = detection.get('class', 'vehicle')
            confidence = detection.get('confidence', 0)
            
            # Box color based on class
            if vehicle_class == 'car':
                color = config.COLORS['GREEN']
            elif vehicle_class == 'motorcycle':
                color = config.COLORS['BLUE']
            elif vehicle_class == 'bus':
                color = config.COLORS['ORANGE']
            elif vehicle_class == 'truck':
                color = config.COLORS['PURPLE']
            else:
                color = config.COLORS['WHITE']
            
            # Draw box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Label
            label = f"{vehicle_class}"
            if 'id' in detection:
                label = f"#{detection['id']} {label}"
            
            cv2.putText(frame, label, (bbox[0], bbox[1]-5), 
                       config.FONT, 0.4, color, 1)
        
        return frame
    
    def _create_info_panel(self, traffic_analysis, signal_status, stats):
        """
        Create information panel showing signals and stats
        
        Args:
            traffic_analysis: Traffic analysis results
            signal_status: Signal status
            stats: System statistics
            
        Returns:
            np.array: Info panel image
        """
        # Create black panel
        panel = np.zeros((self.display_height, self.panel_width, 3), dtype=np.uint8)
        
        y_offset = 30
        x_margin = 15
        
        # Title
        cv2.putText(panel, "TRAFFIC CONTROL", (x_margin, y_offset), 
                   config.FONT, 0.7, config.COLORS['WHITE'], 2)
        y_offset += 40
        
        # Separator line
        cv2.line(panel, (x_margin, y_offset), (self.panel_width-x_margin, y_offset), 
                config.COLORS['WHITE'], 1)
        y_offset += 30
        
        # Traffic Signals Section
        cv2.putText(panel, "SIGNALS", (x_margin, y_offset), 
                   config.FONT, 0.6, config.COLORS['YELLOW'], 2)
        y_offset += 30
        
        # Display each signal
        if signal_status and 'signals' in signal_status:
            for direction in config.DETECTION_ZONES.keys():
                state = signal_status['signals'].get(direction)
                
                # Signal indicator (circle)
                if state:
                    state_name = state.value
                    color = config.COLORS[state_name]
                else:
                    state_name = "RED"
                    color = config.COLORS['RED']
                
                # Draw signal light
                cv2.circle(panel, (x_margin + 20, y_offset), 12, color, -1)
                cv2.circle(panel, (x_margin + 20, y_offset), 12, config.COLORS['WHITE'], 1)
                
                # Direction name
                cv2.putText(panel, direction, (x_margin + 45, y_offset + 5), 
                           config.FONT, 0.5, config.COLORS['WHITE'], 1)
                
                # Show countdown for active signal
                if signal_status.get('active_direction') == direction:
                    time_remaining = signal_status.get('time_remaining', 0)
                    phase = signal_status.get('phase', 'GREEN')
                    countdown_text = f"{phase} {int(time_remaining)}s"
                    cv2.putText(panel, countdown_text, (x_margin + 45, y_offset + 20), 
                               config.FONT, 0.4, config.COLORS['YELLOW'], 1)
                
                y_offset += 50
        
        y_offset += 20
        
        # Separator
        cv2.line(panel, (x_margin, y_offset), (self.panel_width-x_margin, y_offset), 
                config.COLORS['WHITE'], 1)
        y_offset += 30
        
        # Traffic Density Section
        cv2.putText(panel, "TRAFFIC DENSITY", (x_margin, y_offset), 
                   config.FONT, 0.6, config.COLORS['YELLOW'], 2)
        y_offset += 30
        
        if traffic_analysis:
            for direction, data in traffic_analysis.items():
                count = data.get('count', 0)
                density = data.get('density', 'N/A')
                color = data.get('color', config.COLORS['WHITE'])
                
                # Density bar
                bar_width = int((count / 30) * (self.panel_width - 2*x_margin - 100))
                bar_width = min(bar_width, self.panel_width - 2*x_margin - 100)
                
                cv2.rectangle(panel, (x_margin, y_offset-12), 
                            (x_margin + bar_width, y_offset), color, -1)
                
                # Text
                text = f"{direction[:1]}: {count}"
                cv2.putText(panel, text, (x_margin + bar_width + 10, y_offset - 2), 
                           config.FONT, 0.4, config.COLORS['WHITE'], 1)
                
                y_offset += 25
        
        y_offset += 20
        
        # Separator
        cv2.line(panel, (x_margin, y_offset), (self.panel_width-x_margin, y_offset), 
                config.COLORS['WHITE'], 1)
        y_offset += 30
        
        # Statistics
        cv2.putText(panel, "STATISTICS", (x_margin, y_offset), 
                   config.FONT, 0.6, config.COLORS['YELLOW'], 2)
        y_offset += 30
        
        if stats:
            # FPS
            fps = stats.get('fps', 0)
            cv2.putText(panel, f"FPS: {fps:.1f}", (x_margin, y_offset), 
                       config.FONT, 0.5, config.COLORS['WHITE'], 1)
            y_offset += 25
            
            # Total vehicles
            total = stats.get('total_vehicles', 0)
            cv2.putText(panel, f"Total Vehicles: {total}", (x_margin, y_offset), 
                       config.FONT, 0.5, config.COLORS['WHITE'], 1)
            y_offset += 25
            
            # Active tracks
            tracks = stats.get('active_tracks', 0)
            cv2.putText(panel, f"Active Tracks: {tracks}", (x_margin, y_offset), 
                       config.FONT, 0.5, config.COLORS['WHITE'], 1)
            y_offset += 25
            
            # Cycle count
            if signal_status:
                cycles = signal_status.get('cycle_count', 0)
                cv2.putText(panel, f"Signal Cycles: {cycles}", (x_margin, y_offset), 
                           config.FONT, 0.5, config.COLORS['WHITE'], 1)
                y_offset += 25
        
        # Footer
        y_offset = self.display_height - 40
        cv2.putText(panel, "Press 'q' to quit", (x_margin, y_offset), 
                   config.FONT, 0.4, config.COLORS['WHITE'], 1)
        cv2.putText(panel, "Press 's' for next signal", (x_margin, y_offset + 20), 
                   config.FONT, 0.4, config.COLORS['WHITE'], 1)
        
        return panel
    
    def show(self, frame, window_name=None):
        """Display frame in window"""
        if window_name is None:
            window_name = self.window_name
        cv2.imshow(window_name, frame)
    
    def wait_key(self, delay=1):
        """Wait for key press"""
        return cv2.waitKey(delay)
    
    def destroy(self):
        """Close all windows"""
        cv2.destroyAllWindows()


# Test function
if __name__ == "__main__":
    print("Testing Traffic Visualizer...")
    
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(test_frame, "TEST VIDEO FEED", (400, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Create visualizer
    viz = TrafficVisualizer()
    
    # Test data
    test_traffic = {
        'NORTH': {'count': 15, 'density': 'MEDIUM', 'color': (0, 255, 255)},
        'SOUTH': {'count': 5, 'density': 'LOW', 'color': (0, 255, 0)},
        'EAST': {'count': 22, 'density': 'HIGH', 'color': (0, 165, 255)},
        'WEST': {'count': 8, 'density': 'MEDIUM', 'color': (0, 255, 255)}
    }
    
    from signal_controller import SignalState
    test_signals = {
        'signals': {
            'NORTH': SignalState.GREEN,
            'SOUTH': SignalState.RED,
            'EAST': SignalState.RED,
            'WEST': SignalState.RED
        },
        'active_direction': 'NORTH',
        'phase': 'GREEN',
        'time_remaining': 25.5,
        'cycle_count': 3
    }
    
    test_stats = {
        'fps': 24.5,
        'total_vehicles': 50,
        'active_tracks': 15
    }
    
    # Add detections on frame
    test_detections = [
        {'bbox': [100, 100, 200, 200], 'class': 'car', 'id': 1, 'confidence': 0.95},
        {'bbox': [300, 150, 400, 250], 'class': 'bus', 'id': 2, 'confidence': 0.88},
    ]
    
    # Draw on test frame
    test_frame = viz.draw_detections_and_zones(test_frame, test_detections, test_traffic)
    
    # Create full display
    display = viz.create_frame(test_frame, test_detections, test_traffic, test_signals, test_stats)
    
    # Show
    viz.show(display)
    print("Press any key to close...")
    viz.wait_key(0)
    viz.destroy()
