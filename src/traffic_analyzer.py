"""
Traffic Density Analysis Module
Analyzes vehicle density per detection zone and classifies congestion levels
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
import config


class TrafficAnalyzer:
    """
    Analyzes traffic density in different zones
    Classifies congestion levels for signal control
    """
    
    def __init__(self, frame_shape=None):
        """
        Initialize traffic analyzer
        
        Args:
            frame_shape: Tuple of (height, width) for the video frame
        """
        self.frame_shape = frame_shape
        self.zones = {}
        self.zone_polygons = {}
        
        # Initialize zones if frame shape is provided
        if frame_shape:
            self._initialize_zones(frame_shape)
        
        # Statistics
        self.history = {zone: [] for zone in config.DETECTION_ZONES.keys()}
        self.max_history_length = 30  # Keep last 30 readings
        
        print("[INFO] Traffic analyzer initialized")
    
    def _initialize_zones(self, frame_shape):
        """
        Initialize detection zones based on frame dimensions
        
        Args:
            frame_shape: Tuple of (height, width)
        """
        height, width = frame_shape[:2]
        self.frame_shape = (height, width)
        
        # Convert normalized coordinates to pixel coordinates
        for zone_name, normalized_coords in config.DETECTION_ZONES.items():
            pixel_coords = []
            for x, y in normalized_coords:
                pixel_x = int(x * width)
                pixel_y = int(y * height)
                pixel_coords.append([pixel_x, pixel_y])
            
            self.zone_polygons[zone_name] = np.array(pixel_coords, dtype=np.int32)
        
        print(f"[INFO] Initialized {len(self.zone_polygons)} detection zones")
    
    def set_frame_shape(self, frame_shape):
        """Update frame shape and reinitialize zones"""
        self._initialize_zones(frame_shape)
    
    def analyze_traffic(self, detections, emergency_vehicles=None):
        """
        Analyze traffic density across all zones
        
        Args:
            detections: List of vehicle detections from VehicleDetector
            emergency_vehicles: Optional list of emergency vehicle detections
            
        Returns:
            dict: Traffic analysis for each zone
                  {zone_name: {'count': int, 'density': str, 'vehicles': list,
                               'has_emergency': bool, 'emergency_count': int,
                               'vehicle_types': {'car': int, 'motorcycle': int, 'bus': int, 'truck': int}}}
        """
        if not self.zone_polygons:
            raise ValueError("Zones not initialized. Call set_frame_shape() first.")
        
        # Initialize results
        results = {}
        
        for zone_name, polygon in self.zone_polygons.items():
            # Count vehicles in this zone
            vehicles_in_zone = self._count_vehicles_in_zone(detections, polygon)
            vehicle_count = len(vehicles_in_zone)
            
            # Count vehicles by type in this zone
            vehicle_types_count = self._count_vehicles_by_type(vehicles_in_zone)
            
            # Count emergency vehicles in this zone
            emergency_in_zone = []
            emergency_count = 0
            has_emergency = False
            
            if emergency_vehicles:
                emergency_in_zone = self._count_vehicles_in_zone(emergency_vehicles, polygon)
                emergency_count = len(emergency_in_zone)
                has_emergency = emergency_count > 0
            
            # Classify density level
            density_level = self._classify_density(vehicle_count)
            
            # Store results
            results[zone_name] = {
                'count': vehicle_count,
                'density': density_level,
                'vehicles': vehicles_in_zone,
                'vehicle_types': vehicle_types_count,
                'color': config.DENSITY_COLORS[density_level],
                'has_emergency': has_emergency,
                'emergency_count': emergency_count,
                'emergency_vehicles': emergency_in_zone
            }
            
            # Update history
            self.history[zone_name].append(vehicle_count)
            if len(self.history[zone_name]) > self.max_history_length:
                self.history[zone_name].pop(0)
        
        return results
    
    def _count_vehicles_in_zone(self, detections, polygon):
        """
        Count vehicles whose centers are inside the zone polygon
        
        Args:
            detections: List of detections
            polygon: Zone polygon (numpy array)
            
        Returns:
            list: Detections that are in this zone
        """
        vehicles_in_zone = []
        
        for detection in detections:
            center = detection['center']
            
            # Check if center point is inside polygon
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                vehicles_in_zone.append(detection)
        
        return vehicles_in_zone
    
    def _count_vehicles_by_type(self, detections):
        """
        Count vehicles by their type/class
        
        Args:
            detections: List of detections
            
        Returns:
            dict: Count of each vehicle type {'car': int, 'motorcycle': int, 'bus': int, 'truck': int}
        """
        type_count = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        for detection in detections:
            vehicle_type = detection.get('class', 'unknown').lower()
            
            # Map vehicle types to count dictionary
            if vehicle_type in type_count:
                type_count[vehicle_type] += 1
            elif vehicle_type == 'motorcycle' or vehicle_type == 'bike':
                type_count['motorcycle'] += 1
        
        return type_count
    
    def _classify_density(self, vehicle_count):
        """
        Classify traffic density based on vehicle count
        
        Args:
            vehicle_count: Number of vehicles in zone
            
        Returns:
            str: Density level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if vehicle_count <= config.DENSITY_THRESHOLDS['LOW']:
            return 'LOW'
        elif vehicle_count <= config.DENSITY_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        elif vehicle_count <= config.DENSITY_THRESHOLDS['HIGH']:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_average_density(self, zone_name, window=10):
        """
        Get average vehicle count for a zone over recent frames
        
        Args:
            zone_name: Name of the zone
            window: Number of recent frames to average
            
        Returns:
            float: Average vehicle count
        """
        if zone_name not in self.history or len(self.history[zone_name]) == 0:
            return 0.0
        
        recent = self.history[zone_name][-window:]
        return sum(recent) / len(recent)
    
    def draw_zones(self, frame, analysis_results=None):
        """
        Draw detection zones on frame
        
        Args:
            frame: Input frame
            analysis_results: Optional traffic analysis results to color zones
            
        Returns:
            frame: Frame with drawn zones
        """
        # Create overlay for semi-transparent zones
        overlay = frame.copy()
        
        for zone_name, polygon in self.zone_polygons.items():
            # Determine color
            if analysis_results and zone_name in analysis_results:
                color = analysis_results[zone_name]['color']
                count = analysis_results[zone_name]['count']
                density = analysis_results[zone_name]['density']
            else:
                color = config.COLORS['BLUE']
                count = 0
                density = 'N/A'
            
            # Draw filled polygon (semi-transparent)
            cv2.fillPoly(overlay, [polygon], color)
            
            # Draw polygon outline
            cv2.polylines(frame, [polygon], True, color, 2)
            
            # Add zone label
            # Calculate center of polygon for label
            moments = cv2.moments(polygon)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Zone name
                cv2.putText(
                    frame,
                    zone_name,
                    (cx - 30, cy - 20),
                    config.FONT,
                    0.7,
                    config.COLORS['WHITE'],
                    2
                )
                
                # Vehicle count and density
                label = f"{count} vehicles | {density}"
                cv2.putText(
                    frame,
                    label,
                    (cx - 60, cy + 10),
                    config.FONT,
                    0.5,
                    config.COLORS['WHITE'],
                    1
                )
        
        # Blend overlay with original frame
        alpha = 0.3  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def get_congestion_priority(self, analysis_results):
        """
        Determine which zone has highest priority based on congestion
        
        Args:
            analysis_results: Traffic analysis results
            
        Returns:
            str: Zone name with highest priority
        """
        # Priority order: CRITICAL > HIGH > MEDIUM > LOW
        priority_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        max_priority = 0
        max_count = 0
        priority_zone = None
        
        for zone_name, data in analysis_results.items():
            density = data['density']
            count = data['count']
            priority = priority_order[density]
            
            # Higher priority or same priority but more vehicles
            if priority > max_priority or (priority == max_priority and count > max_count):
                max_priority = priority
                max_count = count
                priority_zone = zone_name
        
        return priority_zone


# Test function
if __name__ == "__main__":
    print("Testing Traffic Analyzer...")
    
    # Create a test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Initialize analyzer
    analyzer = TrafficAnalyzer(test_frame.shape)
    
    # Create some test detections
    test_detections = [
        {'center': (640, 200), 'bbox': [630, 190, 650, 210], 'class': 'car'},  # NORTH
        {'center': (640, 210), 'bbox': [630, 200, 650, 220], 'class': 'car'},  # NORTH
        {'center': (640, 500), 'bbox': [630, 490, 650, 510], 'class': 'car'},  # SOUTH
        {'center': (900, 400), 'bbox': [890, 390, 910, 410], 'class': 'bus'},  # EAST
    ]
    
    # Analyze traffic
    results = analyzer.analyze_traffic(test_detections)
    
    # Print results
    print("\nTraffic Analysis Results:")
    for zone, data in results.items():
        print(f"{zone}: {data['count']} vehicles - Density: {data['density']}")
    
    # Draw zones
    frame = analyzer.draw_zones(test_frame, results)
    
    # Display
    cv2.imshow('Traffic Analysis Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
