"""
Web Processing Module
Handles video/image processing for web interface
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from vehicle_detector import VehicleDetector
from traffic_analyzer import TrafficAnalyzer
from signal_controller import TrafficSignalController
import config


class WebTrafficProcessor:
    """
    Processes traffic data from uploaded videos/images for web interface
    """
    
    def __init__(self):
        """Initialize processing components"""
        self.detector = VehicleDetector()
        self.analyzer = None  # Will be initialized when we know frame size
        self.controller = TrafficSignalController()
        
    def process_direction_video(self, video_path: str, direction: str, max_frames: int = 100) -> Dict:
        """
        Process video for a single direction
        
        Args:
            video_path: Path to video file
            direction: Direction name (NORTH, SOUTH, EAST, WEST)
            max_frames: Maximum frames to process (for performance)
            
        Returns:
            dict: Processing results including vehicle counts, density, and annotated frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize analyzer with frame shape
        if self.analyzer is None:
            self.analyzer = TrafficAnalyzer((frame_height, frame_width))
        
        # Process frames
        all_detections = []
        vehicle_counts = []
        frame_count = 0
        sample_frames = []
        
        # Calculate frame skip for better performance
        frame_skip = max(1, total_frames // max_frames)
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Detect vehicles
            detections = self.detector.detect_vehicles(frame)
            all_detections.extend(detections)
            vehicle_counts.append(len(detections))
            
            # Save sample frames (first, middle, last)
            if frame_count == 0 or frame_count == max_frames // 2 or not ret:
                # Draw detections
                annotated_frame = self._draw_detections(frame.copy(), detections)
                sample_frames.append(annotated_frame)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        avg_vehicles = np.mean(vehicle_counts) if vehicle_counts else 0
        max_vehicles = max(vehicle_counts) if vehicle_counts else 0
        
        # Count vehicles by type
        vehicle_types_count = self._count_vehicles_by_type(all_detections)
        
        # Classify density based on average
        density = self._classify_density(int(avg_vehicles))
        
        return {
            'direction': direction,
            'total_frames_processed': frame_count,
            'average_vehicles': round(avg_vehicles, 2),
            'max_vehicles': int(max_vehicles),
            'density_level': density,
            'sample_frames': sample_frames,
            'all_detections': len(all_detections),
            'vehicle_types': vehicle_types_count,
            'frames_processed': frame_count
        }
    
    def process_direction_image(self, image_path: str, direction: str) -> Dict:
        """
        Process image for a single direction
        
        Args:
            image_path: Path to image file
            direction: Direction name (NORTH, SOUTH, EAST, WEST)
            
        Returns:
            dict: Processing results
        """
        # Read image
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Initialize analyzer with frame shape
        if self.analyzer is None:
            self.analyzer = TrafficAnalyzer(frame.shape[:2])
        
        # Detect vehicles
        detections = self.detector.detect_vehicles(frame)
        vehicle_count = len(detections)
        
        # Count vehicles by type
        vehicle_types_count = self._count_vehicles_by_type(detections)
        
        # Draw detections
        annotated_frame = self._draw_detections(frame.copy(), detections)
        
        # Classify density
        density = self._classify_density(vehicle_count)
        
        return {
            'direction': direction,
            'vehicle_count': vehicle_count,
            'density_level': density,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'vehicle_types': vehicle_types_count
        }
    
    def aggregate_results(self, direction_results: Dict[str, Dict]) -> Dict:
        """
        Aggregate results from all four directions
        
        Args:
            direction_results: Dictionary mapping direction names to their results
            
        Returns:
            dict: Aggregated analysis and signal recommendations
        """
        # Create traffic analysis format
        traffic_analysis = {}
        
        for direction, result in direction_results.items():
            if 'average_vehicles' in result:  # Video processing
                count = int(result['average_vehicles'])
            else:  # Image processing
                count = result['vehicle_count']
            
            # Get vehicle types
            vehicle_types = result.get('vehicle_types', {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0})
            
            traffic_analysis[direction] = {
                'count': count,
                'density': result['density_level'],
                'vehicles': [],
                'vehicle_types': vehicle_types
            }
        
        # Update signal controller
        signal_status = self.controller.update(traffic_analysis)
        
        # Generate recommendations
        recommendations = self._generate_signal_recommendations(traffic_analysis)
        
        return {
            'traffic_analysis': traffic_analysis,
            'signal_status': signal_status,
            'recommendations': recommendations
        }
    
    def _generate_signal_recommendations(self, traffic_analysis: Dict) -> Dict:
        """
        Generate traffic signal timing recommendations
        
        Args:
            traffic_analysis: Traffic analysis for all directions
            
        Returns:
            dict: Signal timing recommendations
        """
        recommendations = {}
        
        for direction, data in traffic_analysis.items():
            density = data['density']
            
            # Get timing from config
            timing = config.SIGNAL_TIMINGS.get(density, config.SIGNAL_TIMINGS['LOW'])
            
            recommendations[direction] = {
                'density_level': density,
                'vehicle_count': data['count'],
                'green_duration': timing['green'],
                'yellow_duration': timing['yellow'],
                'total_duration': timing['green'] + timing['yellow']
            }
        
        # Calculate optimal sequence
        sequence = self._calculate_optimal_sequence(traffic_analysis)
        recommendations['optimal_sequence'] = sequence
        
        return recommendations
    
    def _calculate_optimal_sequence(self, traffic_analysis: Dict) -> List[str]:
        """
        Calculate optimal traffic light sequence based on congestion
        
        Args:
            traffic_analysis: Traffic analysis for all directions
            
        Returns:
            list: Ordered list of directions for signal sequence
        """
        # Priority mapping
        density_priority = {
            'CRITICAL': 4,
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        
        # Sort directions by priority
        sorted_directions = sorted(
            traffic_analysis.items(),
            key=lambda x: (density_priority.get(x[1]['density'], 0), x[1]['count']),
            reverse=True
        )
        
        return [direction for direction, _ in sorted_directions]
    
    def _count_vehicles_by_type(self, detections: List) -> Dict:
        """
        Count vehicles by their type/class
        
        Args:
            detections: List of detection dictionaries
            
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
    
    def _classify_density(self, vehicle_count: int) -> str:
        """
        Classify traffic density based on vehicle count
        
        Args:
            vehicle_count: Number of vehicles
            
        Returns:
            str: Density level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if vehicle_count <= 5:
            return 'LOW'
        elif vehicle_count <= 15:
            return 'MEDIUM'
        elif vehicle_count <= 25:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _draw_detections(self, frame, detections, plates=None):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of vehicle detections
            plates: Optional list of plate detections
            
        Returns:
            frame: Annotated frame
        """
        # Draw vehicle detections
        for detection in detections:
            bbox = detection['bbox']
            vehicle_class = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for vehicles
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )
            
            # Draw label
            label = f"{vehicle_class}: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw license plate detections
        if plates:
            for plate in plates:
                bbox = plate['bbox']
                text = plate.get('text', 'PLATE')
                confidence = plate.get('confidence', 0.0)
                
                # Draw bounding box in yellow
                color = (0, 255, 255)  # Yellow for plates
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    2
                )
                
                # Draw label
                label = f"{text}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    2
                )
        
        return frame
    
    def save_annotated_frame(self, frame, output_path: str):
        """
        Save annotated frame to file
        
        Args:
            frame: Frame to save
            output_path: Output file path
        """
        cv2.imwrite(output_path, frame)
