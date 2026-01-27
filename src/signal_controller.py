"""
Traffic Signal Controller Module
Dynamically controls traffic signals based on traffic density
"""

import time
from enum import Enum
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
import config


class SignalState(Enum):
    """Traffic signal states"""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class TrafficSignalController:
    """
    Controls traffic signals for a 4-way intersection
    Dynamically adjusts timing based on traffic density
    """
    
    def __init__(self, directions=None):
        """
        Initialize traffic signal controller
        
        Args:
            directions: List of direction names (default: NORTH, SOUTH, EAST, WEST)
        """
        if directions is None:
            directions = list(config.DETECTION_ZONES.keys())
        
        self.directions = directions
        self.num_directions = len(directions)
        
        # Initialize all signals to RED
        self.signals = {direction: SignalState.RED for direction in directions}
        
        # Current active direction (has green light)
        self.active_direction = None
        self.active_direction_index = 0
        
        # Timing
        self.signal_start_time = time.time()
        self.current_green_duration = config.MIN_GREEN_DURATION
        self.current_yellow_duration = config.SIGNAL_TIMINGS['LOW']['yellow']
        
        # State tracking
        self.in_yellow_phase = False
        self.yellow_start_time = None
        
        # Traffic data for decision making
        self.traffic_data = {}
        
        # Statistics
        self.cycle_count = 0
        self.total_wait_time = {}
        
        # Emergency vehicle priority
        self.emergency_active = False
        self.emergency_direction = None
        self.emergency_start_time = None
        
        print(f"[INFO] Traffic signal controller initialized for {self.num_directions} directions")
        
        # Start first signal
        self._activate_next_signal()
    
    def update(self, traffic_analysis):
        """
        Update signal controller with latest traffic analysis
        Manages signal state transitions and emergency vehicle priority
        
        Args:
            traffic_analysis: Traffic analysis results from TrafficAnalyzer
            
        Returns:
            dict: Current signal states and timing info
        """
        self.traffic_data = traffic_analysis
        current_time = time.time()
        
        # Check for emergency vehicles and handle priority
        if config.EMERGENCY_PRIORITY_ENABLED:
            emergency_direction = self._check_emergency_priority(traffic_analysis)
            
            if emergency_direction:
                # Emergency vehicle detected
                if not self.emergency_active or emergency_direction != self.emergency_direction:
                    # New emergency or different direction
                    print(f"[EMERGENCY] Emergency vehicle detected in {emergency_direction}!")
                    self._activate_emergency_priority(emergency_direction)
                    return self.get_status()
        
        # Check if it's time to change signal
        if self.in_yellow_phase:
            # In yellow phase
            yellow_elapsed = current_time - self.yellow_start_time
            
            if yellow_elapsed >= self.current_yellow_duration:
                # Yellow phase complete, move to next green
                self._activate_next_signal()
        else:
            # In green phase
            green_elapsed = current_time - self.signal_start_time
            
            # Check if minimum green time has elapsed
            if green_elapsed >= config.MIN_GREEN_DURATION:
                # Check if we should extend or switch
                if green_elapsed >= self.current_green_duration:
                    # Time to switch to yellow
                    self._start_yellow_phase()
        
        # Return current state
        return self.get_status()
    
    def _activate_next_signal(self):
        """Activate green signal for next direction in sequence"""
        # Set all to RED first
        for direction in self.directions:
            self.signals[direction] = SignalState.RED
        
        # Move to next direction
        self.active_direction_index = (self.active_direction_index + 1) % self.num_directions
        self.active_direction = self.directions[self.active_direction_index]
        
        # Set active direction to GREEN
        self.signals[self.active_direction] = SignalState.GREEN
        
        # Calculate green duration based on traffic density
        self.current_green_duration = self._calculate_green_duration(self.active_direction)
        
        # Reset timing
        self.signal_start_time = time.time()
        self.in_yellow_phase = False
        
        # Update cycle count
        if self.active_direction_index == 0:
            self.cycle_count += 1
        
        print(f"[SIGNAL] {self.active_direction} → GREEN (duration: {self.current_green_duration}s)")
    
    def _start_yellow_phase(self):
        """Start yellow phase for current active direction"""
        if self.active_direction:
            self.signals[self.active_direction] = SignalState.YELLOW
            self.in_yellow_phase = True
            self.yellow_start_time = time.time()
            
            print(f"[SIGNAL] {self.active_direction} → YELLOW")
    
    def _calculate_green_duration(self, direction):
        """
        Calculate optimal green light duration based on traffic density
        
        Args:
            direction: Direction name
            
        Returns:
            int: Green light duration in seconds
        """
        # Default duration if no traffic data
        if not self.traffic_data or direction not in self.traffic_data:
            return config.MIN_GREEN_DURATION
        
        # Get density level for this direction
        density_level = self.traffic_data[direction]['density']
        
        # Get base duration from config
        base_duration = config.SIGNAL_TIMINGS[density_level]['green']
        
        # Apply constraints
        duration = max(config.MIN_GREEN_DURATION, min(base_duration, config.MAX_GREEN_DURATION))
        
        # Optional: Adjust based on other lanes
        # If other lanes are also congested, reduce duration slightly
        other_lanes_congestion = self._get_other_lanes_congestion(direction)
        if other_lanes_congestion > 2:  # More than 2 other lanes are congested
            duration = int(duration * 0.8)  # Reduce by 20%
        
        return duration
    
    def _check_emergency_priority(self, traffic_analysis):
        """
        Check if any direction has emergency vehicles
        
        Args:
            traffic_analysis: Traffic analysis results
            
        Returns:
            str: Direction with emergency vehicle, or None
        """
        for direction, data in traffic_analysis.items():
            if data.get('has_emergency', False):
                return direction
        return None
    
    def _activate_emergency_priority(self, emergency_direction):
        """
        Activate emergency priority for a direction
        Immediately switch to yellow, then green for emergency direction
        
        Args:
            emergency_direction: Direction with emergency vehicle
        """
        if emergency_direction == self.active_direction:
            # Already green for emergency direction, extend duration
            self.current_green_duration = max(
                self.current_green_duration,
                config.EMERGENCY_MIN_GREEN_DURATION
            )
            self.emergency_active = True
            self.emergency_direction = emergency_direction
            self.emergency_start_time = time.time()
            print(f"[EMERGENCY] Extending green for {emergency_direction} to {self.current_green_duration}s")
        else:
            # Different direction, start yellow phase then switch
            self.emergency_active = True
            self.emergency_direction = emergency_direction
            self.emergency_start_time = time.time()
            
            # Force yellow phase with emergency override delay
            self.current_yellow_duration = config.EMERGENCY_OVERRIDE_DELAY
            self._start_yellow_phase()
            print(f"[EMERGENCY] Switching to {emergency_direction} in {config.EMERGENCY_OVERRIDE_DELAY}s")
    
    def _get_other_lanes_congestion(self, current_direction):
        """
        Count how many other lanes have MEDIUM or higher congestion
        
        Args:
            current_direction: Current active direction
            
        Returns:
            int: Number of other congested lanes
        """
        congested_count = 0
        
        for direction, data in self.traffic_data.items():
            if direction != current_direction:
                if data['density'] in ['MEDIUM', 'HIGH', 'CRITICAL']:
                    congested_count += 1
        
        return congested_count
    
    def get_status(self):
        """
        Get current status of all signals
        
        Returns:
            dict: Status information including states, timings, and active direction
        """
        current_time = time.time()
        
        # Calculate remaining time
        if self.in_yellow_phase:
            elapsed = current_time - self.yellow_start_time
            remaining = max(0, self.current_yellow_duration - elapsed)
        else:
            elapsed = current_time - self.signal_start_time
            remaining = max(0, self.current_green_duration - elapsed)
        
        return {
            'signals': self.signals.copy(),
            'active_direction': self.active_direction,
            'phase': 'YELLOW' if self.in_yellow_phase else 'GREEN',
            'time_remaining': round(remaining, 1),
            'time_elapsed': round(elapsed, 1),
            'green_duration': self.current_green_duration,
            'cycle_count': self.cycle_count
        }
    
    def get_signal_color(self, direction):
        """
        Get BGR color for a direction's signal state
        
        Args:
            direction: Direction name
            
        Returns:
            tuple: BGR color
        """
        state = self.signals.get(direction, SignalState.RED)
        
        if state == SignalState.GREEN:
            return config.COLORS['GREEN']
        elif state == SignalState.YELLOW:
            return config.COLORS['YELLOW']
        else:
            return config.COLORS['RED']
    
    def force_next_signal(self):
        """Force immediate transition to next signal (for manual control/testing)"""
        self._start_yellow_phase()
    
    def get_statistics(self):
        """
        Get controller statistics
        
        Returns:
            dict: Statistics including cycle count, average wait times, etc.
        """
        return {
            'cycle_count': self.cycle_count,
            'active_direction': self.active_direction,
            'total_directions': self.num_directions
        }


# Test function
if __name__ == "__main__":
    print("Testing Traffic Signal Controller...")
    
    # Initialize controller
    controller = TrafficSignalController()
    
    # Simulate traffic data
    test_traffic_data = {
        'NORTH': {'count': 20, 'density': 'HIGH', 'vehicles': []},
        'SOUTH': {'count': 5, 'density': 'LOW', 'vehicles': []},
        'EAST': {'count': 12, 'density': 'MEDIUM', 'vehicles': []},
        'WEST': {'count': 8, 'density': 'MEDIUM', 'vehicles': []}
    }
    
    # Simulate for 2 minutes
    print("\nSimulating signal control for 60 seconds...")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    try:
        while time.time() - start_time < 60:
            # Update controller
            status = controller.update(test_traffic_data)
            
            # Print status
            print(f"\r[{status['cycle_count']}] {status['active_direction']} = {status['phase']} "
                  f"| Remaining: {status['time_remaining']}s   ", end='', flush=True)
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped")
    
    # Print final statistics
    stats = controller.get_statistics()
    print(f"\nCompleted {stats['cycle_count']} signal cycles")
