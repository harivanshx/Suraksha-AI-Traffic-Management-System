"""
Flask Web Application for AI Traffic Management System
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import json
from pathlib import Path
import shutil
import cv2
import base64
from datetime import datetime

from web_processor import WebTrafficProcessor

app = Flask(__name__)

# Use SECRET_KEY from environment or .env, fallback to fixed key for development
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('uploads/results')
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = str(RESULTS_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename, file_type='video'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    else:
        return ext in ALLOWED_VIDEO_EXTENSIONS or ext in ALLOWED_IMAGE_EXTENSIONS


def get_session_folder():
    """Get or create session folder"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_folder = UPLOAD_FOLDER / 'sessions' / session['session_id']
    session_folder.mkdir(parents=True, exist_ok=True)
    
    return session_folder


def cleanup_old_sessions():
    """Clean up old session folders (older than 1 hour)"""
    import time
    sessions_folder = UPLOAD_FOLDER / 'sessions'
    
    if not sessions_folder.exists():
        return
    
    current_time = time.time()
    for session_dir in sessions_folder.iterdir():
        if session_dir.is_dir():
            # Check if older than 1 hour
            if current_time - session_dir.stat().st_mtime > 3600:
                shutil.rmtree(session_dir, ignore_errors=True)


@app.route('/')
def landing():
    """Landing page"""
    # Cleanup old sessions on page load
    cleanup_old_sessions()
    
    return render_template('landing.html')


@app.route('/traffic')
def traffic():
    """Traffic analysis page"""
    # Reset session for new traffic analysis
    session.pop('session_id', None)
    session.pop('uploaded_files', None)
    session.pop('processing_mode', None)
    
    return render_template('index.html')


@app.route('/api/session-test', methods=['GET'])
def session_test():
    """Debug endpoint to test session functionality"""
    # Set a test value
    session['test_value'] = f'Session test at {datetime.now()}'
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': 'Session test value set',
        'session_id': request.cookies.get('session', 'NO COOKIE'),
        'session_content': dict(session),
        'has_test_value': 'test_value' in session
    })


@app.route('/accident')
def accident():
    """Accident detection page"""
    return render_template('accident.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        direction = request.form.get('direction', '').upper()
        
        if direction not in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            return jsonify({'error': 'Invalid direction'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Determine file type
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            file_type = 'video'
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            file_type = 'image'
        else:
            return jsonify({'error': 'Invalid file type. Allowed: video (mp4, avi, mov, mkv) or image (jpg, png)'}), 400
        
        # Check consistency with processing mode
        if 'processing_mode' in session:
            if session['processing_mode'] != file_type:
                return jsonify({'error': f'Please upload all {session["processing_mode"]} files'}), 400
        else:
            session['processing_mode'] = file_type
        
        # Save file
        session_folder = get_session_folder()
        direction_folder = session_folder / direction.lower()
        direction_folder.mkdir(exist_ok=True)
        
        filepath = direction_folder / filename
        file.save(filepath)
        
        # Track uploaded files
        if 'uploaded_files' not in session:
            session['uploaded_files'] = {}
        
        session['uploaded_files'][direction] = str(filepath)
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'File uploaded for {direction}',
            'direction': direction,
            'filename': filename,
            'file_type': file_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_simulation():
    """Process traffic simulation"""
    try:
        # Debug logging
        print(f"\n{'='*60}")
        print(f"DEBUG: /process endpoint called")
        print(f"DEBUG: Session ID cookie: {request.cookies.get('session', 'NOT SET')}")
        print(f"DEBUG: Session content keys: {list(session.keys())}")
        
        if 'uploaded_files' not in session or not session['uploaded_files']:
            error_msg = 'No files uploaded. Please upload files for all 4 directions first, then click Run Simulation.'
            print(f"DEBUG ERROR: {error_msg}")
            print(f"DEBUG: Current session data: {dict(session)}")
            print(f"{'='*60}\n")
            return jsonify({
                'error': error_msg,
                'debug_info': {
                    'session_exists': bool(session),
                    'has_uploaded_files': 'uploaded_files' in session
                }
            }), 400
        
        uploaded_files = session['uploaded_files']
        processing_mode = session.get('processing_mode', 'video')
        
        # Initialize processor
        processor = WebTrafficProcessor()
        
        # Process each direction
        direction_results = {}
        
        for direction, filepath in uploaded_files.items():
            if processing_mode == 'video':
                result = processor.process_direction_video(filepath, direction)
            else:
                result = processor.process_direction_image(filepath, direction)
            
            direction_results[direction] = result
        
        # Aggregate results
        aggregated = processor.aggregate_results(direction_results)
        
        # Save processed images
        results_folder = RESULTS_FOLDER / session['session_id']
        results_folder.mkdir(parents=True, exist_ok=True)
        
        processed_images = {}
        
        for direction, result in direction_results.items():
            if processing_mode == 'video':
                # Save sample frames
                for idx, frame in enumerate(result.get('sample_frames', [])):
                    output_path = results_folder / f'{direction.lower()}_frame_{idx}.jpg'
                    processor.save_annotated_frame(frame, str(output_path))
                    
                    if idx == 0:  # Use first frame as representative
                        processed_images[direction] = str(output_path)
            else:
                # Save annotated image
                output_path = results_folder / f'{direction.lower()}_annotated.jpg'
                processor.save_annotated_frame(result['annotated_frame'], str(output_path))
                processed_images[direction] = str(output_path)
        
        # Prepare results for display
        results_data = {
            'processing_mode': processing_mode,
            'direction_results': {},
            'recommendations': aggregated['recommendations'],
            'processed_images': processed_images
        }
        
        for direction, result in direction_results.items():
            if processing_mode == 'video':
                results_data['direction_results'][direction] = {
                    'average_vehicles': result['average_vehicles'],
                    'max_vehicles': result['max_vehicles'],
                    'density_level': result['density_level'],
                    'frames_processed': result['total_frames_processed'],
                    'vehicle_types': result.get('vehicle_types', {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0})
                }
            else:
                results_data['direction_results'][direction] = {
                    'vehicle_count': result['vehicle_count'],
                    'density_level': result['density_level'],
                    'vehicle_types': result.get('vehicle_types', {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0})
                }
        
        # Save results to session
        session['results'] = results_data
        session.modified = True
        
        return jsonify({
            'success': True,
            'redirect': url_for('results')
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    """Display results page"""
    if 'results' not in session:
        return redirect(url_for('index'))
    
    results_data = session['results']
    
    # Convert images to base64 for display
    images_base64 = {}
    for direction, image_path in results_data['processed_images'].items():
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                images_base64[direction] = f'data:image/jpeg;base64,{img_data}'
    
    results_data['images_base64'] = images_base64
    
    return render_template('results.html', results=results_data)


@app.route('/reset', methods=['POST'])
def reset():
    """Reset session and start new simulation"""
    # Cleanup session folder
    if 'session_id' in session:
        session_folder = UPLOAD_FOLDER / 'sessions' / session['session_id']
        if session_folder.exists():
            shutil.rmtree(session_folder, ignore_errors=True)
        
        results_folder = RESULTS_FOLDER / session['session_id']
        if results_folder.exists():
            shutil.rmtree(results_folder, ignore_errors=True)
    
    # Clear session
    session.clear()
    
    return jsonify({'success': True, 'redirect': url_for('traffic')})


@app.route('/accident/analyze', methods=['POST'])
def analyze_accident():
    """Analyze uploaded image for accidents"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG, BMP'}), 400
        
        # Save file temporarily
        session_folder = get_session_folder()
        filepath = session_folder / 'accident_image' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        file.save(filepath)
        
        # Initialize processor and detect vehicles
        processor = WebTrafficProcessor()
        
        # Read image
        frame = cv2.imread(str(filepath))
        
        if frame is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        # Detect vehicles
        detections = processor.detector.detect_vehicles(frame)
        
        # Detect license plates
        plates = processor.detector.detect_license_plates(frame, use_ocr=True)
        
        # Draw detections on frame (both vehicles and plates)
        annotated_frame = processor._draw_detections(frame.copy(), detections, plates)
        
        # Save annotated image
        results_folder = RESULTS_FOLDER / session['session_id']
        results_folder.mkdir(parents=True, exist_ok=True)
        annotated_path = results_folder / 'accident_annotated.jpg'
        cv2.imwrite(str(annotated_path), annotated_frame)
        
        # Convert annotated image to base64
        with open(annotated_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
            annotated_base64 = f'data:image/jpeg;base64,{img_data}'
        
        # Simple accident detection heuristic:
        # If multiple vehicles detected close together, potential accident
        accident_detected = False
        if len(detections) >= 2:
            # Check if any two vehicles are overlapping or very close
            for i, det1 in enumerate(detections):
                for det2 in detections[i+1:]:
                    # Check bounding box overlap/proximity
                    x1_1, y1_1, x2_1, y2_1 = det1['bbox']
                    x1_2, y1_2, x2_2, y2_2 = det2['bbox']
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
                    overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
                    
                    if overlap_x > 0 and overlap_y > 0:
                        accident_detected = True
                        break
                if accident_detected:
                    break
        
        # Prepare vehicle data
        vehicles = []
        for det in detections:
            vehicles.append({
                'class': det['class'],
                'confidence': det['confidence']
            })
        
        return jsonify({
            'success': True,
            'accident_detected': accident_detected,
            'vehicles': vehicles,
            'plates': plates,
            'annotated_image': annotated_base64
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chatbot():
    """Handle chatbot messages using Gemini API with traffic context"""
    try:
        import google.generativeai as genai
        
        # Get Gemini API key from environment
        gemini_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_key:
            return jsonify({'error': 'Gemini API key not configured'}), 500
        
        genai.configure(api_key=gemini_key)
        
        # Get request data
        user_message = request.json.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        # Get traffic data from session
        traffic_data = session.get('results', {})
        
        # Build context from traffic analysis
        context = _build_traffic_context(traffic_data)
        
        # Create system prompt with traffic context
        system_prompt = f"""You are a traffic management assistant helping analyze and explain traffic patterns.
You have access to real-time traffic analysis data from a 4-way intersection.

CURRENT TRAFFIC DATA:
{context}

Guidelines:
- Provide concise, clear answers about traffic conditions
- Suggest signal timing adjustments based on congestion
- Answer questions about specific directions and vehicle types
- Use the provided data to support your recommendations
- Be professional and helpful"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(
            f"{system_prompt}\n\nUser Question: {user_message}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )
        
        bot_response = response.text
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except ImportError:
        return jsonify({'error': 'Google Generative AI library not installed. Install with: pip install google-generativeai'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chatbot error: {str(e)}'}), 500


def _build_traffic_context(traffic_data):
    """Build human-readable traffic context from analysis data"""
    if not traffic_data:
        return "No traffic data available."
    
    context_lines = []
    direction_results = traffic_data.get('direction_results', {})
    recommendations = traffic_data.get('recommendations', {})
    
    for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
        if direction in direction_results and direction in recommendations:
            result = direction_results[direction]
            rec = recommendations[direction]
            
            # Vehicle counts
            if 'average_vehicles' in result:
                vehicle_count = f"Average: {result['average_vehicles']}, Peak: {result['max_vehicles']}"
            else:
                vehicle_count = f"Count: {result.get('vehicle_count', 0)}"
            
            # Vehicle types
            vehicle_types = result.get('vehicle_types', {})
            types_str = ", ".join([f"{k}: {v}" for k, v in vehicle_types.items() if v > 0])
            
            # Recommendations
            context_lines.append(
                f"{direction}: "
                f"Density={rec.get('density_level', 'N/A')}, "
                f"Vehicles={vehicle_count}, "
                f"Types=[{types_str}], "
                f"Green={rec.get('green_duration', 0)}s, "
                f"Yellow={rec.get('yellow_duration', 0)}s"
            )
    
    return "\n".join(context_lines) if context_lines else "Traffic data not fully loaded."


if __name__ == '__main__':
    # Get port from environment variable (for deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print("=" * 60)
    print("AI TRAFFIC MANAGEMENT SYSTEM - WEB INTERFACE")
    print("=" * 60)
    print(f"\nStarting Flask server on port {port}...")
    print(f"Mode: {'Development' if debug_mode else 'Production'}")
    print(f"Access the application at: http://127.0.0.1:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
