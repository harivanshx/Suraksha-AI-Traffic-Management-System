from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)

# Global camera instance
# In a production app, use a better way to manage this resource
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera(source=0) # Use 0 for webcam, or path to MP4
    return camera

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    cam = get_camera()
    return jsonify(cam.get_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
