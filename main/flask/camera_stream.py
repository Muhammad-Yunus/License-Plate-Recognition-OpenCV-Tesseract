from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = camera.read()
        h, w, c = frame.shape
        h1=int(0.50*h )
        h2=int(0.85*h )
        w1=int(0.3*w )
        w2=int(0.7*w)
        cv2.rectangle(frame, (w1, h1), (w2, h2), (0, 0, 255), 2)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host="0.0.0.0")