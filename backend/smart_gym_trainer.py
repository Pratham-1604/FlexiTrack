from flask import Flask, request, send_file
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import tempfile
import math as m
from keras.models import load_model
import os
# from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)
api = Api(app)

import cv2
import numpy as np
import mediapipe as mp
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils
def detectPose(image_pose, pose):
    if image_pose is None:
      return None, None
    final_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    if resultant.pose_landmarks:
        mp_drawing.draw_landmarks(image=final_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                       thickness=2, circle_radius=2))
    return resultant.pose_landmarks, final_image
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return angle
def generate_video(images):
    if images is None or images[0] is None:
      return []
    height, width, layers = images[0].shape
    data = []
    for image in images:
        landmarks, res_image = detectPose(image, pose_image)
        if landmarks:
          data.append(landmarks.landmark)
    return data
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    images=[]
    while success:
      success, image = vidObj.read()
      images.append(image)
    vidObj.release()
    return images

def correction(path):
  video_frames = FrameCapture(path)
  video_data = generate_video(video_frames)
  final_video=[]
  flag_r=0
  if video_data[0][mp_pose.PoseLandmark.RIGHT_ANKLE].z<=video_data[0][mp_pose.PoseLandmark.LEFT_ANKLE].z:
    flag_r=1
  height, width, layers = video_frames[0].shape
  video = cv2.VideoWriter('video.avi',
                        cv2.VideoWriter_fourcc(*'MP4V'), 24,(width, height))
  for i in range(len(video_frames)-1):
        # print(i)
        if flag_r==1:
            a = video_data[i][mp_pose.PoseLandmark.RIGHT_ANKLE]
            s = video_data[i][mp_pose.PoseLandmark.RIGHT_SHOULDER]
            w = video_data[i][mp_pose.PoseLandmark.RIGHT_WRIST]
            h = video_data[i][mp_pose.PoseLandmark.RIGHT_HIP]
            k = video_data[i][mp_pose.PoseLandmark.RIGHT_KNEE]
        else:
            a = video_data[i][mp_pose.PoseLandmark.LEFT_ANKLE]
            s = video_data[i][mp_pose.PoseLandmark.LEFT_SHOULDER]
            w = video_data[i][mp_pose.PoseLandmark.LEFT_WRIST]
            h = video_data[i][mp_pose.PoseLandmark.LEFT_HIP]
            k = video_data[i][mp_pose.PoseLandmark.LEFT_KNEE]
        a_x=int(a.x*video_frames[i].shape[1])
        a_y=int(a.y*video_frames[i].shape[0])
        s_x=int(s.x*video_frames[i].shape[1])
        s_y=int(s.y*video_frames[i].shape[0])
        w_x=int(w.x*video_frames[i].shape[1])
        w_y=int(w.y*video_frames[i].shape[0])
        final_frame = video_frames[i].copy()
        cv2.line(final_frame, (s_x, s_y), (a_x,a_y), (0, 0, 255), 2)
        cv2.line(final_frame, (s_x, s_y), (s_x,w_y), (0, 0, 255), 2)
        # print(calculate_angle([h.x,h.y,h.z],[k.x,k.y,k.z],[a.x,a.y,a.z]))
        # print(calculate_angle([k.x,k.y,k.z],[h.x,h.y,h.z],[s.x,s.y,s.z]))
        if not(160<=calculate_angle([h.x,h.y,h.z],[k.x,k.y,k.z],[a.x,a.y,a.z])<=195):
        # print('Incorrect At Knee')
            final_frame = cv2.putText(final_frame, 'Incorrect At Knee', (100,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, 2)
        if not(160<=calculate_angle([k.x,k.y,k.z],[h.x,h.y,h.z],[s.x,s.y,s.z])<=195):
        # print('Incorrect At Hip')
            final_frame = cv2.putText(final_frame, 'Incorrect At Back', (100,100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, 2)
        # final_video.append(final_frame)
        final_video.append(detectPose(final_frame, pose_image)[1])
  for image in final_video:
        video.write(image)
  return video

def pushups(path):
    encoder = load_model('encoder.h5')
    decoder = load_model('decoder.h5')
    input_frames = FrameCapture(path)
    height, width, layers = input_frames[0].shape
    video = cv2.VideoWriter('video.avi',
                            cv2.VideoWriter_fourcc(*'MP4V'), 40,(width, height))
    for z in input_frames:
        input_data_landmarks, _ = detectPose(z, pose_image)
        if input_data_landmarks is None: continue
        input_data = []
        for j in input_data_landmarks.landmark:
            input_data.append([j.x,j.y,j.z])
        encoded_imgs = encoder.predict(np.array(input_data).reshape(1,99))
        decoded_imgs = decoder.predict(encoded_imgs)
        ans_ = decoded_imgs[0].reshape(33,3)
        # for center in ans_:
        #   cx,cy=center[0]*z.shape[1],center[1]*z.shape[0]
        #   cv2.circle(z, (int(cx), int(cy)), 2, (0,0,255), 2)
        #   video.write(z)
        cv2.line(z, (int(ans_[11][0]*z.shape[1]),int(ans_[11][1]*z.shape[0])),(int(ans_[13][0]*z.shape[1]),int(ans_[13][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[13][0]*z.shape[1]),int(ans_[13][1]*z.shape[0])),(int(ans_[15][0]*z.shape[1]),int(ans_[15][1]*z.shape[0])),(0,0,255), 2)

        cv2.line(z, (int(ans_[12][0]*z.shape[1]),int(ans_[12][1]*z.shape[0])),(int(ans_[14][0]*z.shape[1]),int(ans_[14][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[14][0]*z.shape[1]),int(ans_[14][1]*z.shape[0])),(int(ans_[16][0]*z.shape[1]),int(ans_[16][1]*z.shape[0])),(0,0,255), 2)

        cv2.line(z, (int(ans_[12][0]*z.shape[1]),int(ans_[12][1]*z.shape[0])),(int(ans_[24][0]*z.shape[1]),int(ans_[24][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[11][0]*z.shape[1]),int(ans_[11][1]*z.shape[0])),(int(ans_[23][0]*z.shape[1]),int(ans_[23][1]*z.shape[0])),(0,0,255), 2)

        cv2.line(z, (int(ans_[24][0]*z.shape[1]),int(ans_[24][1]*z.shape[0])),(int(ans_[26][0]*z.shape[1]),int(ans_[26][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[26][0]*z.shape[1]),int(ans_[26][1]*z.shape[0])),(int(ans_[28][0]*z.shape[1]),int(ans_[28][1]*z.shape[0])),(0,0,255), 2)

        cv2.line(z, (int(ans_[23][0]*z.shape[1]),int(ans_[23][1]*z.shape[0])),(int(ans_[25][0]*z.shape[1]),int(ans_[25][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[25][0]*z.shape[1]),int(ans_[25][1]*z.shape[0])),(int(ans_[27][0]*z.shape[1]),int(ans_[27][1]*z.shape[0])),(0,0,255), 2)

        cv2.line(z, (int(ans_[11][0]*z.shape[1]),int(ans_[11][1]*z.shape[0])),(int(ans_[12][0]*z.shape[1]),int(ans_[12][1]*z.shape[0])),(0,0,255), 2)
        cv2.line(z, (int(ans_[23][0]*z.shape[1]),int(ans_[23][1]*z.shape[0])),(int(ans_[24][0]*z.shape[1]),int(ans_[24][1]*z.shape[0])),(0,0,255), 2)

        err = np.mean(((np.array(input_data)-np.array(ans_))*100)**2)
        if err>=10:
            z=cv2.putText(z, 'Incorrect', (100,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, 2)
        else:
            z=cv2.putText(z, 'Correct', (100,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, 2)
        video.write(detectPose(z, pose_image)[1])
    video.release()
    return video

def squats(path):
    def findAngle(x1, y1, x2, y2):
        theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
        degree = int(180/m.pi)*theta
        return degree
    red = (50, 50, 255)

    # Initialize mediapipe pose class.
    pose = mp_pose.Pose()
    def FrameCapture(path):
        cap = cv2.VideoCapture(path)
        count = 0
        success = 1
        images=[]
        while success:
            success, image = cap.read()
            if not success:
                break
            # Get fps.
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Get height and width of the frame.
            h, w = image.shape[:2]
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image.
            keypoints = pose.process(image)
            # Convert the image back to BGR.
            lm = keypoints.pose_landmarks
            lmPose  = mp_pose.PoseLandmark


            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Left Knee.
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

            # Left ankle.
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)



            # Calculate angles
            ang1 = findAngle(l_hip_x,l_hip_y,l_shldr_x,l_shldr_y)
            ang2 = findAngle(l_knee_x,l_knee_y,l_hip_x,l_hip_y)
            ang3 = findAngle(l_ankle_x,l_ankle_y,l_knee_x,l_knee_y)

            if ang1 < 20:
                cv2.putText(image,"bend forward",(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            if ang1 > 45:
                cv2.putText(image,"bend backwards",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            if ang2 > 50 and ang2 < 80:
                cv2.putText(image,"lower your hips",(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            if ang3 > 30:
                cv2.putText(image,"knee falling over toe",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            if ang2 > 95:
                cv2.putText(image,"squat too deep",(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, red, 2)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, red, 2)
            cv2.circle(image, (l_knee_x, l_knee_y), 7, red, 2)
            cv2.circle(image, (l_ankle_x, l_ankle_y), 7, red, 2)

            cv2.putText(image,str(ang1),(l_hip_x, l_hip_y),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)
            cv2.putText(image,str(ang2),(l_knee_x, l_knee_y),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)
            cv2.putText(image,str(ang3),(l_ankle_x, l_ankle_y),cv2.FONT_HERSHEY_SIMPLEX,1,red,2, cv2.LINE_AA)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            images.append(image)
        cap.release()
        return images

    video_output = FrameCapture(path)
    height, width, layers = video_output[0].shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MP4V'), 24,(width, height))
    for image in video_output:
        video.write(image)
    video.release()
    return video

input_model = api.model('InputModel', {
    'choice': fields.Integer(description='The choice (1, 2, or 3)'),
    'video_file': fields.Raw(description='The video file (MP4)'),
})

parser = reqparse.RequestParser()
parser.add_argument('choice', type=int, help='1: The Plank \n 2: The Pushups \n 3: The Squats', required=True)
parser.add_argument('video_file', type=FileStorage, location='files', required=True)

@api.route('/upload')
class ProcessVideo(Resource):
    @api.expect(parser, validate=True)
    def post(self):
        try:
           os.remove('video.avi')
        except:
           pass
        data = parser.parse_args()
        choice = data['choice']
        video_file = data['video_file']

        # Create a temporary file to save the uploaded video content
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_file.save(temp_file)
            temp_file_path = temp_file.name

        # Perform the desired operations based on 'choice'
        if choice == 1:
             correction(temp_file_path)
        elif choice == 2:
            pushups(temp_file_path)
        else:
            squats(temp_file_path)

        return send_file('video.avi', as_attachment=True)


if __name__ == '__main__':
    app.run()
