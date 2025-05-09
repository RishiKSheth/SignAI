import mediapipe
import cv2
from mediapipe.python.solutions.hands import Hands
from mediapipe.tasks.python.vision.face_stylizer import FaceStylizer

cap = cv2.VideoCapture(0)
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_drawing = mediapipe.solutions.drawing_utils
mp_hands = mediapipe.solutions.hands

hands = mp_hands.Hands()
drawing = mp_drawing.DrawingSpec()
#face_stylizer = mp_face_stylizer.FaceStylizer()

def detectHumanHands(frame_rgb):
    results = hands.process(frame_rgb)
    height, width, _ = frame_rgb.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0),
                                       thickness=2, circle_radius=2)
            )
            #hand landmarks y (tips)
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            #conversion to pixels
            thumb_tip_y_pixels = thumb_tip_y * frame.shape[0]
            index_finger_tip_y_pixels = index_finger_tip_y * frame.shape[0]
            middle_finger_tip_y_pixels = middle_finger_tip_y * frame.shape[0]
            ring_finger_tip_y_pixels = ring_finger_tip_y * frame.shape[0]
            pinky_finger_tip_y_pixels = pinky_finger_tip_y * frame.shape[0]
            #The Letter Case A
            if (thumb_tip_y_pixels < index_finger_tip_y_pixels and
                    thumb_tip_y_pixels < middle_finger_tip_y_pixels and
                    thumb_tip_y_pixels < ring_finger_tip_y_pixels and
                    thumb_tip_y_pixels < pinky_finger_tip_y_pixels):
                print('THE LETTER A')
            #The Middle Finger
            if (middle_finger_tip_y_pixels < index_finger_tip_y_pixels and
                    middle_finger_tip_y_pixels < ring_finger_tip_y_pixels and
                    middle_finger_tip_y_pixels < pinky_finger_tip_y_pixels and
                    middle_finger_tip_y_pixels < thumb_tip_y_pixels):
                print("FUCK YOU ")


            #draws circle in for loop
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            center_x_px = int(center_x * frame_rgb.shape[1])
            center_y_px = int(center_y * frame_rgb.shape[0])
            cv2.circle(frame_rgb, (center_x_px, center_y_px), 155, (0,0,0), 10)


    return frame_rgb

while True:
    ret, frame = cap.read()
    if(cv2.waitKey(1) == ord('q')):
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = detectHumanHands(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", frame_bgr)

cap.release()
cv2.destroyAllWindows()
