import argparse
import cv2

# Parser Arguments
parser = argparse.ArgumentParser()
parser.add_argument("video_path", nargs='?',  help="path to video", default=0)
args = parser.parse_args()

# Open video
cap = cv2.VideoCapture(args.video_path)

if not cap.isOpened():
    print("Cannot open camera/video")
    exit()

# Get video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Text Properties
color_text = (0, 0, 0)
position_text_x = width - 100
position_text_y = 30
position_text = (position_text_x, position_text_y)
text = "Oishik"
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
thickness = 2

(text_width, text_height) = cv2.getTextSize(text, font, font_size, thickness)[0]

color_box = (255, 255, 255)
start_box = (position_text_x, position_text_y - text_height)
end_box = (position_text_x + text_width, position_text_y)

while True:
    ret, original = cap.read()

    if ret:
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Display box
        original = cv2.rectangle(original, start_box, end_box, color_box, -1)
        gray = cv2.rectangle(gray, start_box, end_box, color_box, -1)

        # Add text
        cv2.putText(original, "Oishik", position_text, font, font_size, color_text, thickness)
        cv2.putText(gray, "Oishik", position_text, font, font_size, color_text, thickness)

        # Display the resulting frame
        cv2.imshow('Gray', gray)
        cv2.imshow('Original', original)

        cv2.moveWindow("Gray", 100, 100)
        cv2.moveWindow("Original", 800, 100)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()