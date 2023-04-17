import cv2

# Load the pre-trained Haar Cascade classifier for human detection
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Capture the video stream from the webcam
cap = cv2.VideoCapture('in.avi')

while True:
    # Read each frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for human detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect human bodies in the frame using the Haar Cascade classifier
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected human bodies
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Show the resulting frame with human detection
    cv2.imshow('Human Detection', frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
