import cv2

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture()

i = 12  # counter for filenames

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show the live video
    cv2.imshow("Press 's' to save, 'q' to quit", frame)
    
    # Check key press
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Press 's' to save photo
        filename = f"selfie_{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Photo saved as {filename}")
        i += 1
        
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()