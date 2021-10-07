import cv2
import matplotlib.plt as plt

detector = cv2.CascadeClassifier('./haarcascade_smile.xml')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    if ret==False:
        continue
    
    all_smiles = detector.detectMultiScale(frame, 1.5, 45)
    
    sorted_smile = sorted(all_smiles, key = lambda s: s[2]*s[3])  # s stores x, y, w, h so we sort the ractangles on basis od decreasing area.
    
#     for smile in all_smiles:
    if sorted_smile:
        x, y, w, h = sorted_smile[0]
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Smile Detector", frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break;
        
cam.release()
cv2.destroyAllWindows()