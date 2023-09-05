import cv2
from util import get_parking_spots_bboxes, empty_or_not, calc_diff
import numpy as np

mask = './samples/mask_1920_1080.png'
video_path = './samples/parking_1920_1080.mp4'



mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)
# Bounding boxes
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
ret = True
frame_idx =  0
spots_status = [None for j in spots]
diffs = [None for j in spots]
prev_frame = None


while ret:
    ret, frame = cap.read()

    if frame_idx %30 == 0 and prev_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1,y1,w,h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_idx] = calc_diff(spot_crop, prev_frame[y1:y1+h, x1:x1+w, :])
        # print([diffs[j] for j in np.argsort(diffs)][::-1])



    if frame_idx % 30 == 0:
        if prev_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in arr_:
            spot = spots[spot_idx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status

# Set previous frame
    if frame_idx % 30 == 0:
        prev_frame = frame.copy()
# Draw rectangle frames
    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]
        if spot_status:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (255, 0, 0), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_idx += 1


cap.release()
cv2.destroyAllWindows()    