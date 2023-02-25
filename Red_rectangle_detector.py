import cv2
import numpy as np
from mss import mss

cap = cv2.VideoCapture(0)
#mon = {'left': 1026, 'top': 518, 'width': 890, 'height': 500}
#ekran = mss()

font = cv2.FONT_HERSHEY_COMPLEX

while (1):
    ret, frame = cap.read()
    # frame=np.array(ekran.grab(mon))
    # frame=frame[:,:,::-1]
    # frame=frame[:,:,1::]
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", threshold)

    # kırmızı renginin üst ve alt renklerini ayarlamak:
    #alt_kirmizi = np.array([161, 155, 84])
    d_red=np.array([157,67,126])
    u_red = np.array([179, 255,209])
    color_range = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Kırmızı rengini maskelemek:
    red_mask = cv2.inRange(color_range, d_red, u_red)
    # maskedeki gürültüyü morphology ile azaltmak
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # maskelediğimiz degeri kırmızı göstermek:
    red = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("kirmizi", red)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draws boundary of contours.
    for cnt in contours:
        # nokta sayısı 3ten fazlam mı
        if (len(cnt) > 3):
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < 5:
                x1, y1, w, h = cv2.boundingRect(approx)
                centerx, centery = int(x1 + w / 2), int(y1 + h / 2)
                # çok küçük şekilleri algılama
                if (50 < w and 50 < h):
                    ratio = float(w) / h
                    if ratio >= 0.9 and ratio <= 1.1:
                        frame = cv2.drawContours(frame, [approx], -1, (0, 255, 255), 3)
                        cv2.putText(frame, 'Square', (x1, y1), font, 0.6, (255, 255, 0), 4)
                        cv2.circle(frame, (centerx, centery), 5, (255, 0, 0), -1)
                    else:
                        cv2.putText(frame, 'Rectangle', (x1, y1), font, 0.6, (0, 255, 0), 4)
                        frame = cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                        # merkezine daire çiz
                        cv2.circle(frame, (centerx, centery), 5, (255, 0, 0), -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()
