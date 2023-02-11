import cv2

# โหลดตัวแยกประเภท คนกับแมว Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

# อ่านภาพ
img = cv2.imread('input-1.jpg')

# แปลงภาพเป็นโทนสีเทาและปรับฮิสโตแกรมให้เท่ากัน
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# ตรวจจับใบหน้าในภาพโทนสีเทา
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# ตรวจจับหน้าแมวในภาพโทนสีเทา
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# วาดสี่เหลี่ยมรอบใบหน้าคนหรือก็คือกรอบ และแสดงชื่อ คน
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, 'People', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
# วาดสี่เหลี่ยมรอบหน้าแมวหรือก็คือกรอบ และแสดงชื่อ แมว
for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, 'Cats', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# แสดงผล
cv2.imshow('Faces and Cats', img)
cv2.waitKey(0)
cv2.destroyAllWindows()