import cv2, sys, numpy, os,time
cv2.__version__


size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'


(im_width, im_height) = (112, 92)

# Part 1: Create fisherRecognizer
print('Training...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    print (subdirs," ",dirs," ",files)
    for subdir in dirs:
        print (subdir)
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            print (filename)
            path = subjectpath + '/' + filename
            lable = id
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (im_width, im_height))
            images.append(img)
            lables.append(int(lable))
        id += 1

(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, lables)

size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(1)
# frame = cv2.flip(frame,1,0) to flip
print(names)

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        person,pred = model.predict(face_resize)
        if person == 0:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(im, names[person], (int(x -20), int(y - 20)), cv2.FONT_HERSHEY_DUPLEX,2,(0, 255, 0))
        else:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) #rectangle colour changer
            cv2.putText(im, names[person], (int(x -20), int(y - 20)), cv2.FONT_HERSHEY_DUPLEX,2,(0, 0, 255))
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
        cv2.destroyAllWindows()
        
        