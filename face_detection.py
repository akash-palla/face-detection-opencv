import cv2
import glob

def load_image(file_path):
    return cv2.imread(file_path)

def detect_faces(image, cascade_classifier):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
    input_directory = "*.jpg"
    cascade_classifier_file = "haarcascade_frontalface_default.xml"

    all_img = glob.glob(input_directory)
    cascade_classifier = cv2.CascadeClassifier(cascade_classifier_file)

    for im in all_img:
        image = load_image(im)
        if image is None:
            print(f"Unable to load image: {im}")
            continue

        faces = detect_faces(image, cascade_classifier)
        final_image = image.copy()

        if len(faces) == 0:
            print(f"No faces detected in {im}")
        else:
            draw_faces(final_image, faces)
            cv2.imshow("Face Detection using OpenCV", final_image)
            key = cv2.waitKey(2000)

            if key == 27:  # 27 corresponds to the Esc key
                cv2.destroyAllWindows()
                break
            else:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
