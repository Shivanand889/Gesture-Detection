import cv2
import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import img_to_array
import os
# Load the MobileNet model
model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to extract features from an image using the MobileNet model
def extract_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Function to calculate cosine similarity between two feature vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to process the gesture input (image or video)
def process_gesture_input(input_path, model):
    if input_path.endswith('.mp4'):
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read the video file.")
        return extract_features(frame, model)
    else:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Could not read the image file.")
        return extract_features(image, model)

def main():
    # Hardcoded paths to input files (gesture representation and test video)
    current_directory = os.getcwd()
    print(current_directory)
    path1 = os.path.join(current_directory+ '\inputs\ges2.jpg')
    path2 = os.path.join(current_directory+ '\inputs\ges.mp4')
    gesture_file_path = path1  # Specify the path to your gesture image
    test_video_path = path2  # Specify the path to your test video
    # Process gesture input
    gesture_features = process_gesture_input(gesture_file_path, model)

    # Process test video
    cap = cv2.VideoCapture(test_video_path)
    threshold = 0.6
    detected_frames = []
    a = 0 
    b = 0 
    s= 0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame_features = extract_features(frame, model)
        similarity = cosine_similarity(gesture_features, frame_features)
        s = s+ similarity
        if similarity > threshold:
            a = a+1
            cv2.putText(frame, 'DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frames.append(frame)
        else :
            b = b+1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Create a directory to store detected frames
    output_folder = r'C:\Users\hp\OneDrive\Desktop\g'
    os.makedirs(output_folder, exist_ok=True)

    # Save detected frames as images
    for i, frame in enumerate(detected_frames):
        output_path = os.path.join(output_folder, f"detected_frame_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved frame {i} as {output_path}")

    print(a)
    print(b)
    print(s/(a+b))

if __name__ == "__main__":
    main()
