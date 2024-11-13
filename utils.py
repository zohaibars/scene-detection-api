import imghdr
import cv2
import os 
def check_file_type(file_path):
    # Check if the file is an image using imghdr
    image_type = imghdr.what(file_path)
    print(image_type)
    if image_type:
        return "image"

    # Check if the file is a video using OpenCV
    cap = cv2.VideoCapture(file_path)
    is_video = cap.isOpened()
    cap.release()
    if is_video:
        return "video"

    # If neither an image nor a video, return None
    return "file format not support"
# check_file_type(os.path.join('TESTData','video','e.mp4'))
def extract_frames(video_path, output_dir="temp"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Unable to open video file"

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize variables
    frame_count = 0
    success = True
    # Extract filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create folder using the filename
    folder_path = os.path.join(output_dir, filename_without_extension)
    os.makedirs(folder_path, exist_ok=True)
    # Read and save frames
    frames_path=[]
    image_name=0
    while success:
        success, frame = cap.read()
        if frame_count % int(fps) == 0:
            frame_path = os.path.join(folder_path, f"frame_{image_name}.jpg")
            cv2.imwrite(frame_path, frame)
            image_name += 1
            frames_path.append(frame_path)
        frame_count += 1

    # Release the video capture object
    cap.release()
    os.remove(video_path)
    return {"folder_path":folder_path,"all_frames":frames_path}


# video_path = os.path.join('TESTData','video','e.mp4')

# print(extract_frames(video_path))
