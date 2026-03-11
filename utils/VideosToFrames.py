import cv2
import os

# Save 5 frames per video
def videos_to_frames1(videos_path, frame_path, num_frames):
    # Ensure the frame folder exists
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # Iterate through all files in the video folder
    for filename in os.listdir(videos_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video formats as needed
            video_path = os.path.join(videos_path, filename)

            # Initialize video capture object
            cap = cv2.VideoCapture(video_path)

            # Check if the video opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video {filename}.")
                continue

            # Frame counter
            frame_count = 0

            # Create a folder for frames of the current video
            video_frame_folder = os.path.join(frame_path, filename.split('.')[0])
            if not os.path.exists(video_frame_folder):
                os.makedirs(video_frame_folder)

            # Read frames from the video
            while True:
                # Read the next frame
                ret, frame = cap.read()

                # If frame is read correctly, ret is True
                if not ret:
                    print(f"Done extracting all frames from {filename}.")
                    break

                # If less than num_frames per video have been saved, save this frame
                if frame_count < num_frames:
                    frame_filename = f'{video_frame_folder}/{frame_count}.png'
                    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(frame_filename, frame)
                    print(
                        f"Frame {frame_count} from video {filename[:-4]} has been saved")
                    frame_count += 1
                else:
                    # If we've saved the desired number of frames, break the loop
                    break
            # Release the video capture object
            cap.release()

    print("All frames have been extracted and saved.")

# Save 5 frames per second
def videos_to_frames2(videos_path, frame_path, num_frames, total_seconds=None):
    # Ensure the frame folder exists
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # Iterate through all files in the video folder
    for filename in os.listdir(videos_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video formats as needed
            video_path = os.path.join(videos_path, filename)

            # Initialize video capture object
            cap = cv2.VideoCapture(video_path)

            # Check if the video opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video {filename}.")
                continue

            # Frame counter
            frame_count = 0
            # Current second counter
            current_second = 0
            # Frame counter within each second
            frames_per_second_count = 0

            # Create a folder for frames of the current video
            video_frame_folder = os.path.join(frame_path, filename.split('.')[0])
            if not os.path.exists(video_frame_folder):
                os.makedirs(video_frame_folder)

            # Get the frame rate of the video
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Frames per second: {fps}")

            # Read frames from the video
            while True:
                # Read the next frame
                ret, frame = cap.read()

                # If frame is read correctly, ret is True
                if not ret:
                    print(f"Done extracting all frames from {filename}.")
                    break

                # Update the frame counter and second counter
                if frame_count % int(fps) == 0:
                    frames_per_second_count = 0
                    current_second += 1
                    if total_seconds is not None and current_second > total_seconds:
                        break

                # If less than num_frames per second have been saved, save this frame
                if frames_per_second_count < num_frames:
                    frame_filename = f'{video_frame_folder}/{current_second}_{frames_per_second_count}.png'
                    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(frame_filename, frame)
                    print(
                        f"Frame {frames_per_second_count} of second {current_second} from video {filename[:-4]} has been saved")
                    frames_per_second_count += 1
                frame_count += 1

            # Release the video capture object
            cap.release()

    print("All frames have been extracted and saved.")


if __name__ == "__main__":
    # Path to the video files
    videos_path = r'..\..\share\2023-ICCV-DRUVA-videos'
    # Path to the extracted frames
    save_path = r'..\dataset\test\DRUVA'
    # Number of saved frames in per second
    num_frames = 5
    # videos_to_frames1(videos_path, save_path, num_frames)
    total_seconds = 10
    videos_to_frames2(videos_path, save_path, num_frames, total_seconds)
