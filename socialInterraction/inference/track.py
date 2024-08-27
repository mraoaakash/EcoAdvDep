from ultralytics import YOLO
import os 
import cv2
import argparse
import numpy as np
import pandas as pd

# fish_name = 'rsec-left'
DATAPATH="" # ADD THE GLOBAL DATA PATH HERE

def runner(PRE_POST,video_path, modelpath,savepath, outpath=None):
    video_path = f'{video_path}/{PRE_POST}_intervention/resized'
    for fish_name in ['r10-left.mp4']:#os.listdir(video_path):
        fish_name = fish_name.split('.')[0]
        if fish_name == '.DS_Store':
            continue
        video_path = f'{video_path}/{fish_name}.mp4'
        save_path = f'{DATAPATH}/{PRE_POST}_intervention/tracked'
        procesed = os.listdir(save_path)
        processed = [i.split('.')[0] for i in procesed]
        if fish_name in processed:
            # continue
            pass


        os.makedirs(save_path, exist_ok=True)
        model = YOLO(modelpath)

        cap = cv2.VideoCapture(video_path)
        if outpath is not None:
            out = cv2.VideoWriter(f'{outpath}/{fish_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        
        results = []

        df = pd.DataFrame(columns=['frame', 'x', 'y', 'w', 'h'])
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_arr = np.arange(0, num_frames+1, 1)
        df["frame"] = frame_arr

        # Loop through the video frames
        counter = 0
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()


            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=False)
                boxes = results[0].boxes.xywh.cpu().numpy()

                if len(boxes) > 0:
                    boxes = boxes[0]
                    df.loc[counter] = [counter, boxes[0], boxes[1], boxes[2], boxes[3]]
                else:
                    df.loc[counter] = [counter, 0, 0, 0, 0]


                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Write the frame to the output video
                out.write(annotated_frame)


                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)


                # Break the loop if 'q' is {PRE_POST}ssed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
            counter += 1

        # Release the video capture object and close the display window
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        df.to_csv(f'{save_path}/{fish_name}.csv', index=False)


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--PRE_POST', type=str, help='pre or post')
    # argparser.add_argument('--video_path', type=str, help='path to video')
    # argparser.add_argument('--modelpath', type=str, help='path to model')
    # argparser.add_argument('--savepath', type=str, help='path to save')
    # argparser.add_argument('--outpath', type=str, help='path to save', default=None)
    # args = argparser.parse_args()

    PRE_POST = "" # ADD THE PRE OR POST HERE
    video_path = "" # ADD THE PATH TO VIDEO HERE
    modelpath = "" # ADD THE PATH TO MODEL HERE
    savepath = "" # ADD THE PATH TO SAVE HERE
    outpath = "" # ADD THE PATH TO SAVE HERE
    runner(PRE_POST, video_path, modelpath, savepath, outpath)
    # runner('pre', 'data/pre_intervention/``