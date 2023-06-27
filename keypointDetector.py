import tensorflow as tf
import numpy as np
import cv2
import os

class PoseEstimator:
    def __init__(self, folderPath="tensorflowModels/leapTd/", disable_tensrflow_feedback=True):
        self.folderPath = folderPath
        self.frozen_graph_path = folderPath + 'frozen_graph.pb'
        self.info_path = folderPath + 'info.json'
        
        if disable_tensrflow_feedback:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load frozen graph and create TensorFlow session
        with tf.io.gfile.GFile(self.frozen_graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.graph_def = graph_def
        self.session = tf.compat.v1.Session()
        self.session.graph.as_default()
        tf.import_graph_def(self.graph_def, name='')

        # Get input and output tensor name
        self.input_tensor_name = self.session.graph.get_operations()[0].name + ':0'
        self.output_tensor_name = self.session.graph.get_operations()[-1].name + ':0'
        # Get input tensor shape
        self.input_tensor_shape = self.session.graph.get_tensor_by_name(self.input_tensor_name).shape.as_list()

        # Check input tensor shape
        if len(self.input_tensor_shape) != 4:
            raise ValueError(f'Expected input tensor to have shape [batch, height, width, channels], but got shape: {self.input_tensor_shape}')


    def load_video(self, video_path="videos/channel1.mp4"):
        # Load input video
        cap = cv2.VideoCapture(video_path)
        # Define list to store images
        images = []
        # Loop through video frames and convert to images
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Preprocess input image, Resize to input shape [height, width]
                image = cv2.resize(
                    frame, 
                    (self.input_tensor_shape[2], self.input_tensor_shape[1])
                )  
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # Convert to uint8 data type
                image = image.astype(np.uint8)
                # Add batch dimension
                image = np.expand_dims(image, axis=0)
                images.append(image)
            else:
                break
        # Release video capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, image: np.ndarray, visualize_output=False) -> np.ndarray:
        # Get the input and output tensors
        input_tensor = self.session.graph.get_tensor_by_name(self.input_tensor_name)
        output_tensor = self.session.graph.get_tensor_by_name(self.output_tensor_name)

        # Run the session and get the output
        output = self.session.run(output_tensor, feed_dict={input_tensor: image})

        if visualize_output:
            '''
            Index; keypoint - 0: Left Claw - 1: Left Elbow - 2: Left eye 
            3: Tail - 4: Right Claw - 5: Right Elbow - 6: Right eye
            TODO This part has been moved to the pipeline, write drawing function 
            and use same function both here and in the pipeline
            '''
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600, 480)
            # Colors used for drawing keypoints
            colors = [(0, 0, 255), 
                    (0, 255, 0), 
                    (255, 0, 0), 
                    (255, 255, 0), 
                    (255, 0, 255), 
                    (0, 255, 255), 
                    (255, 255, 255),
                    ]
            for lobster in output[0]:
                for i in range(0, output.shape[2] - 1):
                    if not np.isnan(lobster[i][0]) and not np.isnan(lobster[i][1]):
                        cv2.circle(image[0], (int(lobster[i][0]), int(lobster[i][1])), 2, colors[i], -1)
            cv2.imshow('image', image[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return output




