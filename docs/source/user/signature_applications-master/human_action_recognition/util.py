"""
Utility functions for landmark-based human action recognition using path signatures
"""

import requests

import cv2
from moviepy.editor import VideoFileClip, clips_array
import numpy as np
from tqdm.notebook import tqdm

def download(source_url, target_filename, chunk_size=1024):
    """
    Download a file via HTTP.

    Parameters
    ----------
    source_url: str
        URL of the file to download
    target_filename: str
        Target filename
    chunk_size: int
        Chunk size
    """
    response = requests.get(source_url, stream=True)
    file_size = int(response.headers['Content-Length'])

    with open(target_filename, 'wb') as handle:
        for data in tqdm(response.iter_content(chunk_size=chunk_size),
                         total=int(file_size / chunk_size), unit='KB',
                         desc='Downloading dataset:'):
            handle.write(data)

class Skeleton():
    """
    Skeleton representation for visualisation and animation consisting of dots
    representing landmarks and lines representing connections.
    """

    # When transforming key points, use a fraction of the image size as a minimum of empty,
    # black region around the skeleton
    IMG_BORDER_FRAC = 0.1

    def __init__(self,
                 target_width,
                 target_height,
                 radius=10,
                 confidence_coord=None,
                 draw_connections=True,
                 transform_keypoints=False):
        """
        Parameters
        ----------
        target_width: int
            Image width for visualisation
        target_height: int
            Image height for visualisation
        radius: int
            Landmark radius
        confidence_coord: int
            Index of confidence estimates
        draw_connections: bool
            Whether to draw connections between the key points
         transform_keypoints: bool
            Whether to transform key points
        """
        self.landmarks = [(255, 42, 0), (255, 113, 0), (255, 14, 0),
                          (56, 255, 0), (28, 0, 255), (184, 255, 0),
                          (155, 0, 255), (99, 255, 0), (70, 0, 255),
                          (212, 255, 0), (198, 0, 255), (141, 255, 0),
                          (127, 0, 255), (240, 255, 0), (226, 0, 255)]

        self.connections = [(14, 10), (10, 6), (12, 8), (8, 4), (13, 9),
                            (9, 5), (11, 7), (7, 3), (0, 1), (4, 0), (3, 0),
                            (2, 0), (6, 5), (6, 1), (5, 1)]
        self.radius = radius
        self.confidence_coord = confidence_coord
        self.draw_connections = draw_connections
        self.target_width = target_width
        self.target_height = target_height
        self.transform_keypoints = transform_keypoints

    def draw(self, keypoints):

        """
        Plot a static keypoint image.

        Parameters
        ----------
        keypoints: numpy array
            Keypoints to be drawn
        """
        img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        keypoints = keypoints[:, 0:2].copy()
        if self.transform_keypoints:
            keypoints = self._transform_keypoints(keypoints)

        keypoints = np.around(keypoints).astype(np.int32)

        if self.draw_connections:
            self._draw_connections(keypoints, img)

        self._draw_landmarks(keypoints, img)

        return img

    def _draw_landmarks(self, keypoints, img):
        for i, colour in enumerate(self.landmarks):
            # Skip points with confidence 0 (usually means not detected)
            if self.confidence_coord is not None and keypoints[i, self.confidence_coord] == 0:
                continue
            point = tuple(keypoints[i])
            if point[0] != 0 or point[1] != 0:
                cv2.circle(img, point[0:2], self.radius, colour, -1)

    def _draw_connections(self, keypoints, img):
        for i, j in self.connections:
            # Skip connections where at least one of the points has
            # confidence 0 (usually means not detected)
            if self.confidence_coord is not None:
                if keypoints[i, self.confidence_coord] == 0 or \
                        keypoints[j, self.confidence_coord] == 0:
                    continue
            pt1 = tuple(keypoints[i, 0:2])
            pt2 = tuple(keypoints[j, 0:2])
            if (pt1[0] != 0 or pt1[1] != 0) and (pt2[0] != 0 or pt2[1] != 0):
                colour = (int((self.landmarks[i][0] + self.landmarks[j][0]) / 2),
                          int((self.landmarks[i][1] + self.landmarks[j][1]) / 2),
                          int((self.landmarks[i][2] + self.landmarks[j][2]) / 2))
                cv2.line(img, pt1, pt2, colour, max(int(self.radius / 2), 1))

    def _transform_keypoints(self, keypoints):
        keypoints -= np.amin(keypoints, axis=0)
        keypoint_scale = np.amax(keypoints, axis=0)
        keypoints *= np.array((self.target_width, self.target_height)) * (
            1 - 2 * Skeleton.IMG_BORDER_FRAC) / keypoint_scale
        keypoints += np.array((self.target_width, self.target_height)) * Skeleton.IMG_BORDER_FRAC

        return keypoints

    def animate(self, keypoints, filename=None, fps=25, codec='XVID'):
        """
        Convert key points to a animation and output to a video file.

        Parameters
        ----------
        keypoints: numpy array
            Array of keypoints in the form [frame,landmark,coords]
        filename: string, optional (default is None)
            If given the video is saved to the specified file
        fps: float
            Number of frames per second
        codec: str
            Video codec represented in fourcc format
        """
        if filename is not None:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vid_file = cv2.VideoWriter(filename, fourcc, fps,
                                       (self.target_width, self.target_height))

        vid = np.zeros((keypoints.shape[0], self.target_height, self.target_width, 3),
                       dtype=np.uint8)

        for i in range(vid.shape[0]):
            vid[i] = self.draw(keypoints[i])
            if filename is not None:
                frame = cv2.cvtColor(vid[i], cv2.COLOR_RGB2BGR)
                vid_file.write(frame)
        if filename is not None:
            vid_file.release()

        return vid

def display_animation(filename, keypoints, clip_height=768, display_height=240,
                      temp_file='__temp__.avi', include_source_video=True):
    """
    Display a side-by-side animation comprising the skeleton and (optionally) the source
    video.

    Parameters
    ----------
    filename : str
        The video file to read
    keypoints : numpy array
        Array of keypoints in the form [frame,landmark,coords]
    clip_height : int
        Desired clip height (applies to both source video and skeleton, the former is upscaled)
    display_height : int
        Desired display height
    temp_file : int
        Temporary file for transcoding
    include_source_video: bool
        Whether to include the source video
    """
    clip_original = VideoFileClip(filename)
    rescaling_factor = clip_height / clip_original.h
    clip_original = clip_original.resize(height=clip_height)
    keypoints = np.copy(keypoints) * rescaling_factor

    skeleton = Skeleton(target_width=clip_original.w, target_height=clip_original.h)
    skeleton.animate(keypoints, temp_file, fps=len(keypoints)/clip_original.duration)
    clip_skeleton = VideoFileClip(temp_file)
    if include_source_video:
        clip = clips_array([[clip_original, clip_skeleton]])
    else:
        clip = clip_skeleton

    return clip.ipython_display(height=display_height, rd_kwargs=dict(logger=None))
