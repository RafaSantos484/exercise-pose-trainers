import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np

from .utils import is_img_file

PoseLandmark = mp.solutions.pose.PoseLandmark  # type: ignore
pose_model = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=open(
            "pose_landmarker_full.task", "rb").read()),
        # base_options=python.BaseOptions(model_asset_buffer=open("pose_landmarker_heavy.task", "rb").read()),
        running_mode=vision.RunningMode.IMAGE,
    )
)


class Point3d:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_landmark(landmark):
        return Point3d(landmark.x, landmark.y, landmark.z)

    def __add__(self, other):
        return Point3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def to_list(self):
        return [self.x, self.y, self.z]

    def get_mid_point(self, other):
        return Point3d((self.x + other.x) / 2, (self.y + other.y) / 2, (self.z + other.z) / 2)

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        return Point3d(self.x / n, self.y / n, self.z / n)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Point3d(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def get_angle(self, other, degrees=False):
        dot_product = self.dot(other)
        norm_self = self.norm()
        norm_other = other.norm()
        angle_rad = math.acos(dot_product / (norm_self * norm_other))
        if not degrees:
            return angle_rad
        return math.degrees(angle_rad)


def get_landmarks(image_path: str, mirror=False):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mirror:
        image_rgb = cv2.flip(image_rgb, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = pose_model.detect(mp_image)
    if not result.pose_world_landmarks:
        print(f"No landmarks found for {image_path}")
        return None

    # landmarks = result.pose_landmarks[0]
    landmarks = result.pose_world_landmarks[0]

    return landmarks


def invert_landmarks(landmarks):
    if landmarks is None:
        return None

    landmarks_inverted = {}
    landmarks_inverted[PoseLandmark.LEFT_WRIST] = landmarks[PoseLandmark.RIGHT_WRIST]
    landmarks_inverted[PoseLandmark.RIGHT_WRIST] = landmarks[PoseLandmark.LEFT_WRIST]
    landmarks_inverted[PoseLandmark.LEFT_ELBOW] = landmarks[PoseLandmark.RIGHT_ELBOW]
    landmarks_inverted[PoseLandmark.RIGHT_ELBOW] = landmarks[PoseLandmark.LEFT_ELBOW]
    landmarks_inverted[PoseLandmark.LEFT_SHOULDER] = landmarks[PoseLandmark.RIGHT_SHOULDER]
    landmarks_inverted[PoseLandmark.RIGHT_SHOULDER] = landmarks[PoseLandmark.LEFT_SHOULDER]
    landmarks_inverted[PoseLandmark.LEFT_HIP] = landmarks[PoseLandmark.RIGHT_HIP]
    landmarks_inverted[PoseLandmark.RIGHT_HIP] = landmarks[PoseLandmark.LEFT_HIP]
    landmarks_inverted[PoseLandmark.LEFT_KNEE] = landmarks[PoseLandmark.RIGHT_KNEE]
    landmarks_inverted[PoseLandmark.RIGHT_KNEE] = landmarks[PoseLandmark.LEFT_KNEE]
    landmarks_inverted[PoseLandmark.LEFT_ANKLE] = landmarks[PoseLandmark.RIGHT_ANKLE]
    landmarks_inverted[PoseLandmark.RIGHT_ANKLE] = landmarks[PoseLandmark.LEFT_ANKLE]
    landmarks_inverted[PoseLandmark.LEFT_FOOT_INDEX] = landmarks[PoseLandmark.RIGHT_FOOT_INDEX]
    landmarks_inverted[PoseLandmark.RIGHT_FOOT_INDEX] = landmarks[PoseLandmark.LEFT_FOOT_INDEX]

    return landmarks_inverted


def get_angle_from_joints_triplet(landmarks, triplet, degrees=False, normalize=True):
    a = Point3d.from_landmark(landmarks[PoseLandmark[triplet[0]]])
    b = Point3d.from_landmark(landmarks[PoseLandmark[triplet[1]]])
    c = Point3d.from_landmark(landmarks[PoseLandmark[triplet[2]]])

    vec1 = b - a
    vec2 = c - b
    angle = vec1.get_angle(vec2, degrees=degrees)
    if normalize:
        if degrees:
            angle /= 180
        else:
            angle /= np.pi

    return angle


def extract_features(landmarks):
    sides_triplets = [
        [["LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER"],
         ["RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER"],],

        [["LEFT_WRIST", "LEFT_ELBOW", "RIGHT_ELBOW"],
         ["RIGHT_WRIST", "RIGHT_ELBOW", "LEFT_ELBOW"],],

        [["LEFT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
         ["RIGHT_WRIST", "RIGHT_SHOULDER", "LEFT_SHOULDER"],],

        [["LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
         ["RIGHT_ELBOW", "RIGHT_SHOULDER", "LEFT_SHOULDER"],],

        [["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP"],
         ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"],],

        [["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_KNEE"],
         ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_KNEE"],],

        [["LEFT_WRIST", "LEFT_SHOULDER", "LEFT_ANKLE"],
         ["RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_ANKLE"],],

        [["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
         ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],],

        [["LEFT_SHOULDER", "LEFT_KNEE", "LEFT_ANKLE"],
         ["RIGHT_SHOULDER", "RIGHT_KNEE", "RIGHT_ANKLE"],],

        [["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
         ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],],

        [["LEFT_ANKLE", "LEFT_SHOULDER", "RIGHT_SHOULDER"],
         ["RIGHT_ANKLE", "RIGHT_SHOULDER", "LEFT_SHOULDER"],],

        [["LEFT_ANKLE", "LEFT_HIP", "RIGHT_HIP"],
         ["RIGHT_ANKLE", "RIGHT_HIP", "LEFT_HIP"],],

        [["LEFT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE"],
         ["RIGHT_ANKLE", "RIGHT_KNEE", "LEFT_KNEE"],],
    ]

    features = []
    for left_side_triplet, right_side_triplet in sides_triplets:
        left_angle = get_angle_from_joints_triplet(
            landmarks, left_side_triplet)
        right_angle = get_angle_from_joints_triplet(
            landmarks, right_side_triplet)
        diff = left_angle - right_angle
        features.append([left_angle, right_angle, diff])

    return features


def load_features(base_path: str):
    features = []
    labels = []
    imgs_landmarks = {}
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if label in ["__pycache__"] or not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            if not is_img_file(img_file):
                continue

            if img_file not in imgs_landmarks:
                img_path = os.path.join(label_path, img_file)
                imgs_landmarks[img_file] = (get_landmarks(
                    img_path), get_landmarks(img_path, mirror=True))

            landmarks1, landmarks2 = imgs_landmarks[img_file]
            landmarks3 = invert_landmarks(landmarks1)
            landmarks4 = invert_landmarks(landmarks2)
            landmarks = [landmarks1, landmarks2, landmarks3, landmarks4]

            for landmark in landmarks:
                if landmark:
                    features.append(extract_features(landmark))
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels
