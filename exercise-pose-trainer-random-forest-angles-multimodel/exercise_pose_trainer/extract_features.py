from itertools import combinations
import json
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

    def __mul__(self, scalar: float):
        return Point3d(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

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


class CoordinateSystem3D:
    def __init__(self, origin: Point3d, x_dir: Point3d, y_dir: Point3d):
        self.origin = origin
        self.x_axis = x_dir.normalize()
        self.z_axis = self.x_axis.cross(y_dir).normalize()
        self.y_axis = self.z_axis.cross(
            self.x_axis).normalize()  # Re-orthogonalize y

    def to_local(self, point: Point3d) -> Point3d:
        # Vetor do ponto em relação à origem do sistema
        relative = point - self.origin

        # Projeção nos eixos do sistema local
        x_local = relative.dot(self.x_axis)
        y_local = relative.dot(self.y_axis)
        z_local = relative.dot(self.z_axis)

        return Point3d(x_local, y_local, z_local)

    def to_global(self, point_local: Point3d) -> Point3d:
        # Conversão do ponto local para coordenadas globais
        global_vector = (
            self.x_axis * point_local.x +
            self.y_axis * point_local.y +
            self.z_axis * point_local.z
        )
        return self.origin + global_vector


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


canonical_system = CoordinateSystem3D(
    Point3d(0, 0, 0), Point3d(1, 0, 0), Point3d(0, 1, 0))


def get_custom_system(landmarks):
    left_wrist_point = Point3d.from_landmark(
        landmarks[PoseLandmark.LEFT_WRIST])
    right_wrist_point = Point3d.from_landmark(
        landmarks[PoseLandmark.RIGHT_WRIST])
    wrist_mid_point = left_wrist_point.get_mid_point(right_wrist_point)

    left_shoulder_point = Point3d.from_landmark(
        landmarks[PoseLandmark.LEFT_SHOULDER])
    right_shoulder_point = Point3d.from_landmark(
        landmarks[PoseLandmark.RIGHT_SHOULDER])
    shoulder_mid_point = left_shoulder_point.get_mid_point(
        right_shoulder_point)

    left_foot_index_point = Point3d.from_landmark(
        landmarks[PoseLandmark.LEFT_FOOT_INDEX])
    right_foot_index_point = Point3d.from_landmark(
        landmarks[PoseLandmark.RIGHT_FOOT_INDEX])
    foot_index_mid_point = left_foot_index_point.get_mid_point(
        right_foot_index_point)

    origin = foot_index_mid_point
    x_dir = wrist_mid_point - foot_index_mid_point
    y_dir = shoulder_mid_point - wrist_mid_point
    return CoordinateSystem3D(origin, x_dir, y_dir)


def get_feature_from_joints_triplet(landmarks, triplet, degrees=False, normalize=True):
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


def extract_features(landmarks, features: list[str] | list[list[str]]):
    if isinstance(features[0], str):
        triplets = list(combinations(features, 3))
    else:
        triplets = features

    angles = []
    for triplet in triplets:
        angles.append(get_feature_from_joints_triplet(landmarks, triplet))
    return angles


def load_features(base_path: str):
    with open(os.path.join(base_path, "labels.json"), "r") as f:
        labels_dict = json.load(f)

    classes_features = labels_dict["classes_features"]

    classes = list(classes_features.keys())
    features = {}
    labels = {}
    for c in classes:
        features[c] = []
        labels[c] = []

    imgs_labels = labels_dict["labels"]
    imgs_path = os.path.join(base_path, "images")
    for img_file in os.listdir(imgs_path):
        if not is_img_file(img_file):
            continue
        if img_file not in imgs_labels:
            print(f"Warning: {img_file} not found in labels.json")
            continue

        img_labels = imgs_labels[img_file]

        img_path = os.path.join(imgs_path, img_file)
        landmarks1, landmarks2 = (get_landmarks(img_path),
                                  get_landmarks(img_path, mirror=True))
        landmarks = [landmarks1, landmarks2]

        for landmark in landmarks:
            if landmark:
                for c in classes:
                    if len(img_labels) == 0:
                        features[c].append(extract_features(
                            landmark, classes_features[c]["angles"]))
                        labels[c].append("correct")
                    elif c == "full_body" or c in img_labels:
                        features[c].append(extract_features(
                            landmark, classes_features[c]["angles"]))
                        labels[c].append("incorrect")

    return features, labels, classes_features
