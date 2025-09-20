import cv2
import numpy as np
import math
from collections import deque
import json
from geometry_viewer import GeometryViewer

class Estimate:
    MARKER_SIZE_M = 0.03
    WIDTH = 1280
    HEIGHT = 720
    
    def __init__(self):
        # Camera intrinsics 
        with open("calib.json", "r") as f:
            self.data = json.load(f)
        self.K = np.array([[self.data["fx"], 0, self.data["cx"]],
                    [0, self.data["fy"], self.data["cy"]],
                    [0,  0,  1]], dtype=np.float64)
        self.dist = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        
        self.aruco = cv2.aruco
        dic = self.aruco.getPredefinedDictionary(self.aruco.DICT_4X4_50)
        params = self.aruco.DetectorParameters()
        self.detector = self.aruco.ArucoDetector(dic, params)
        
        half = self.MARKER_SIZE_M / 2.0
        self.objp = np.array([
            [-half,  half, 0.0],  # TL
            [ half,  half, 0.0],  # TR
            [ half, -half, 0.0],  # BR
            [-half, -half, 0.0],  # BL
        ], dtype=np.float32)
        
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.d, self.alpha = 0, 0
        
        self.viewer = GeometryViewer("Degree and Pos Visualizer")

    def get_inplane_angle(self, rvec):
        """Return marker orientation about camera Z-axis in degrees [-180, 180]."""
        R, _ = cv2.Rodrigues(rvec)
        angle_rad = math.atan2(R[1,0], R[0,0])
        angle_deg = math.degrees(angle_rad)
        return angle_deg # [-180, 180]

    # Estimate distance (z) from marker to cam
    def get_distance(self, c):
        TL, TR, BR, BL = c

        # Average side length in pixels
        side_px = (np.linalg.norm(TR-TL) + np.linalg.norm(TR-BR) +
                np.linalg.norm(BR-BL) + np.linalg.norm(BL-TL)) / 4.0

        fx = self.K[0,0]  # Focal length in pixels
        # Pinhole model: size_in_pixels = (f * real_size) / Z
        Z = (fx * self.MARKER_SIZE_M) / side_px
        return Z * 1.39 # 1.39 determined experimentally
    
    def get_degree_in_game(self, rvec, tvec, frame, ok_pnp):
        point_3d = np.array([[0, 0, 0]], dtype=np.float32)  # marker center
        point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, self.K, self.dist)

        x, y = point_2d.ravel()
        cv2.circle(frame, (int(x), int(y)), 20, (0, 0, 255), 10)
        
        # Get ratio of x pos / total width, convert to a degree by doing ratio * 180
        artistic = 0.80 # This factor makes it so the gun doesn't actually move that much, for better artisitc value
        r = 0.4 # Radius
        self.d = (x / self.WIDTH * r - r / 2) * artistic
        arc_deg = x / self.WIDTH * 180 - 90
      
        self.alpha, _ = self.viewer.update(r=0.4, d=self.d, arc_deg=arc_deg)

    def get_measurements(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None and len(ids) > 0:
            self.aruco.drawDetectedMarkers(frame, corners, ids)
            for c in corners:
                c = c.reshape(-1, 2).astype(np.float32)
                # rvec is rotation vector (stores rotation of aruco relative to cam), tvec is translation vector relative to camera
                ok_pnp, rvec, tvec = cv2.solvePnP(self.objp, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                self.get_degree_in_game(rvec, tvec, frame, ok_pnp)
                
        return frame

if __name__ == "__main__":
    estimator = Estimate()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    kernel = np.ones((5,5), np.uint8)
    
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        frame = estimator.get_measurements(frame)

        cv2.imshow("Cam Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()