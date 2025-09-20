import cv2
import numpy as np
import math
from collections import deque
import json
from scipy.spatial.transform import Rotation as Rot

def quat_mul_wxyz(qA, qB):
    # (w,x,y,z) * (w,x,y,z)
    w1,x1,y1,z1 = qA
    w2,x2,y2,z2 = qB
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

def axis_angle_wxyz(axis: str, deg: float):
    half = np.deg2rad(deg) * 0.5
    c, s = np.cos(half), np.sin(half)
    if axis == 'x': return np.array([c, s, 0.0, 0.0], dtype=np.float32)
    if axis == 'y': return np.array([c, 0.0, s, 0.0], dtype=np.float32)
    if axis == 'z': return np.array([c, 0.0, 0.0, s], dtype=np.float32)
    raise ValueError("axis must be 'x','y','z'")

class Estimate:
    MARKER_SIZE_M = 0.03
    
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
        
        # Measurements
        self.tip_angle, self.prev_tip_angle = 0, []
        self.distance_to_cam, self.prev_distance_to_cam = 0, []
        self.lr_deg, self.lr_deg_buffer = 0, deque(maxlen=10)
        self.ud_deg, self.ud_deg_buffer = 0, deque(maxlen=10)
        
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_inplane_angle(self, rvec):
        """Return marker orientation about camera Z-axis in degrees [-180, 180]."""
        R, _ = cv2.Rodrigues(rvec)
        angle_rad = math.atan2(R[1,0], R[0,0])
        angle_deg = math.degrees(angle_rad)
        return angle_deg # [-180, 180]

    def get_orientation_signs(self, c, frame):
        TL, TR, BR, BL = c
        horizontal_len = (math.hypot(TL[0] - TR[0], TL[1] - TR[1]) +
                        math.hypot(BL[0] - BR[0], BL[1] - BR[1])) / 2
        vertical_len = (math.hypot(TL[0] - BL[0], TL[1] - BL[1]) +
                        math.hypot(TR[0] - BR[0], TR[1] - BR[1])) / 2

        h, w = frame.shape[:2]

        # Axis-aligned bounding box
        xs = np.array([TL[0], TR[0], BR[0], BL[0]], dtype=np.float32)
        ys = np.array([TL[1], TR[1], BR[1], BL[1]], dtype=np.float32)
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        def clamp_rect(x1, y1, x2, y2):
            x1 = max(0, min(w - 1, int(round(x1))))
            x2 = max(0, min(w,     int(round(x2))))
            y1 = max(0, min(h - 1, int(round(y1))))
            y2 = max(0, min(h,     int(round(y2))))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            return x1, y1, x2, y2

        # ROIs
        x1, y1, x2, y2 = clamp_rect(min_x - horizontal_len * 1.5, min_y - vertical_len * 1.5,
                                        max_x + horizontal_len * 1.5, min_y + vertical_len * 1.5)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = frame_rgb[y1:y2, x1:x2]

        def count_pixels_rgb_ratio(rgb_roi):   
            R, G, B = rgb_roi[..., 0].astype(np.float32), rgb_roi[..., 1].astype(np.float32), rgb_roi[..., 2].astype(np.float32)
            R += 1e-6
            G += 1e-6
            B += 1e-6
            blue_mask = ((B > 180) & (B/R > 1.5) & (G/R > 2))
            red_mask = ((R > 150) & (R/B > 2) & (R/G > 2))
            num_pixels = rgb_roi.shape[0] * rgb_roi.shape[1]
            return np.count_nonzero(blue_mask) / num_pixels, np.count_nonzero(red_mask) / num_pixels

        blue_ratio, red_ratio = count_pixels_rgb_ratio(roi)
        x_sign = -1 if red_ratio > 0.02 else 1
        y_sign = -1 if blue_ratio > 0.02 else 1

        return x_sign, y_sign

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

    def get_lr_ud_angles(self, rvec) -> tuple[float, float]:
        """
        From rvec (object->camera rotation), compute:
        lr_deg: left/right (yaw)  (+ right, - left)
        ud_deg: up/down   (pitch) (+ up,    - down)
        Conventions: camera x right, y down, z forward (OpenCV).
        """
        
        Rmat, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rot.from_matrix(Rmat).as_quat()      # (x,y,z,w)
        q_cam = np.array([qw, qx, - qy, qz], dtype=np.float32)  # -> (w,x,y,z)

        q_fix = axis_angle_wxyz('x', 180.0)                   # flip Y (OpenCV y↓ -> GL y↑)
        q_corr = quat_mul_wxyz(q_fix, q_cam)                  # apply correction
        q_corr /= np.linalg.norm(q_corr)                      # normalize
        self.quaternion = q_corr
        
        n = Rmat[:, 2]                          # marker's plane normal in camera coords

        # LR (yaw): angle of normal projected onto XZ plane
        lr_rad = np.arctan2(n[0], n[2])      # + means normal leans toward +x (right)
        # UD (pitch): angle of normal projected onto YZ plane
        ud_rad = np.arctan2(-n[1], n[2])     # + means normal leans toward -y (up)

        lr_deg = float(np.degrees(lr_rad))
        ud_deg = float(np.degrees(ud_rad))
        return lr_deg, ud_deg

    def get_measurements(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            self.aruco.drawDetectedMarkers(frame, corners, ids)

            for c in corners:
                imgp = c.reshape(-1, 2).astype(np.float32)
                # rvec is rotation vector (stores rotation of aruco relative to cam), tvec is translation vector relative to camera
                ok_pnp, rvec, tvec = cv2.solvePnP(self.objp, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                # Orientation angle
                x0, y0 = int(imgp[0,0]), int(imgp[0,1])
                self.tip_angle = self.get_inplane_angle(rvec)
                cv2.putText(frame, f"{self.tip_angle:+.1f}°", (x0, y0-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Distance
                self.distance_to_cam = self.get_distance(c.reshape(4, 2).astype(np.float32))
                cv2.putText(frame, f"Dist {round(self.distance_to_cam, 2)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # LR and UD angles, make more sensitive to right and down
                x_sign, y_sign = self.get_orientation_signs(c.reshape(4, 2).astype(np.float32), frame)
                lr_deg, ud_deg = self.get_lr_ud_angles(rvec)
                
                # Correct for being less sensitive to changes to the right and down
                lr_deg = (180 - abs(lr_deg)) * x_sign
                ud_deg = (180 - abs(ud_deg)) * y_sign
                lr_deg *= 2.0 if x_sign > 0 else 1
                ud_deg *= 2.0 if x_sign < 0 else 1
                
                self.lr_deg_buffer.append(lr_deg)
                self.ud_deg_buffer.append(ud_deg)
                    
                self.lr_deg = float(np.median(self.lr_deg_buffer))
                self.ud_deg = float(np.median(self.ud_deg_buffer))
                
                cv2.putText(frame, f"LR {self.lr_deg:+.1f}°  UD {self.ud_deg:+.1f}°",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
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

