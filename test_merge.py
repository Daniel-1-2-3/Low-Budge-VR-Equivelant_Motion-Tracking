from Game.main import Game
from estimate_legacy import Estimate
import cv2
import numpy as np

def get_quaternion(tip_angle_deg: float, lr_deg: float, ud_deg: float):
    """
    Build a quaternion (x, y, z, w) from:
      tip_angle_deg : roll about Z (in-plane)
      lr_deg        : yaw  about Y (+ right)
      ud_deg        : pitch about X (+ up)

    Composition: R = Rz(roll) * Rx(pitch) * Ry(yaw)
    Returns (qx, qy, qz, qw)
    """
    def _axis_angle_quat(axis, angle_rad):
        s = np.sin(angle_rad * 0.5)
        c = np.cos(angle_rad * 0.5)
        if axis == 'x': return (s, 0.0, 0.0, c)
        if axis == 'y': return (0.0, s, 0.0, c)
        if axis == 'z': return (0.0, 0.0, s, c)
        raise ValueError("axis must be 'x','y','z'")

    def _qmul(q1, q2):
        x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    def _qnorm(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        return tuple((q / (n if n > 0 else 1.0)).tolist())

    # Degrees -> radians
    roll_z  = np.deg2rad(tip_angle_deg)
    yaw_y   = np.deg2rad(lr_deg)
    pitch_x = np.deg2rad(ud_deg)

    # Per-axis quats
    Qz = _axis_angle_quat('z', roll_z)
    Qx = _axis_angle_quat('x', pitch_x)
    Qy = _axis_angle_quat('y', yaw_y)

    # Compose: R = Rz * Rx * Ry  (apply Ry, then Rx, then Rz)
    q = _qmul(_qmul(Qz, Qx), Qy)
    return np.asarray(_qnorm(q), dtype=np.float32) # (qx, qy, qz, qw)

game = Game()
estimator = Estimate()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

kernel = np.ones((5,5), np.uint8)

cv2.namedWindow("Cam Feed", cv2.WINDOW_NORMAL)
while game.running:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    frame = estimator.get_measurements(frame)
    
    cv2.imshow("Cam Feed", frame)
    cv2.waitKey(1)
    
    game.handle_events()
    game.update()
    game.render()
    game.clock.tick(60)

cap.release()
cv2.destroyAllWindows()