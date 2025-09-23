import cv2
import mediapipe as mp
import numpy as np


class HeadPoseTracker:
    def __init__(self, image_width=640, image_height=480):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.FACE_LANDMARK_INDICES = [1, 33, 263, 61, 291, 199]

        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
            [0.0, -330.0, -65.0]
        ], dtype=np.float64)

        focal_length = max(image_width, image_height)
        self.camera_matrix = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

        self.rotation_threshold = 10
        self.pitch_up_threshold = 4
        self.position_threshold = 15
        self.stability_time = 0

        self.is_stable = False
        self.stable_counter = 0
        self.initial_position = None

        self.kalman_pitch = 0
        self.kalman_yaw = 0
        self.kalman_roll = 0
        self.kalman_gain = 0.25

        self.reset_tracking_state()

    def reset_tracking_state(self):
        self.stability_time = 0
        self.is_stable = False
        self.stable_counter = 0
        self.initial_position = None
        self.kalman_pitch = 0
        self.kalman_yaw = 0
        self.kalman_roll = 0


    def calculate_head_pose(self, landmarks, image_shape):
        image_points = []
        model_points = []

        height, width = image_shape[:2]

        for i, landmark_idx in enumerate(self.FACE_LANDMARK_INDICES):
            landmark = landmarks.landmark[landmark_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            image_points.append([x, y])
            model_points.append(self.face_3d_model[i])

        image_points = np.array(image_points, dtype=np.float64)
        model_points = np.array(model_points, dtype=np.float64)

        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)

                pose_mat = cv2.hconcat([rotation_mat, translation_vec])
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

                pitch = float(euler_angles[0].item()) if hasattr(euler_angles[0], 'item') else float(euler_angles[0])
                yaw = float(euler_angles[1].item()) if hasattr(euler_angles[1], 'item') else float(euler_angles[1])
                roll = float(euler_angles[2].item()) if hasattr(euler_angles[2], 'item') else float(euler_angles[2])


                if pitch > 0: pitch = -(180 - pitch)
                else: pitch = pitch + 180
                yaw = (yaw + 180) % 360 - 180
                roll = (roll + 180) % 360 - 180

                return {
                    'rotation': rotation_vec,
                    'translation': translation_vec,
                    'euler_angles': (pitch, yaw, roll),
                    'image_points': image_points
                }
        except Exception as e:
            print(f"Ошибка расчета позы: {e}")

        return None

    def check_stability(self, pose_data):
        if pose_data is None:
            return {
                'stable': False,
                'pitch_stable': False,
                'yaw_stable': False,
                'roll_stable': False,
                'position_stable': False,
                'angles': (0, 0, 0)
            }

        pitch, yaw, roll = pose_data['euler_angles']

        self.kalman_pitch = self.kalman_pitch + self.kalman_gain * (pitch - self.kalman_pitch)
        self.kalman_yaw = self.kalman_yaw + self.kalman_gain * (yaw - self.kalman_yaw)
        self.kalman_roll = self.kalman_roll + self.kalman_gain * (roll - self.kalman_roll)


        if self.kalman_pitch < 0:
            pitch_stable = abs(self.kalman_pitch) < self.pitch_up_threshold
        else:
            pitch_stable = abs(self.kalman_pitch) < self.rotation_threshold


        yaw_stable = abs(self.kalman_yaw) < self.rotation_threshold
        roll_stable = abs(self.kalman_roll) < self.rotation_threshold

        tx, ty, tz = pose_data['translation'].flatten()

        if self.initial_position is None:
            self.initial_position = [tx, ty, tz]

        relative_tx = tx - self.initial_position[0]
        relative_ty = ty - self.initial_position[1]

        self.initial_position[0] = tx
        self.initial_position[1] = ty

        position_stable = (abs(relative_tx) < self.position_threshold and
                           abs(relative_ty) < self.position_threshold)

        all_stable = pitch_stable and yaw_stable and position_stable and roll_stable

        if all_stable:
            self.is_stable = True
            self.stability_time += 1
        else:
            self.is_stable = False
            self.stability_time = 0

        return {
            'stable': self.is_stable,
            'pitch_stable': pitch_stable,
            'yaw_stable': yaw_stable,
            'roll_stable': roll_stable,
            'position_stable': position_stable,
            'angles': (pitch, yaw, roll)
        }

    def draw_pose_info(self, image, stability_info):
        height, width = image.shape[:2]

        status_text = ""
        if stability_info['stable'] and self.stability_time >= 6:
            circle_color = (0, 255, 0)

        else:
            circle_color = (0, 0, 255)
            status_text = "ADJUST HEAD POSITION"

        circle_center = (50, height - 50)
        circle_radius = 20
        cv2.circle(image, circle_center, circle_radius, circle_color, -1)

        cv2.circle(image, circle_center, circle_radius, (255, 255, 255), 2)

        if not stability_info['stable'] or self.stability_time < 6:
            text_position = (circle_center[0] + 40, circle_center[1] + 5)
            cv2.putText(image, status_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_frame(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            primary_face = results.multi_face_landmarks[0]
            pose_data = self.calculate_head_pose(primary_face, image.shape)
            stability_info = self.check_stability(pose_data)
            self.draw_pose_info(image, stability_info)

        else:
            cv2.putText(image, "NO FACE DETECTED - MOVE INTO FRAME", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return image


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return

    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить первый кадр")
        cap.release()
        return

    h, w = frame.shape[:2]

    tracker = HeadPoseTracker(image_width=w, image_height=h)

    print("Запуск системы трекинга головы...")
    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = tracker.process_frame(frame)

        cv2.imshow('Head Pose Tracker', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()