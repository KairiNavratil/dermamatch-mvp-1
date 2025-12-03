import mediapipe as mp
import cv2
import numpy as np

class FaceMapper:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.landmark_map = {}
        self._initialize_full_mesh_map()

    def _initialize_full_mesh_map(self):
        """
        Maps ALL 468 MediaPipe landmarks to semantic regions.
        We use broader categories to avoid mislabeling edge cases.
        """
        # 1. LIPS (Inner & Outer)
        lips = set(range(0, 100)).intersection({0, 13, 14, 17, 37, 39, 40, 61, 80, 81, 82, 84, 87, 88, 91, 95})
        lips.update({146, 178, 181, 185, 191, 267, 269, 270, 291, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415})
        self._map_indices(list(lips), "Lips")

        # 2. LEFT EYE (Subject's Left)
        # Includes lids, corners, and immediate perimeter
        left_eye = {263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398, 469, 470, 471, 472}
        self._map_indices(list(left_eye), "Left Eye Area")

        # 3. RIGHT EYE (Subject's Right)
        right_eye = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 474, 475, 476, 477}
        self._map_indices(list(right_eye), "Right Eye Area")

        # 4. LEFT EYEBROW
        left_brow = {276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 298, 301, 299, 284}
        self._map_indices(list(left_brow), "Left Eyebrow")

        # 5. RIGHT EYEBROW
        right_brow = {46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 68, 71, 69, 54}
        self._map_indices(list(right_brow), "Right Eyebrow")

        # 6. NOSE (Entire Structure)
        # Tip, Bridge, Nostrils, and the immediate slopes
        nose = {1, 2, 4, 5, 6, 168, 195, 197, 279, 49, 278, 48, 218, 219, 220, 275, 438, 439, 440, 294, 19, 94, 274, 456, 281}
        self._map_indices(list(nose), "Nose Region")

        # 7. FOREHEAD
        # Central and side forehead, excluding temples which are separate
        forehead = {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 8, 9, 107, 66, 69, 109, 103, 67, 336, 296, 299, 338, 332, 297, 337, 336, 296, 293, 334, 295, 151, 108, 104, 68, 71}
        self._map_indices(list(forehead), "Forehead")

        # 8. CHIN
        chin = {152, 199, 175, 200, 18, 83, 313, 18, 200, 175, 152}
        self._map_indices(list(chin), "Chin")

        # 9. CHEEKS (The Rest)
        # We assume anything not defined above is likely cheek/skin/jaw.
        # We split roughly by index ranges or coordinate logic if needed, 
        # but for now, we rely on the specific lists below.
        
        # Right Cheek Specifics
        right_cheek = {50, 205, 203, 142, 123, 147, 213, 192, 214, 212, 138, 135, 216, 206, 207, 211, 210, 209, 126, 47, 121, 120, 111, 36, 101, 227, 137, 177, 215, 138, 135, 169, 170, 171, 140, 141, 142}
        self._map_indices(list(right_cheek), "Right Cheek Region")

        # Left Cheek Specifics
        left_cheek = {280, 425, 423, 371, 352, 376, 433, 416, 434, 432, 367, 364, 436, 426, 427, 431, 430, 429, 355, 277, 350, 349, 340, 266, 330, 447, 366, 401, 435, 367, 364, 394, 395, 396, 369, 370, 371}
        self._map_indices(list(left_cheek), "Left Cheek Region")
        
        # 10. JAWLINE
        jaw = {172, 136, 150, 149, 176, 397, 365, 379, 378, 400, 127, 234, 93, 132, 58, 356, 454, 323, 361, 288}
        self._map_indices(list(jaw), "Jawline")

    def _map_indices(self, indices, description):
        for idx in indices:
            self.landmark_map[idx] = description

    def get_region_for_index(self, idx):
        # Fallback for the ~100 points not explicitly listed (mid-face fillers)
        return self.landmark_map.get(idx, "Face Mesh (General)")

    def get_face_landmarks(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None
        height, width, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None
        landmarks = results.multi_face_landmarks[0].landmark
        
        mapped_coordinates = {}
        for idx, lm in enumerate(landmarks):
            region = self.get_region_for_index(idx)
            key = f"id_{idx}"
            mapped_coordinates[key] = {
                "x": int(lm.x * width),
                "y": int(lm.y * height),
                "region": region,
                "index": idx
            }
        return {"width": width, "height": height, "landmarks": mapped_coordinates}