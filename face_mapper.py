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

        # PRE-DEFINED SEMANTIC MAP
        # Maps specific landmark indices to granular text descriptions.
        self.landmark_map = {}
        self._initialize_landmark_map()

    def _initialize_landmark_map(self):
        """
        Populates the landmark_map with specific regions.
        Source: Standard MediaPipe Face Mesh Topology.
        """
        # --- NOSE ---
        self._map_indices([1, 4, 5, 195, 197], "Nose Tip")
        self._map_indices([6, 168, 8, 351, 417], "Nose Bridge - Upper") # Glabella area
        self._map_indices([122, 196, 174, 198, 49, 131], "Nose - Right Nostril/Base")
        self._map_indices([351, 419, 456, 279, 360], "Nose - Left Nostril/Base")
        self._map_indices([218, 219, 220, 237], "Nose - Right Wing")
        self._map_indices([438, 439, 440, 457], "Nose - Left Wing")

        # --- LIPS ---
        self._map_indices([0, 267, 269, 37, 39, 40, 185], "Upper Lip - Center")
        self._map_indices([61, 146, 91, 181, 84], "Upper Lip - Right Corner")
        self._map_indices([291, 375, 321, 405, 314], "Upper Lip - Left Corner")
        self._map_indices([17, 18, 200, 314, 402, 83], "Lower Lip - Center")
        self._map_indices([78, 95, 88, 178, 87, 14], "Lower Lip - Right Side")
        self._map_indices([308, 324, 318, 402, 317, 14], "Lower Lip - Left Side")
        self._map_indices([164, 0, 267], "Philtrum (Above Lip)")

        # --- CHIN ---
        self._map_indices([152, 199, 175], "Chin - Center Tip")
        self._map_indices([377, 378, 379, 365], "Chin - Left Side")
        self._map_indices([148, 149, 150, 136], "Chin - Right Side")
        self._map_indices([200, 201, 194, 32, 262], "Chin - Upper Crease")

        # --- EYES (RIGHT - Subject's Right) ---
        self._map_indices([33, 246, 161, 160, 159, 158, 157, 173], "Right Eye - Upper Lid")
        self._map_indices([163, 144, 145, 153, 154, 155, 133], "Right Eye - Lower Lid")
        self._map_indices([133, 173, 155, 154], "Right Eye - Inner Corner")
        self._map_indices([33, 7, 163, 144], "Right Eye - Outer Corner")
        # Right Under Eye (Orbital) - CRITICAL AREA
        self._map_indices([116, 117, 118, 119, 100, 101], "Right Under Eye - Upper Orbital")
        self._map_indices([230, 229, 228, 226, 31], "Right Under Eye - Lower Orbital")
        self._map_indices([35, 124, 46, 53, 52, 65, 55], "Right Eyebrow")

        # --- EYES (LEFT - Subject's Left) ---
        self._map_indices([263, 466, 388, 387, 386, 385, 384, 398], "Left Eye - Upper Lid")
        self._map_indices([362, 382, 381, 380, 374, 373, 390, 249], "Left Eye - Lower Lid")
        self._map_indices([362, 398, 384, 385], "Left Eye - Inner Corner")
        self._map_indices([263, 249, 390, 373], "Left Eye - Outer Corner")
        # Left Under Eye (Orbital)
        self._map_indices([345, 346, 347, 348, 329, 330], "Left Under Eye - Upper Orbital")
        self._map_indices([450, 449, 448, 446, 261], "Left Under Eye - Lower Orbital")
        self._map_indices([265, 353, 276, 283, 282, 295, 285], "Left Eyebrow")

        # --- CHEEKS (RIGHT) ---
        self._map_indices([50, 205, 203, 206, 207], "Right Cheek - Apple (Center)")
        self._map_indices([123, 147, 192, 213, 216], "Right Cheek - Inner (Near Nose)")
        self._map_indices([227, 34, 111, 116], "Right Cheek - Upper (Bone)")
        self._map_indices([127, 234, 93, 132], "Right Cheek - Outer (Near Ear)")
        self._map_indices([210, 214, 212, 57], "Right Cheek - Lower (Near Jaw)")

        # --- CHEEKS (LEFT) ---
        self._map_indices([280, 425, 423, 426, 427], "Left Cheek - Apple (Center)")
        self._map_indices([352, 376, 416, 433, 436], "Left Cheek - Inner (Near Nose)")
        self._map_indices([447, 264, 340, 345], "Left Cheek - Upper (Bone)")
        self._map_indices([356, 454, 323, 361], "Left Cheek - Outer (Near Ear)")
        self._map_indices([430, 434, 432, 287], "Left Cheek - Lower (Near Jaw)")

        # --- FOREHEAD ---
        self._map_indices([10, 151, 9], "Forehead - Center Low")
        self._map_indices([109, 67, 103], "Forehead - Center High")
        self._map_indices([338, 337, 299, 296], "Forehead - Left Low")
        self._map_indices([297, 332, 284], "Forehead - Left High")
        self._map_indices([109, 108, 69, 66], "Forehead - Right Low")
        self._map_indices([67, 103, 54], "Forehead - Right High")
        self._map_indices([68, 104, 63, 105], "Forehead - Right Temple")
        self._map_indices([298, 333, 293, 334], "Forehead - Left Temple")
        
        # --- JAWLINE ---
        self._map_indices([172, 136, 150, 149, 176], "Jawline - Right")
        self._map_indices([397, 365, 379, 378, 400], "Jawline - Left")
        self._map_indices([127, 234, 93, 132, 58], "Jaw Angle - Right")
        self._map_indices([356, 454, 323, 361, 288], "Jaw Angle - Left")

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
            # We treat every point as a potential anchor now
            key = f"id_{idx}"
            mapped_coordinates[key] = {
                "x": int(lm.x * width),
                "y": int(lm.y * height),
                "region": region,
                "index": idx
            }
        return {"width": width, "height": height, "landmarks": mapped_coordinates}