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

    def get_region_for_index(self, idx):
        """
        Maps a MediaPipe landmark index (0-467) to a semantic region.
        Based on standard MediaPipe Face Mesh topology.
        """
        # Specific Keypoints (High Priority)
        if idx == 1: return "Nose Tip"
        if idx == 152: return "Chin Center"
        if idx == 10: return "Forehead Center"
        if idx == 61: return "Mouth Corner Right"
        if idx == 291: return "Mouth Corner Left"

        # Ranges/Sets for broad regions
        # Note: These sets are approximations of the mesh topology
        
        # Lips
        if idx in {0, 13, 14, 17, 37, 39, 40, 61, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415}:
            return "Lips"
            
        # Left Eye (Subject's Left)
        if idx in {263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398, 469, 470, 471, 472}:
            return "Left Eye Area"
            
        # Right Eye (Subject's Right)
        if idx in {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 474, 475, 476, 477}:
            return "Right Eye Area"
            
        # Left Eyebrow
        if idx in {276, 282, 283, 285, 293, 295, 296, 300, 334, 336}:
            return "Left Eyebrow"
            
        # Right Eyebrow
        if idx in {46, 52, 53, 55, 63, 65, 66, 70, 105, 107}:
            return "Right Eyebrow"
            
        # Face Oval / Silhouette
        if idx in {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109}:
            return "Face Outline"

        # General Regional Estimation (if not in specific sets above)
        # We use a rough coordinate logic or known index ranges for filler points
        
        # Nose Area (Central indices)
        if idx in {1, 2, 4, 5, 6, 168, 195, 197, 279, 49, 278, 48}:
            return "Nose"
            
        # Forehead (Upper indices not in outline)
        if idx in {8, 9, 107, 66, 69, 109, 103, 67, 336, 296, 299, 338, 332, 297}:
            return "Forehead"

        # If we can't categorize strictly, we return "Face Mesh"
        # The AI will primarily use the coordinate proximity anyway.
        return "General Face Mesh"

    def get_face_landmarks(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None: return None

        height, width, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks: return None

        landmarks = results.multi_face_landmarks[0].landmark
        
        mapped_coordinates = {}
        
        # Iterate ALL 468 landmarks
        for idx, lm in enumerate(landmarks):
            region = self.get_region_for_index(idx)
            
            # Key Format: "id_{index}" (e.g., "id_0", "id_467")
            # We add the region description into the value object later for the AI
            key = f"id_{idx}"
            
            pixel_x = int(lm.x * width)
            pixel_y = int(lm.y * height)
            
            # Store data
            mapped_coordinates[key] = {
                "x": pixel_x,
                "y": pixel_y,
                "region": region,
                "index": idx
            }

        return {
            "width": width,
            "height": height,
            "landmarks": mapped_coordinates
        }