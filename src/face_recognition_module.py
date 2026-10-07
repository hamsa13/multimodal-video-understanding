"""
Face Detection & Recognition Module
Detects faces and identifies people in video frames
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Face bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class FaceDetection:
    """A detected face"""
    bbox: BoundingBox
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class FaceMatch:
    """A face matched to known identity"""
    detection: FaceDetection
    identity: str
    distance: float
    confidence: float


@dataclass
class FrameFaces:
    """All faces detected in a frame"""
    frame_number: int
    timestamp: float
    detections: List[FaceDetection]
    matches: List[FaceMatch] = field(default_factory=list)


class FaceProcessor:
    """
    Face detection and recognition pipeline
    Uses multiple backends for detection and recognition
    """
    
    def __init__(self, config):
        self.config = config
        self.detector = None
        self.recognizer = None
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self.face_db = None
        
        self._init_detector()
        self._init_recognizer()
        self._load_known_faces()
    
    def _init_detector(self) -> None:
        """Initialize face detector"""
        model = self.config.face.detection_model.lower()
        
        if model == "mtcnn":
            self._init_mtcnn()
        elif model == "retinaface":
            self._init_retinaface()
        elif model == "dlib":
            self._init_dlib()
        else:
            logger.warning(f"Unknown detector {model}, using MTCNN")
            self._init_mtcnn()
    
    def _init_mtcnn(self) -> None:
        """Initialize MTCNN detector"""
        try:
            from mtcnn import MTCNN
            # MTCNN API differs between versions; try with min_face_size, else fallback
            try:
                self.detector = MTCNN(
                    min_face_size=self.config.face.min_face_size
                )
            except TypeError:
                # Older/newer MTCNN versions may not accept min_face_size kwarg
                self.detector = MTCNN()
            self._detector_type = "mtcnn"
            logger.info("MTCNN face detector initialized")
        except ImportError:
            logger.error("MTCNN not installed. Run: pip install mtcnn")
            self.detector = None
            self._detector_type = None
        except Exception as e:
            logger.warning(f"Failed to initialize MTCNN detector: {e}")
            # Disable detector to avoid aborting the pipeline
            self.detector = None
            self._detector_type = None
    
    def _init_retinaface(self) -> None:
        """Initialize RetinaFace detector"""
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            self._detector_type = "retinaface"
            logger.info("RetinaFace detector initialized")
        except ImportError:
            logger.warning("RetinaFace not available, falling back to MTCNN")
            self._init_mtcnn()
    
    def _init_dlib(self) -> None:
        """Initialize dlib detector"""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self._detector_type = "dlib"
            logger.info("dlib face detector initialized")
        except ImportError:
            logger.warning("dlib not available, falling back to MTCNN")
            self._init_mtcnn()
    
    def _init_recognizer(self) -> None:
        """Initialize face recognizer for embeddings"""
        # DeepFace can trigger runtime issues (Conv2D errors) on some setups.
        # Disable DeepFace-based recognition by default to keep pipeline robust.
        try:
            import deepface  # type: ignore
            logger.warning("DeepFace is available but will be disabled to avoid runtime issues in this environment.")
        except Exception:
            logger.info("DeepFace not available; face recognition disabled.")

        self.recognizer = None
    
    def _load_known_faces(self) -> None:
        """Load known faces from directory"""
        known_dir = Path(self.config.face.known_faces_dir)
        if not known_dir.exists():
            known_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created known faces directory: {known_dir}")
            return
        
        for person_dir in known_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                embeddings = []
                
                for img_path in person_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            embedding = self._get_embedding(str(img_path))
                            if embedding is not None:
                                embeddings.append(embedding)
                        except Exception as e:
                            logger.warning(f"Failed to process {img_path}: {e}")
                
                if embeddings:
                    self.known_faces[person_name] = embeddings
                    logger.info(f"Loaded {len(embeddings)} faces for {person_name}")
        
        logger.info(f"Loaded {len(self.known_faces)} known people")
    
    def detect_faces(
        self,
        image: Union[str, np.ndarray],
        return_landmarks: bool = True
    ) -> List[FaceDetection]:
        """
        Detect faces in an image
        
        Args:
            image: Image path or numpy array (BGR format)
            return_landmarks: Whether to detect facial landmarks
        
        Returns:
            List of FaceDetection objects
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return []
        
        # Convert to RGB for face detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = []
        
        try:
            if self._detector_type == "mtcnn" and self.detector is not None:
                detections = self._detect_mtcnn(rgb_image, return_landmarks)
            elif self._detector_type == "retinaface" and self.detector is not None:
                detections = self._detect_retinaface(rgb_image, return_landmarks)
            elif self._detector_type == "dlib" and self.detector is not None:
                detections = self._detect_dlib(rgb_image, return_landmarks)
            else:
                detections = []
        except Exception as e:
            logger.warning(f"Face detector error: {e}")
            detections = []
        
        # Filter by confidence
        threshold = self.config.face.confidence_threshold
        detections = [d for d in detections if d.confidence >= threshold]
        
        return detections
    
    def _detect_mtcnn(
        self,
        image: np.ndarray,
        return_landmarks: bool
    ) -> List[FaceDetection]:
        """Detect faces using MTCNN"""
        results = self.detector.detect_faces(image)
        
        detections = []
        for r in results:
            bbox = BoundingBox(
                x1=r['box'][0],
                y1=r['box'][1],
                x2=r['box'][0] + r['box'][2],
                y2=r['box'][1] + r['box'][3]
            )
            
            landmarks = None
            if return_landmarks and 'keypoints' in r:
                landmarks = r['keypoints']
            
            detection = FaceDetection(
                bbox=bbox,
                confidence=r['confidence'],
                landmarks=landmarks
            )
            detections.append(detection)
        
        return detections
    
    def _detect_retinaface(
        self,
        image: np.ndarray,
        return_landmarks: bool
    ) -> List[FaceDetection]:
        """Detect faces using RetinaFace"""
        results = self.detector.detect_faces(image)
        
        detections = []
        for key, face in results.items():
            bbox = BoundingBox(
                x1=int(face['facial_area'][0]),
                y1=int(face['facial_area'][1]),
                x2=int(face['facial_area'][2]),
                y2=int(face['facial_area'][3])
            )
            
            landmarks = None
            if return_landmarks and 'landmarks' in face:
                landmarks = face['landmarks']
            
            detection = FaceDetection(
                bbox=bbox,
                confidence=face.get('score', 1.0),
                landmarks=landmarks
            )
            detections.append(detection)
        
        return detections
    
    def _detect_dlib(
        self,
        image: np.ndarray,
        return_landmarks: bool
    ) -> List[FaceDetection]:
        """Detect faces using dlib"""
        import dlib
        
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        
        detections = []
        for face in faces:
            bbox = BoundingBox(
                x1=face.left(),
                y1=face.top(),
                x2=face.right(),
                y2=face.bottom()
            )
            
            detection = FaceDetection(
                bbox=bbox,
                confidence=1.0,  # dlib doesn't provide confidence
                landmarks=None
            )
            detections.append(detection)
        
        return detections
    
    def _get_embedding(
        self,
        image: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Get face embedding from image"""
        if self.recognizer is None:
            return None
        
        try:
            result = self.recognizer.represent(
                image,
                model_name=self._recognizer_model,
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
        
        return None
    
    def recognize_face(
        self,
        image: Union[str, np.ndarray],
        detection: FaceDetection
    ) -> Optional[FaceMatch]:
        """
        Try to identify a detected face
        
        Args:
            image: Original image
            detection: Face detection to identify
        
        Returns:
            FaceMatch if identified, None otherwise
        """
        if not self.known_faces:
            return None
        
        if isinstance(image, str):
            image = cv2.imread(image)

        # Crop face region with safety checks
        bbox = detection.bbox
        h, w = image.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(w - 1, bbox.x1))
        y1 = max(0, min(h - 1, bbox.y1))
        x2 = max(0, min(w, bbox.x2))
        y2 = max(0, min(h, bbox.y2))

        # Ensure non-empty box
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
            logger.debug(f"Face bbox too small or invalid: {(x1,y1,x2,y2)}; skipping recognition")
            return None

        face_img = image[y1:y2, x1:x2]

        # Get embedding (guarded inside _get_embedding)
        try:
            embedding = self._get_embedding(face_img)
        except Exception as e:
            logger.warning(f"Unexpected error extracting embedding: {e}")
            return None

        if embedding is None:
            return None

        detection.embedding = embedding

        # Compare with known faces
        best_match = None
        min_distance = float('inf')

        try:
            for name, known_embeddings in self.known_faces.items():
                for known_emb in known_embeddings:
                    distance = self._compute_distance(embedding, known_emb)

                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
        except Exception as e:
            logger.warning(f"Error comparing embeddings: {e}")
            return None

        # Check threshold
        if min_distance > self.config.face.threshold:
            return None

        confidence = 1.0 - min_distance

        return FaceMatch(
            detection=detection,
            identity=best_match,
            distance=min_distance,
            confidence=confidence
        )
    
    def _compute_distance(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Compute distance between embeddings"""
        metric = self.config.face.distance_metric
        
        if metric == "cosine":
            # Cosine distance
            dot = np.dot(emb1, emb2)
            norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            return 1 - (dot / norm) if norm > 0 else 1.0
        else:
            # Euclidean distance
            return np.linalg.norm(emb1 - emb2)
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        recognize: bool = True
    ) -> FrameFaces:
        """
        Process a single frame for faces
        
        Args:
            frame: Frame image (BGR)
            frame_number: Frame index
            timestamp: Frame timestamp in seconds
            recognize: Whether to attempt face recognition
        
        Returns:
            FrameFaces with all detected/recognized faces
        """
        detections = self.detect_faces(frame)
        matches = []
        
        if recognize and self.known_faces:
            for det in detections:
                match = self.recognize_face(frame, det)
                if match:
                    matches.append(match)
        
        return FrameFaces(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            matches=matches
        )
    
    def process_video_frames(
        self,
        frames: List[Dict[str, Any]],
        recognize: bool = True
    ) -> List[FrameFaces]:
        """
        Process multiple frames
        
        Args:
            frames: List of dicts with 'frame_number', 'timestamp', 'image'
            recognize: Whether to attempt face recognition
        
        Returns:
            List of FrameFaces results
        """
        from tqdm import tqdm
        
        results = []
        
        for frame_data in tqdm(frames, desc="Processing faces"):
            try:
                result = self.process_frame(
                    frame=frame_data['image'],
                    frame_number=frame_data['frame_number'],
                    timestamp=frame_data['timestamp'],
                    recognize=recognize
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Face processing error on frame {frame_data.get('frame_number')}: {e}")
                # Continue without faces for this frame
                results.append(FrameFaces(
                    frame_number=frame_data.get('frame_number', -1),
                    timestamp=frame_data.get('timestamp', 0.0),
                    detections=[],
                    matches=[]
                ))
        
        return results
    
    def cluster_unknown_faces(
        self,
        frame_faces: List[FrameFaces],
        min_samples: int = 3
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Cluster unknown faces to identify unique people
        
        Args:
            frame_faces: List of FrameFaces from video
            min_samples: Minimum samples for a cluster
        
        Returns:
            Dict mapping cluster ID to list of (frame_number, detection_index)
        """
        from sklearn.cluster import DBSCAN
        
        # Collect all embeddings
        embeddings = []
        locations = []
        
        for ff in frame_faces:
            for i, det in enumerate(ff.detections):
                if det.embedding is not None:
                    embeddings.append(det.embedding)
                    locations.append((ff.frame_number, i))
        
        if len(embeddings) < min_samples:
            return {}
        
        embeddings = np.array(embeddings)
        
        # Cluster
        clustering = DBSCAN(eps=0.5, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for label, loc in zip(labels, locations):
            if label >= 0:  # Ignore noise (-1)
                clusters[f"Person_{label}"].append(loc)
        
        return dict(clusters)
    
    def add_known_face(
        self,
        name: str,
        image: Union[str, np.ndarray]
    ) -> bool:
        """
        Add a new known face
        
        Args:
            name: Person's name
            image: Face image path or array
        
        Returns:
            True if successful
        """
        embedding = self._get_embedding(image)
        if embedding is None:
            return False
        
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(embedding)
        
        # Optionally save to disk
        if isinstance(image, np.ndarray):
            save_dir = Path(self.config.face.known_faces_dir) / name
            save_dir.mkdir(parents=True, exist_ok=True)
            count = len(list(save_dir.glob("*.jpg")))
            save_path = save_dir / f"{count + 1}.jpg"
            cv2.imwrite(str(save_path), image)
        
        logger.info(f"Added face for {name}")
        return True
    
    def draw_faces(
        self,
        image: np.ndarray,
        frame_faces: FrameFaces,
        draw_landmarks: bool = False
    ) -> np.ndarray:
        """Draw detected faces on image"""
        result = image.copy()
        
        for det in frame_faces.detections:
            # Draw bounding box
            cv2.rectangle(
                result,
                (det.bbox.x1, det.bbox.y1),
                (det.bbox.x2, det.bbox.y2),
                (0, 255, 0), 2
            )
            
            # Draw landmarks
            if draw_landmarks and det.landmarks:
                for name, point in det.landmarks.items():
                    cv2.circle(result, point, 2, (0, 0, 255), -1)
        
        # Draw recognized faces with names
        for match in frame_faces.matches:
            bbox = match.detection.bbox
            label = f"{match.identity} ({match.confidence:.2f})"
            
            cv2.rectangle(
                result,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (255, 0, 0), 2
            )
            
            cv2.putText(
                result, label,
                (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2
            )
        
        return result
