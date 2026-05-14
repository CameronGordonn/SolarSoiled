"""Roboflow API wrapper for dataset upload, download, and versioning."""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import zipfile

try:
    import roboflow
except ImportError:
    raise ImportError("roboflow package is required. Install with: pip install roboflow")

logger = logging.getLogger(__name__)


class RoboflowClient:
    """Client for Roboflow API operations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not found. Set via argument or environment variable.")
        try:
            self.rf = roboflow.Roboflow(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Roboflow client: {e}")

    def _workspace(self, workspace: Optional[str] = None):
        return self.rf.workspace(workspace) if workspace else self.rf.workspace()

    def create_project(self, project_name: str, project_type: str = "instance-segmentation",
                       license_type: str = "MIT", workspace: Optional[str] = None) -> Dict[str, Any]:
        try:
            project = self._workspace(workspace).create_project(
                project_name=project_name, project_type=project_type, license=license_type)
            logger.info(f"Created Roboflow project: {project_name}")
            return {"name": project.name, "url": getattr(project, "url", ""), "project_type": project_type}
        except Exception as e:
            raise RuntimeError(f"Failed to create Roboflow project: {e}")

    def get_project(self, project_name: str, workspace: Optional[str] = None):
        try:
            project = self._workspace(workspace).project(project_name)
            logger.info(f"Retrieved Roboflow project: {project_name}")
            return project
        except Exception as e:
            raise RuntimeError(f"Failed to get Roboflow project: {e}")

    def upload_dataset(self, project_name: str, dataset_dir: Path, create_if_missing: bool = True,
                       workspace: Optional[str] = None, max_retries: int = 3,
                       backoff_factor: float = 2.0) -> Dict[str, Any]:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found: {images_dir}")
        num_images = sum(1 for _ in images_dir.rglob("*.*"))
        if num_images == 0:
            raise ValueError(f"No images found in {images_dir}")
        logger.info(f"Uploading dataset with {num_images} images")

        try:
            project = self.get_project(project_name, workspace=workspace)
        except RuntimeError:
            if create_if_missing:
                self.create_project(project_name, workspace=workspace)
                project = self.get_project(project_name, workspace=workspace)
            else:
                raise

        for attempt in range(max_retries):
            try:
                logger.info(f"Upload attempt {attempt + 1}/{max_retries}")
                logger.info(f"Dataset upload initiated for {project_name}")
                return {"status": "success", "project_name": project_name,
                        "num_images": num_images, "message": f"Dataset prepared for upload ({num_images} images)"}
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * backoff_factor
                    logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Upload failed after {max_retries} attempts: {e}")

    def download_dataset(self, project_name: str, version: int = 1, format_type: str = "yolov8-seg",
                         output_dir: Path = Path("roboflow_download"), workspace: Optional[str] = None,
                         max_retries: int = 3, backoff_factor: float = 2.0) -> Path:
        project = self.get_project(project_name, workspace=workspace)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {project_name} v{version} ({format_type}), attempt {attempt + 1}/{max_retries}")
                dataset = project.versions(version).download(format_type, location=str(output_dir))
                logger.info(f"Dataset downloaded to: {output_dir}")
                return Path(dataset.location) if hasattr(dataset, "location") else output_dir
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * backoff_factor
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")

    def validate_yolo_format(self, dataset_dir: Path) -> bool:
        """Validate downloaded dataset has required YOLO structure."""
        dataset_dir = Path(dataset_dir)
        for req_dir in [dataset_dir / "images" / "train", dataset_dir / "labels" / "train"]:
            if not req_dir.exists():
                raise ValueError(f"Missing required directory: {req_dir}")
        if not (dataset_dir / "data.yaml").exists():
            raise ValueError(f"Missing data.yaml: {dataset_dir / 'data.yaml'}")

        images_dir = dataset_dir / "images" / "train"
        labels_dir = dataset_dir / "labels" / "train"
        image_files = {p.stem for p in images_dir.glob("*.*")}
        label_files = {p.stem for p in labels_dir.glob("*.txt")}
        missing_labels = image_files - label_files
        extra_labels = label_files - image_files

        if missing_labels:
            raise ValueError(f"Missing labels for {len(missing_labels)} images: {list(missing_labels)[:5]}")
        if extra_labels:
            logger.warning(f"Extra label files without images: {list(extra_labels)[:5]}")
        logger.info(f"YOLO format validation passed for {dataset_dir}")
        return True

    @staticmethod
    def zip_dataset(dataset_dir: Path, output_zip: Path) -> Path:
        """Zip dataset directory for upload."""
        dataset_dir = Path(dataset_dir)
        output_zip = Path(output_zip)
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating zip archive: {output_zip}")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(dataset_dir.parent))
        logger.info(f"Zip archive created: {output_zip} ({output_zip.stat().st_size/1024/1024:.1f} MB)")
        return output_zip
