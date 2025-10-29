import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GSBFConfig:
    white_threshold: int = 240
    border_thickness: int = 0
    border_restore_size: int = 100
    border_restore_color: Tuple[int, int, int] = (255, 255, 255)
    num_detectors: int = 6
    kernel_size: Tuple[int, int] = (3, 3)
    min_contour_thickness: int = 5
    min_contour_area: int = 0
    max_contour_area: int = 100000


class GSBFModule:
    """Gradient Slice Boundary Fusion (GSBF) Module"""
    
    def __init__(self, config: Optional[GSBFConfig] = None, device: str = "cpu"):
        """Initialize GSBF module with configuration and device"""
        self.config = config or GSBFConfig()
        self.device = device
        self._check_device_compatibility()
        
        self.kernel1 = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.kernel_size
        )

    def _check_device_compatibility(self) -> None:
        """Check device availability and switch if necessary"""
        if self.device == "cuda" and not cv2.cuda.getCudaEnabledDeviceCount():
            print("WARNING: CUDA not available, switching to CPU")
            self.device = "cpu"
        print(f"Using device: {self.device}")

    def _to_device(self, img: np.ndarray) -> np.ndarray:
        """Move image to target processing device"""
        if self.device == "cuda":
            return cv2.cuda_GpuMat(img).download()
        return img

    def _from_device(self, img: np.ndarray) -> np.ndarray:
        """Move image back from device to CPU"""
        return img

    def _filter_thin_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter thin contours and those with invalid area"""
        filtered_contours = []
        
        def is_thin(contour: np.ndarray) -> bool:
            _, _, w, h = cv2.boundingRect(contour)
            return w < self.config.min_contour_thickness and h < self.config.min_contour_thickness
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_contour_area < area < self.config.max_contour_area:
                if is_thin(cnt):
                    x, y, w, h = cv2.boundingRect(cnt)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    sub_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filtered_contours.extend(sub_contours)
                else:
                    filtered_contours.append(cnt)
            else:
                filtered_contours.append(cnt)
        return filtered_contours

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image: border cropping and morphological operations"""
        h, w, _ = img.shape
        crop_h, crop_w = h - 200, w - 200
        cropped = img[100:100 + crop_h, 100:100 + crop_w]
        
        dilated = cv2.dilate(cropped, self.kernel1, iterations=1)
        return cv2.erode(dilated, self.kernel1, iterations=1)

    def _postprocess(self, processed_img: np.ndarray) -> np.ndarray:
        """Postprocess result: convert black background to white"""
        h_res, w_res = processed_img.shape[:2]
        
        # Convert black background (0,0,0) to white (255,255,255)
        black_mask = np.all(processed_img == [0, 0, 0], axis=-1)
        processed_img[black_mask] = [255, 255, 255]
        
        # Restore borders
        restored = np.full(
            (h_res + 2 * self.config.border_restore_size,
             w_res + 2 * self.config.border_restore_size, 3),
            self.config.border_restore_color,
            dtype=np.uint8
        )
        restored[
            self.config.border_restore_size : self.config.border_restore_size + h_res,
            self.config.border_restore_size : self.config.border_restore_size + w_res
        ] = processed_img
        return restored

    def forward(self, img: np.ndarray) -> np.ndarray:
        """Core processing logic (gradient slice boundary fusion)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        masks = []
        masked_images = []
        
        for i in range(self.config.num_detectors):
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            iter_count = (i // 2) + 2
            blurred = cv2.dilate(blurred, self.kernel1, iterations=iter_count)
            blurred = cv2.erode(blurred, self.kernel1, iterations=iter_count)
            
            edges = cv2.Canny(blurred, 60 + 5 * i, 90 + 7 * i)
            edges_dilated = cv2.dilate(edges, self.kernel1, iterations=1)
            
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = self._filter_thin_contours(contours)
            
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
            masks.append(mask)
            
            masked = cv2.bitwise_and(img, img, mask=mask)
            masked_images.append(masked)
        
        result = np.zeros_like(img)
        for mask, masked in zip(masks, masked_images):
            result[mask > 0] = masked[mask > 0]
        
        return result

    def process(self, img_path: str) -> Optional[np.ndarray]:
        """Complete processing pipeline for single image"""
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        preprocessed = self._preprocess(img)
        preprocessed = self._to_device(preprocessed)
        
        core_result = self.forward(preprocessed)
        
        final_result = self._postprocess(core_result)
        final_result = self._from_device(final_result)
        
        return final_result

    def process_batch(self, img_paths: List[str]) -> List[np.ndarray]:
        """Batch process multiple images"""
        results = []
        for path in img_paths:
            processed = self.process(path)
            if processed is not None:
                results.append(processed)
        return results

    def run(self, input_dir: str, output_dir: str) -> None:
        """Run complete processing pipeline for directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        img_extensions = ('.jpg', '.jpeg', '.png')
        img_paths = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith(img_extensions)
        ]
        
        print(f"Found {len(img_paths)} images for processing...")
        for i, path in enumerate(img_paths):
            filename = os.path.basename(path)
            print(f"Processing [{i+1}/{len(img_paths)}]: {filename}")
            
            output_img = self.process(path)
            if output_img is None:
                print(f"Warning: Skipping {filename} (read failure)")
                continue
            
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, output_img)
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    config = GSBFConfig(
        num_detectors=6,
        border_restore_size=100
    )
    
    gsbf = GSBFModule(config=config, device="cpu")
    
    input_directory = r'F:\DCIM\DJI_202510261502_002\DJI_20251026150647_0002_S_extracted_frames'
    output_directory = r'F:\DCIM\DJI_202510261502_002\DJI_20251026150647_0002_S_extracted_frames____'
    gsbf.run(input_directory, output_directory)