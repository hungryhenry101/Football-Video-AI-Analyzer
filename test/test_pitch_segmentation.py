"""
Test script for soccer pitch segmentation model.

This script tests the soccer_pitch_segmentation.pth model on input videos
and visualizes the segmentation results.
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import matplotlib.pyplot as plt


class SoccerPitch:
    """Static class variables that are specified by the rules of the game """
    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    lines_classes = [
        'Big rect. left bottom',
        'Big rect. left main',
        'Big rect. left top',
        'Big rect. right bottom',
        'Big rect. right main',
        'Big rect. right top',
        'Circle central',
        'Circle left',
        'Circle right',
        'Goal left crossbar',
        'Goal left post left ',
        'Goal left post right',
        'Goal right crossbar',
        'Goal right post left',
        'Goal right post right',
        'Goal unknown',
        'Line unknown',
        'Middle line',
        'Side line bottom',
        'Side line left',
        'Side line right',
        'Side line top',
        'Small rect. left bottom',
        'Small rect. left main',
        'Small rect. left top',
        'Small rect. right bottom',
        'Small rect. right main',
        'Small rect. right top'
    ]

    # RGB values
    palette = {
        'Big rect. left bottom': (127, 0, 0),
        'Big rect. left main': (102, 102, 102),
        'Big rect. left top': (0, 0, 127),
        'Big rect. right bottom': (86, 32, 39),
        'Big rect. right main': (48, 77, 0),
        'Big rect. right top': (14, 97, 100),
        'Circle central': (0, 0, 255),
        'Circle left': (255, 127, 0),
        'Circle right': (0, 255, 255),
        'Goal left crossbar': (255, 255, 200),
        'Goal left post left ': (165, 255, 0),
        'Goal left post right': (155, 119, 45),
        'Goal right crossbar': (86, 32, 139),
        'Goal right post left': (196, 120, 153),
        'Goal right post right': (166, 36, 52),
        'Goal unknown': (0, 0, 0),
        'Line unknown': (0, 0, 0),
        'Middle line': (255, 255, 0),
        'Side line bottom': (255, 0, 255),
        'Side line left': (0, 255, 150),
        'Side line right': (0, 230, 0),
        'Side line top': (230, 0, 0),
        'Small rect. left bottom': (0, 150, 255),
        'Small rect. left main': (254, 173, 225),
        'Small rect. left top': (87, 72, 39),
        'Small rect. right bottom': (122, 0, 255),
        'Small rect. right main': (255, 255, 255),
        'Small rect. right top': (153, 23, 153)
    }


class SegmentationNetwork:
    def __init__(self, model_file, mean_file, std_file, num_classes=29, width=640, height=360):
        file_path = Path(model_file).resolve()
        model = nn.DataParallel(deeplabv3_resnet50(weights=None,weights_backbone=None, num_classes=num_classes))
        self.init_weight(model, nn.init.kaiming_normal_,
                         nn.BatchNorm2d, 1e-3, 0.1,
                         mode='fan_in')

        # Auto-detect best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("MPS (Apple Silicon) detected")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        checkpoint = torch.load(str(file_path), map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model.to(self.device)

        file_path = Path(mean_file).resolve()
        self.mean = np.load(str(file_path))
        file_path = Path(std_file).resolve()
        self.std = np.load(str(file_path))
        self.width = width
        self.height = height
        print(f"Device: {self.device}")
        print(f"Input resolution: {width}x{height}")
        print(f"Number of classes: {num_classes}")

    def init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                    **kwargs):
        for name, m in feature.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def analyse_image(self, image):
        """
        Process image and perform inference, returns mask of detected classes
        :param image: BGR image
        :return: predicted classes mask
        """
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = np.asarray(img, np.float32) / 255.
        img = (img - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            cuda_result = self.model.forward(img)
        output = cuda_result['out'].data[0].cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        return output


def create_color_mask(mask):
    """
    Convert segmentation mask to color visualization.

    Args:
        mask: segmentation mask (HxW) with class indices

    Returns:
        color_mask: RGB color mask (HxWx3)
    """
    # Create palette (index 0 is black for background)
    palette = [0, 0, 0]  # Background
    for class_name in SoccerPitch.lines_classes:
        palette.extend(SoccerPitch.palette[class_name])

    # Convert to color image
    color_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    color_mask.putpalette(palette)
    color_mask = color_mask.convert('RGB')

    return np.array(color_mask)


def blend_images(original, color_mask, alpha=0.6):
    """
    Blend original image with color mask.

    Args:
        original: Original BGR image
        color_mask: RGB color mask
        alpha: blending factor (original image weight)

    Returns:
        blended: Blended image
    """
    # Resize color_mask to match original
    color_mask_resized = cv2.resize(color_mask, (original.shape[1], original.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
    # Convert color_mask from RGB to BGR
    color_mask_bgr = cv2.cvtColor(color_mask_resized, cv2.COLOR_RGB2BGR)

    # Blend
    blended = cv2.addWeighted(original, alpha, color_mask_bgr, 1 - alpha, 0)
    return blended


def visualize_segmentation_detailed(image, mask, class_stats, annotate_confidence=False):
    """
    Create a detailed visualization with segmentation overlay and class statistics.

    Args:
        image: Original BGR image
        mask: Segmentation mask (HxW)
        class_stats: Dictionary of {class_name: percentage}
        annotate_confidence: If True, annotate confidence on each connected region

    Returns:
        viz: Visualization image with multiple panels
    """
    h, w = image.shape[:2]

    # Dynamic font scale based on image dimensions
    base_font_scale = min(h, w) / 1000.0
    title_font_scale = max(0.8, base_font_scale * 1.2)
    stats_font_scale = max(0.5, base_font_scale)
    stats_thickness = max(1, int(base_font_scale * 2))

    # Line height and indicator size scaled to image
    line_height = int(30 * base_font_scale)
    indicator_size = int(15 * base_font_scale)

    # Create color mask
    color_mask = create_color_mask(mask)

    # Annotate confidence on mask if requested
    if annotate_confidence:
        color_mask = annotate_mask_with_confidence(mask, class_stats)

    color_mask_resized = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    color_mask_bgr = cv2.cvtColor(color_mask_resized, cv2.COLOR_RGB2BGR)

    # Blend
    blended = cv2.addWeighted(image, 0.5, color_mask_bgr, 0.5, 0)

    # Create main panel (side by side)
    main_viz = np.hstack([blended, color_mask_bgr])
    main_viz_h, main_viz_w = main_viz.shape[:2]

    # Create horizontal stats panel at the bottom
    # Calculate stats panel height
    num_visible_classes = sum(1 for pct in class_stats.values() if pct >= 0.1)
    stats_panel_height = int(max(120 * base_font_scale, num_visible_classes * line_height + 50))
    stats_panel = np.ones((stats_panel_height, main_viz_w, 3), dtype=np.uint8) * 255

    # Draw title
    cv2.putText(stats_panel, "Detected Classes", (int(15 * base_font_scale), int(35 * base_font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 0, 0), stats_thickness)
    cv2.line(stats_panel,
             (int(15 * base_font_scale), int(42 * base_font_scale)),
             (main_viz_w - int(15 * base_font_scale), int(42 * base_font_scale)),
             (0, 0, 0), stats_thickness)

    # Draw class statistics in a horizontal layout (wrap to next line if needed)
    y_offset = int(60 * base_font_scale)
    x_offset = int(15 * base_font_scale)
    max_x = main_viz_w - int(15 * base_font_scale)

    for i, (class_name, percentage) in enumerate(class_stats.items()):
        if percentage < 0.1:
            continue

        # Get class color
        class_idx = list(SoccerPitch.lines_classes).index(class_name) if class_name in SoccerPitch.lines_classes else -1
        if class_idx >= 0:
            color = tuple(reversed(SoccerPitch.palette[class_name]))  # RGB to BGR
        else:
            color = (0, 0, 0)

        # Truncate class name if too long
        display_name = class_name[:25] if len(class_name) > 25 else class_name

        # Format text with confidence
        text = f"{display_name}: {percentage:5.2f}%"

        # Calculate text width
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, stats_thickness
        )

        # Check if we need to wrap to next line
        if x_offset + indicator_size * 2 + text_width > max_x:
            x_offset = int(15 * base_font_scale)
            y_offset += line_height

        # Draw colored rectangle (indicator)
        cv2.rectangle(stats_panel,
                      (x_offset, int(y_offset - line_height * 0.7)),
                      (x_offset + indicator_size, int(y_offset - line_height * 0.2)),
                      color, -1)

        # Draw text with class name and confidence
        cv2.putText(stats_panel, text, (x_offset + int(20 * base_font_scale), y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, (0, 0, 0), stats_thickness)

        x_offset += text_width + int(40 * base_font_scale)  # Add spacing between items

    # Resize main_viz to match stats panel width if needed
    full_viz = np.vstack([main_viz, stats_panel])

    return full_viz


def annotate_mask_with_confidence(mask, class_stats):
    """
    Annotate each connected component region on the mask with its class confidence.

    Args:
        mask: Segmentation mask (HxW) with class indices
        class_stats: Dictionary of {class_name: percentage}

    Returns:
        annotated_mask: Color mask with confidence labels drawn on each region
    """
    from scipy import ndimage

    h, w = mask.shape
    color_mask = create_color_mask(mask)

    # Dynamic font scale based on image dimensions
    base_font_scale = min(h, w) / 1000.0
    font_scale = max(0.4, base_font_scale * 0.8)
    font_thickness = max(1, int(base_font_scale * 1.5))

    # Process each class
    for class_idx_str, percentage in class_stats.items():
        if percentage < 0.5:  # Skip very small detections
            continue

        # Find class index
        if class_idx_str in SoccerPitch.lines_classes:
            class_idx = SoccerPitch.lines_classes.index(class_idx_str) + 1  # +1 because 0 is background
        else:
            continue

        # Create binary mask for this class
        class_mask = (mask == class_idx).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        # Label each connected component
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:  # Skip tiny regions
                continue

            # Get centroid
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            # Calculate bounding box for text background
            text = f"{percentage:.1f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Get class color (RGB)
            color = SoccerPitch.palette[class_idx_str]

            # Draw background rectangle for better visibility
            pad = 2
            cv2.rectangle(color_mask,
                         (cx - text_width // 2 - pad, cy - text_height - pad),
                         (cx + text_width // 2 + pad, cy + pad),
                         (0, 0, 0), -1)

            # Draw confidence text (white for contrast)
            cv2.putText(color_mask, text, (cx - text_width // 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return color_mask


def create_confidence_heatmap(mask, alpha=0.7):
    """
    Create a confidence heatmap based on class distribution.
    Areas with more diverse class predictions have higher confidence.

    Args:
        mask: Segmentation mask
        alpha: Transparency for overlay

    Returns:
        heatmap: Heatmap visualization
    """
    # For now, create a simple confidence map based on distance from background
    confidence = (mask > 0).astype(np.float32)

    # Apply Gaussian blur for smooth visualization
    from scipy.ndimage import gaussian_filter
    confidence = gaussian_filter(confidence, sigma=5)

    # Normalize to 0-1
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)

    # Convert to heatmap (blue -> green -> red)
    heatmap = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heatmap[:, :, 2] = (confidence * 255).astype(np.uint8)  # Red channel for confidence
    heatmap[:, :, 1] = ((1 - confidence) * 255).astype(np.uint8)  # Green for lower confidence
    heatmap[:, :, 0] = ((1 - confidence) * 128).astype(np.uint8)  # Blue for lower confidence

    return heatmap


def get_class_stats(mask):
    """Get class statistics from mask."""
    unique_classes, counts = np.unique(mask, return_counts=True)
    stats = {}
    total = mask.size
    for cls, count in zip(unique_classes, counts):
        if cls > 0 and cls <= len(SoccerPitch.lines_classes):
            class_name = SoccerPitch.lines_classes[cls - 1]
            percentage = 100.0 * count / total
            stats[class_name] = percentage
    # Sort by percentage
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


def test_image(net, image_path, output_dir, save_masks=False, show_visualization=True):
    """Test on a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    print(f"Processing: {image_path}")

    # Perform segmentation
    semlines = net.analyse_image(image)

    # Get class statistics
    class_stats = get_class_stats(semlines)

    # Create visualizations
    color_mask = create_color_mask(semlines)
    blended = blend_images(image, color_mask, alpha=0.6)

    # Create annotated mask with confidence labels on each region
    annotated_mask = annotate_mask_with_confidence(semlines, class_stats)

    # Create detailed visualization
    detailed_viz = visualize_segmentation_detailed(image, semlines, class_stats)

    # Save outputs
    image_name = Path(image_path).stem

    if save_masks:
        # Save raw mask
        mask_path = output_dir / f"mask_{image_name}.png"
        cv2.imwrite(str(mask_path), semlines)
        print(f"Saved mask: {mask_path}")

        # Save color mask
        color_mask_path = output_dir / f"color_mask_{image_name}.png"
        cv2.imwrite(str(color_mask_path), color_mask)
        print(f"Saved color mask: {color_mask_path}")

        # Save annotated mask with confidence
        annotated_mask_path = output_dir / f"annotated_mask_{image_name}.png"
        cv2.imwrite(str(annotated_mask_path), cv2.cvtColor(annotated_mask, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated mask: {annotated_mask_path}")

    # Save blended result
    blended_path = output_dir / f"blended_{image_name}.png"
    cv2.imwrite(str(blended_path), blended)
    print(f"Saved blended: {blended_path}")

    # Save detailed visualization
    viz_path = output_dir / f"detailed_viz_{image_name}.png"
    cv2.imwrite(str(viz_path), detailed_viz)
    print(f"Saved detailed visualization: {viz_path}")

    # Print class statistics
    print(f"\nClasses detected in {image_name}:")
    for class_name, percentage in class_stats.items():
        if percentage > 0.1:
            print(f"  - {class_name}: {percentage:.2f}%")
    print()

    # Show visualization if requested
    if show_visualization:
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(detailed_viz, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def test_video(net, video_path, output_dir, save_masks=False, max_frames=None,
               save_visualization=False):
    """Test on a video file."""
    global viz_out, viz_video_path
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Create output video writer
    video_name = Path(video_path).stem

    # Output paths
    output_video_path = output_dir / f"segmentation_{video_name}.mp4"
    if save_visualization:
        viz_video_path = output_dir / f"detailed_viz_{video_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if save_visualization:
        viz_out = cv2.VideoWriter(str(viz_video_path), fourcc, fps, (width * 2, height))

    frame_count = 0
    processed_frames = 0
    all_class_stats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and processed_frames >= max_frames:
            break

        print(f"Processing frame {frame_count}/{total_frames if total_frames > 0 else '?'}", end='\r')

        # Perform segmentation
        semlines = net.analyse_image(frame)

        # Get class statistics
        class_stats = get_class_stats(semlines)
        all_class_stats.append(class_stats)

        # Create color visualization
        color_mask = create_color_mask(semlines)
        blended = blend_images(frame, color_mask, alpha=0.6)

        # Write to output video
        out.write(blended)

        if save_visualization:
            # Create detailed visualization with confidence annotated on mask
            detailed_viz = visualize_segmentation_detailed(frame, semlines, class_stats, annotate_confidence=True)
            detailed_viz = cv2.resize(detailed_viz, (width * 2, height))
            viz_out.write(detailed_viz)

        if save_masks and frame_count % 10 == 0:  # Save every 10th frame mask
            mask_path = output_dir / f"mask_{video_name}_{frame_count:05d}.png"
            cv2.imwrite(str(mask_path), semlines)

        processed_frames += 1

    cap.release()
    out.release()
    if save_visualization:
        viz_out.release()

    print(f"\nCompleted! Processed {processed_frames} frames.")
    print(f"Output video: {output_video_path}")
    if save_visualization:
        print(f"Detailed visualization video: {viz_video_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if all_class_stats:
        # Aggregate all class detections
        total_stats = {}
        for stats in all_class_stats:
            for class_name, percentage in stats.items():
                if class_name not in total_stats:
                    total_stats[class_name] = []
                total_stats[class_name].append(percentage)

        print("Average class detection rates:")
        for class_name, percentages in sorted(total_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg = np.mean(percentages)
            if avg > 0.1:
                print(f"  - {class_name}: {avg:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Test soccer pitch segmentation model')
    parser.add_argument('--model', type=str,
                        default='models/soccer_pitch_segmentation.pth',
                        help='Path to the segmentation model')
    parser.add_argument('--mean', type=str,
                        default='sn-calibration/resources/mean.npy',
                        help='Path to mean normalization file')
    parser.add_argument('--std', type=str,
                        default='sn-calibration/resources/std.npy',
                        help='Path to std normalization file')
    parser.add_argument('--input', type=str,
                        default='input_vids/',
                        help='Path to input image/video folder')
    parser.add_argument('--output', type=str,
                        default='output/test_pitch_segmentation/',
                        help='Path to output folder')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width resolution')
    parser.add_argument('--height', type=int, default=360,
                        help='Input height resolution')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save raw masks in output directory')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save detailed visualization videos')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization windows for images')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize network
    net = SegmentationNetwork(
        args.model,
        args.mean,
        args.std,
        num_classes=29,
        width=args.width,
        height=args.height
    )

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    # Check if input is a directory
    input_path = Path(args.input)
    if input_path.is_dir():

        images = [f for f in input_path.glob('*') if f.suffix.lower() in image_extensions]
        videos = [f for f in input_path.glob('*') if f.suffix.lower() in video_extensions]

        print(f"Found {len(images)} images and {len(videos)} videos")

        # Process images first
        for img_path in images:
            test_image(net, img_path, output_dir, args.save_masks, show_visualization=args.show)

        # Process videos
        for vid_path in videos:
            test_video(net, vid_path, output_dir, args.save_masks, args.max_frames,
                      save_visualization=args.save_viz)

    elif input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            test_image(net, input_path, output_dir, args.save_masks, show_visualization=args.show)
        elif input_path.suffix.lower() in video_extensions:
            test_video(net, input_path, output_dir, args.save_masks, args.max_frames,
                      save_visualization=args.save_viz)
        else:
            print(f"Unsupported file type: {input_path.suffix}")
    else:
        print(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
