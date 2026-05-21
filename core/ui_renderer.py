"""
UI Renderer for Goalkeeper Highlight Generator
Provides real-time statistics, mini-map
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from datetime import datetime


class UIRenderer:
    """
    Enhanced UI rendering with:
    - Real-time statistics panel
    - Mini-map for trajectory overview
    - Goalkeeper identification
    - Ball trajectory history
    - Player detection boxes with detailed info
    - Player velocity estimation
    """

    def __init__(self, width, height, has_display=True):
        self.width = width
        self.height = height
        self.has_display = has_display

        # Panel configuration
        self.stats_panel_width = 280
        self.minimap_size = 150
        self.padding = 10

        # Colors (BGR)
        self.colors = {
            'bg_dark': (20, 20, 30),
            'bg_light': (40, 40, 55),
            'text_white': (255, 255, 255),
            'text_gray': (180, 180, 180),
            'ball_yellow': (0, 255, 255),
            'player_green': (0, 255, 0),
            'goalkeeper_red': (0, 0, 255),
            'goalkeeper_blue': (255, 0, 0),
        }

        # Trajectory storage
        self.ball_trajectory = deque(maxlen=100)
        self.player_trajectories = defaultdict(lambda: deque(maxlen=50))
        self.player_positions_history = defaultdict(lambda: deque(maxlen=5))

        # Statistics
        self.stats = {
            'total_players': 0,
            'goalkeepers': [],
            'ball_possession': 'unknown',
            'frame_count': 0,
            'fps_current': 0,
            'fps_average': 0,
        }

        # Frame timing
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = datetime.now()

        # Goalkeeper detection - players near goal areas
        self.potential_goalkeepers = {}

        # Possession tracking
        self.possession_frames = {'home': 0, 'away': 0, 'neutral': 0}
        self.total_frames = 0

    def render(self, frame, ball_xy, ball_state, boxes, ids,
               player_data=None, M_to_ref=None, frame_idx=0):
        """
        Main render function - draws all UI elements on frame.

        Args:
            frame: Input video frame
            ball_xy: Ball position (x, y)
            ball_state: Ball state string ('VISIBLE'/'OCCLUDED')
            boxes: Player bounding boxes (x1, y1, x2, y2)
            ids: Player track IDs
            player_data: Optional dict with additional player info
            M_to_ref: Camera motion compensation matrix
            frame_idx: Current frame index
        """
        # Create main display area and stats panel
        display_width = self.width
        main_frame = frame.copy()
        stats_panel = np.zeros((self.height, self.stats_panel_width, 3), dtype=np.uint8)

        # Identify goalkeepers
        self._identify_goalkeepers(boxes, ids, ball_xy)

        # Calculate player velocities
        velocities = self._calculate_velocities(ids, boxes)

        # Update statistics
        self._update_stats(frame_idx, ids, ball_state, velocities)

        # Draw ball and trajectory
        self._draw_ball(main_frame, ball_xy, ball_state)

        # Draw players with enhanced info
        self._draw_players(main_frame, boxes, ids, velocities)

        # Draw mini-map
        self._draw_minimap(stats_panel, ball_xy, ids, boxes)

        # Draw statistics panel
        self._draw_stats_panel(stats_panel, ball_state, M_to_ref, velocities)

        # Draw possession stats
        self._draw_possession_stats(stats_panel)

        # Combine main frame and stats panel
        combined = np.hstack([main_frame, stats_panel])

        return combined

    def _update_stats(self, frame_idx, ids, ball_state, velocities):
        """Update internal statistics."""
        current_time = datetime.now()
        delta = (current_time - self.last_frame_time).total_seconds()
        if delta > 0:
            self.frame_times.append(1.0 / delta)
            self.stats['fps_current'] = self.frame_times[-1]
            self.stats['fps_average'] = np.mean(list(self.frame_times))
        self.last_frame_time = current_time

        self.stats['frame_count'] = frame_idx
        self.stats['total_players'] = len(set(ids)) if ids is not None else 0
        self.stats['ball_state'] = ball_state
        self.stats['velocities'] = velocities

        # Update possession
        self.total_frames += 1

    def _draw_ball(self, frame, ball_xy, ball_state):
        """Draw ball with trajectory and status."""
        if ball_xy is None:
            return

        x, y = ball_xy

        # Store trajectory
        self.ball_trajectory.append((x, y))

        # Draw ball glow effect
        for radius in range(12, 4, -2):
            alpha = (radius - 4) / 8.0
            color = tuple(int(c * alpha * 0.5) for c in self.colors['ball_yellow'])
            cv2.circle(frame, (x, y), radius, color, -1)

        # Draw ball
        cv2.circle(frame, (x, y), 8, self.colors['ball_yellow'], -1)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 1)

        # Draw ball trajectory
        if len(self.ball_trajectory) > 1:
            pts = list(self.ball_trajectory)
            for i in range(1, len(pts)):
                x1, y1 = pts[i-1]
                x2, y2 = pts[i]
                alpha = i / len(pts)
                color = tuple(int(c * alpha) for c in self.colors['ball_yellow'])
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Draw ball state label
        label = f"BALL: {ball_state}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_bg = (
            max(0, x - label_size[0] // 2 - 4),
            max(0, y - 25),
            x + label_size[0] // 2 + 4,
            y - 5
        )
        cv2.rectangle(frame, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]),
                     self.colors['bg_dark'], -1)
        cv2.putText(frame, label, (x - label_size[0] // 2, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['ball_yellow'], 1)

    def _draw_players(self, frame, boxes, ids, velocities):
        """Draw players with bounding boxes, IDs, and additional info."""
        if boxes is None or ids is None:
            return

        for idx, (track_id, box) in enumerate(zip(ids, boxes)):
            x1, y1, x2, y2 = map(int, box[:4])
            track_id = int(track_id)

            # Check if this is a goalkeeper
            is_gk = track_id in self.potential_goalkeepers

            # Get color for this track (red for goalkeepers)
            if is_gk:
                color = self.colors['goalkeeper_red']
            else:
                color = self._get_track_color(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw corners for style
            corner_len = 10
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Calculate player center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Store trajectory
            self.player_trajectories[track_id].append((cx, cy))

            # Draw player trajectory (faded)
            traj = list(self.player_trajectories[track_id])
            if len(traj) > 1:
                for i in range(1, len(traj)):
                    alpha = i / len(traj) * 0.6
                    pt1 = traj[i-1]
                    pt2 = traj[i]
                    color_faded = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                           (int(pt2[0]), int(pt2[1])), color_faded, 1)

            # Draw ID label with background
            id_label = f"ID:{track_id}"

            # Add velocity if available
            if velocities and track_id in velocities:
                vel = velocities[track_id]
                # Convert pixels/sec to approximate m/s (assuming ~25 pixels = 1 meter)
                vel_ms = vel / 25
                id_label += f" {vel_ms:.1f}m/s"

            # Add GK label for goalkeepers
            if is_gk:
                id_label += " [GK]"

            label_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0] + 8, y1), color, -1)
            cv2.putText(frame, id_label, (x1 + 4, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw detection confidence if available
            if len(box) > 4:
                conf = box[4]
                conf_label = f"{conf:.2f}"
                cv2.putText(frame, conf_label, (x2 - label_size[0] - 4, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _get_track_color(self, track_id):
        """Generate consistent color for track ID."""
        np.random.seed(track_id)
        return tuple(int(c) for c in np.random.randint(50, 255, 3))

    def _identify_goalkeepers(self, boxes, ids, ball_xy):
        """
        Identify potential goalkeepers based on:
        - Position near goal areas
        - Movement patterns (goalkeepers move differently)
        - Distance from ball
        """
        if boxes is None or ids is None:
            return

        self.potential_goalkeepers = {}

        for idx, (track_id, box) in enumerate(zip(ids, boxes)):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Normalize position
            nx, ny = cx / self.width, cy / self.height

            # Check if player is near left or right goal area
            is_near_left_goal = nx < 0.15 and 0.3 < ny < 0.7
            is_near_right_goal = nx > 0.85 and 0.3 < ny < 0.7

            # Calculate distance to ball
            dist_to_ball = None
            if ball_xy is not None:
                dist_to_ball = np.sqrt((cx - ball_xy[0])**2 + (cy - ball_xy[1])**2)

            # Determine if this is likely a goalkeeper
            is_gk = False
            gk_side = None

            if is_near_left_goal:
                is_gk = True
                gk_side = 'left'
            elif is_near_right_goal:
                is_gk = True
                gk_side = 'right'

            if is_gk:
                self.potential_goalkeepers[track_id] = {
                    'side': gk_side,
                    'position': (cx, cy),
                    'dist_to_ball': dist_to_ball
                }

    def _calculate_velocities(self, ids, boxes):
        """Calculate player velocities based on position changes."""
        velocities = {}

        if boxes is None or ids is None:
            return velocities

        for idx, (track_id, box) in enumerate(zip(ids, boxes)):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Store current position
            self.player_positions_history[track_id].append((cx, cy))

            # Need at least 2 positions to calculate velocity
            positions = list(self.player_positions_history[track_id])
            if len(positions) >= 2:
                # Calculate displacement from previous frame
                prev_cx, prev_cy = positions[-2]
                displacement = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

                # Convert to approximate velocity (pixels per frame, assuming 30fps)
                # This is a rough estimate - actual velocity would need real time info
                velocity = displacement * 30  # pixels per second
                velocities[track_id] = velocity

        return velocities

    def _draw_possession_stats(self, panel):
        """Draw ball possession statistics."""
        # Calculate possession percentages
        total = self.total_frames if self.total_frames > 0 else 1

        # Simple possession based on ball position
        if self.ball_trajectory:
            recent_positions = list(self.ball_trajectory)[-30:]
            left_side = sum(1 for x, y in recent_positions if x < self.width / 3)
            right_side = sum(1 for x, y in recent_positions if x > 2 * self.width / 3)

            home_pct = (left_side / len(recent_positions) * 100) if recent_positions else 50
            away_pct = (right_side / len(recent_positions) * 100) if recent_positions else 50
        else:
            home_pct = away_pct = 50

        # Draw possession bar
        poss_y = self.height - 80
        poss_x = self.padding
        poss_w = self.stats_panel_width - 2 * self.padding
        poss_h = 20

        # Background
        cv2.rectangle(panel, (poss_x, poss_y), (poss_x + poss_w, poss_y + poss_h),
                     self.colors['bg_dark'], -1)

        # Home team (left) - blue
        home_w = int(poss_w * (home_pct / 100))
        cv2.rectangle(panel, (poss_x, poss_y), (poss_x + home_w, poss_y + poss_h),
                     (255, 200, 0), -1)

        # Away team (right) - red
        away_w = int(poss_w * (away_pct / 100))
        cv2.rectangle(panel, (poss_x + poss_w - away_w, poss_y),
                     (poss_x + poss_w, poss_y + poss_h), (0, 0, 255), -1)

        # Labels
        cv2.putText(panel, f"HOME {home_pct:.0f}%", (poss_x + 4, poss_y + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(panel, f"AWAY {away_pct:.0f}%", (poss_x + poss_w - 55, poss_y + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Title
        cv2.putText(panel, "POSSESSION (30 frames)", (poss_x + 4, poss_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text_gray'], 1)

    def _draw_minimap(self, panel, ball_xy, ids, boxes):
        """Draw mini-map showing field overview."""
        # Mini-map area
        map_x = self.padding
        map_y = self.padding
        map_w = self.stats_panel_width - 2 * self.padding
        map_h = map_w  # Square mini-map

        # Background
        cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h),
                     self.colors['bg_light'], -1)
        cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h),
                     (80, 80, 80), 1)

        # Draw field outline (simplified)
        field_margin = 10
        field_x = map_x + field_margin
        field_y = map_y + field_margin
        field_w = map_w - 2 * field_margin
        field_h = map_h - 2 * field_margin

        # Field green background
        cv2.rectangle(panel, (field_x, field_y), (field_x + field_w, field_y + field_h),
                     (30, 80, 30), -1)

        # Center line
        cv2.line(panel, (field_x + field_w // 2, field_y),
                (field_x + field_w // 2, field_y + field_h), (200, 200, 200), 1)

        # Center circle
        cv2.ellipse(panel, (field_x + field_w // 2, field_y + field_h // 2),
                   (field_w // 6, field_h // 4), 0, 0, 360, (200, 200, 200), 1)

        # Goal areas
        cv2.rectangle(panel, (field_x, field_y + field_h // 3),
                     (field_x + field_w // 6, field_y + 2 * field_h // 3),
                     (200, 200, 200), 1)
        cv2.rectangle(panel, (field_x + 5 * field_w // 6, field_y + field_h // 3),
                     (field_x + field_w, field_y + 2 * field_h // 3),
                     (200, 200, 200), 1)

        # Draw ball position on mini-map
        if ball_xy is not None:
            mx = field_x + int(ball_xy[0] / self.width * field_w)
            my = field_y + int(ball_xy[1] / self.height * field_h)
            mx = max(field_x, min(field_x + field_w, mx))
            my = max(field_y, min(field_y + field_h, my))
            cv2.circle(panel, (mx, my), 5, self.colors['ball_yellow'], -1)

        # Draw player positions on mini-map
        if ids is not None and boxes is not None:
            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                mx = field_x + int(cx / self.width * field_w)
                my = field_y + int(cy / self.height * field_h)
                mx = max(field_x, min(field_x + field_w, mx))
                my = max(field_y, min(field_y + field_h, my))

                color = self._get_track_color(int(track_id))
                cv2.circle(panel, (mx, my), 3, color, -1)

        # Mini-map label
        cv2.putText(panel, "MINI-MAP", (map_x + 4, map_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_white'], 1)

    def _draw_stats_panel(self, panel, ball_state, M_to_ref, velocities):
        """Draw statistics panel with detailed information."""
        panel_x = self.padding
        panel_y = self.minimap_size + 2 * self.padding

        # Section: Frame Info
        self._draw_section_header(panel, panel_x, panel_y, "FRAME INFO")
        panel_y += 25

        info_items = [
            ("Frame:", str(self.stats['frame_count'])),
            ("FPS (Cur):", f"{self.stats['fps_current']:.1f}"),
            ("FPS (Avg):", f"{self.stats['fps_average']:.1f}"),
            ("Players:", str(self.stats['total_players'])),
        ]

        for label, value in info_items:
            self._draw_stat_row(panel, panel_x, panel_y, label, value)
            panel_y += 20

        # Section: Goalkeeper Info
        panel_y += 10
        self._draw_section_header(panel, panel_x, panel_y, "GOALKEEPERS")
        panel_y += 25

        if self.potential_goalkeepers:
            for gk_id, gk_info in self.potential_goalkeepers.items():
                gk_label = f"ID {gk_id} ({gk_info['side']})"
                cv2.putText(panel, gk_label, (panel_x, panel_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['goalkeeper_red'], 1)
                panel_y += 18
        else:
            cv2.putText(panel, "No GK detected", (panel_x, panel_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_gray'], 1)
            panel_y += 20

        # Section: Player Velocities
        panel_y += 10
        self._draw_section_header(panel, panel_x, panel_y, "PLAYER VELOCITIES")
        panel_y += 25

        if velocities:
            # Show top 3 fastest players
            sorted_vels = sorted(velocities.items(), key=lambda x: x[1], reverse=True)[:3]
            for track_id, vel in sorted_vels:
                vel_ms = vel / 25  # Convert to approximate m/s
                vel_label = f"ID {track_id}: {vel_ms:.1f} m/s"
                color = self.colors['goalkeeper_red'] if track_id in self.potential_goalkeepers else self.colors['text_white']
                cv2.putText(panel, vel_label, (panel_x, panel_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                panel_y += 18
        else:
            cv2.putText(panel, "Calculating...", (panel_x, panel_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_gray'], 1)
            panel_y += 20

        # Section: Ball Status
        panel_y += 10
        self._draw_section_header(panel, panel_x, panel_y, "BALL STATUS")
        panel_y += 25

        ball_color = self.colors['ball_yellow'] if ball_state == 'VISIBLE' else self.colors['text_gray']
        self._draw_stat_row(panel, panel_x, panel_y, "State:", ball_state,
                           value_color=ball_color)
        panel_y += 20

        self._draw_stat_row(panel, panel_x, panel_y, "Trajectory:",
                           f"{len(self.ball_trajectory)} pts")
        panel_y += 20

        # Section: Camera Motion
        panel_y += 10
        self._draw_section_header(panel, panel_x, panel_y, "CAMERA MOTION")
        panel_y += 25

        if M_to_ref is not None:
            # Calculate motion magnitude from transformation matrix
            tx, ty = M_to_ref[0, 2], M_to_ref[1, 2]
            motion_mag = np.sqrt(tx**2 + ty**2)
            motion_dir = np.degrees(np.arctan2(ty, tx))

            panel_y += 20

            self._draw_stat_row(panel, panel_x, panel_y, "Direction:",
                               f"{motion_dir:.0f} deg")
            panel_y += 20

            # Motion vector visualization
            vector_x = panel_x + 80
            vector_y = panel_y + 5
            vector_scale = 3
            end_x = int(vector_x + tx * vector_scale)
            end_y = int(vector_y + ty * vector_scale)
            cv2.line(panel, (vector_x, vector_y), (end_x, end_y),
                    self.colors['player_green'], 2)
            cv2.circle(panel, (vector_x, vector_y), 3, (255, 255, 255), -1)
        else:
            self._draw_stat_row(panel, panel_x, panel_y, "Status:", "Not compensated",
                               value_color=self.colors['text_gray'])
            panel_y += 40

        # Section: Legend
        panel_y += 20
        self._draw_section_header(panel, panel_x, panel_y, "LEGEND")
        panel_y += 25

        legend_items = [
            ("Ball", self.colors['ball_yellow']),
            ("Player", self.colors['player_green']),
            ("Goalkeeper", self.colors['goalkeeper_red']),
            ("Trajectory", (100, 100, 100)),
        ]

        for label, color in legend_items:
            cv2.circle(panel, (panel_x + 10, panel_y + 8), 5, color, -1)
            cv2.putText(panel, label, (panel_x + 22, panel_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_white'], 1)
            panel_y += 20

    def _draw_section_header(self, panel, x, y, title):
        """Draw section header with separator line."""
        cv2.putText(panel, title, (x, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_white'], 1)
        cv2.line(panel, (x, y + 18), (x + self.stats_panel_width - 2 * self.padding, y + 18),
                (60, 60, 60), 1)

    def _draw_stat_row(self, panel, x, y, label, value, value_color=None):
        """Draw a single statistics row."""
        if value_color is None:
            value_color = self.colors['text_white']

        cv2.putText(panel, label, (x, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_gray'], 1)
        cv2.putText(panel, str(value), (x + 80, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, value_color, 1)