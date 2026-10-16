"""Cluster tracked people into 2 teams + referee and goalkeepers

Input is the return of ``PlayerTracker.update`` — ``[{id, bbox, cls, conf, mask}]``
where ``mask`` is a 0/1 raster local to ``bbox``

Two stages:
1. ``identify_jersey_colour`` extract a single colour from every mask
2. ``cluster_colours`` finds the two teams among those per-player colours

Only deal with people standing on the pitch — see ``filter_on_pitch``.

Colour is in CIE-Lab. Clustering uses ``a``/``b`` plus ``L`` at a third of its
scale. Chroma identifies a kit; ``L`` is what separates a black shirt from a
white one — and a referee from a white-shirted team — because both are
achromatic and share the same ``a``/``b``. ``L`` enters muted rather than raw
because it also swings with sun and shadow across the pitch.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque, defaultdict, Counter
from typing import NamedTuple

from core.projection_utils import pixel_to_ground
from core.player_position import foot_point

TEAM_A = 0
TEAM_B = 1

PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"

# Third kits keep fixed swatches; teams are drawn in their own jersey colour.
ROLE_COLOURS = {REFEREE: (0, 255, 255), GOALKEEPER: (255, 0, 255)}
UNKNOWN_COLOUR = (128, 128, 128)

# the weight of Lightness in clustering
L_WEIGHT = 1.0 / 2.5

REF_DIST_FACTOR = 2.4 # distance from the nearest team colour, relative to the spread of the tighter team
REF_DIST_FLOOR = 25.0 # floor for ref dist

TORSO_TOP = 0.20 # head and hair
TORSO_BOTTOM = 0.55 # shorts and socks

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_MARGIN = 1.0 # margin of pitch when deciding on pitch people


class Assignment(NamedTuple):
    """
    Hashable, so a whole history can be voted on as-is. ``team`` is
    ``TEAM_A``/``TEAM_B``, or None for a keeper or the referee.
    """
    team: int | None
    role: str

class TeamAssigner:
    """Assign a team and a role to each tracked person.

    Args:
        min_pixels: fewest torso pixels worth clustering.
        max_samples: cap on the pixels fed to a per-player k-means.
    """

    def __init__(
        self,
        min_pixels: int = 30,
        max_samples: int = 1000,
    ):
        self.min_pixels = min_pixels
        self.max_samples = max_samples

        self._kernel = np.ones((3, 3), np.uint8)
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )

        self.team_colours: dict[int, np.ndarray] = {} # team -> (3,) mean [L, a, b]
        self.colour_acc: dict[int, deque] = defaultdict(lambda: deque(maxlen=20)) # id -> [LAB of each det]
        self.assign_acc: dict[int, deque] = defaultdict(lambda: deque(maxlen=20)) # id -> [Assignments of each det]
        self.voted_colours: dict[int, np.ndarray] = {} # id -> mean LAB of the history
        self.voted_assignments: dict[int, Assignment] = {} # id -> voted assignments
        self.off_pitch: set[int] = set() # ids off pitch

    # ----------------------------------------------------------- the pitch

    def filter_on_pitch(self, tracked_objects, K=None, R=None, t=None, positions=None):
        if positions is None and (K is None or R is None or t is None):
            return list(tracked_objects), set()

        on_pitch, off_pitch = [], set()
        for obj in tracked_objects:
            if self.inside_pitch(self.ground_position(obj, K, R, t, positions)):
                on_pitch.append(obj)
            else:
                off_pitch.add(obj["id"])
        return on_pitch, off_pitch

    # ---------------------------------------------------------------- stage 1

    def torso_lab(self, obj, frame):
        """extract LAB colour from players' torso

        Args:
            obj: the return from ``PlayerTracker.update``.
            frame: the BGR frame ``obj["bbox"]`` refers to.

        Returns:
            (N, 3) float32 CIE-Lab ``[L, a, b]``
            or None if the player is too small to read a colour from.
        """
        mask = obj.get("mask")
        if mask is None:
            return None

        x1, y1, x2, y2 = (int(p) for p in obj["bbox"])
        h = y2 - y1
        if h <= 0 or x2 <= x1:
            return None

        top = int(TORSO_TOP * h)
        bottom = int(TORSO_BOTTOM * h)
        band = mask[top:bottom]
        if band.size == 0:
            return None

        # The border must count as 0: erode's default treats everything outside
        band = cv2.erode(
            band, self._kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        if int(np.count_nonzero(band)) < self.min_pixels:
            return None

        crop = frame[y1 + top : y1 + bottom, x1:x2]
        if crop.shape[:2] != band.shape:
            return None  # bbox ran off the edge of the frame

        pixels = crop[band.astype(bool)].reshape(-1, 1, 3)
        if len(pixels) < self.min_pixels:
            return None

        lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return lab.astype(np.float32)

    def identify_jersey_colour(self, obj, frame):
        """Extract a single colour from a player

        Args:
            obj: the return from ``PlayerTracker.update``.
            frame: the BGR frame ``obj["bbox"]`` refers to.

        Returns:
            (3,) float32 CIE-Lab ``[L, a, b]`` — OpenCV's channel order — or
            None if the player is too small to read a colour from.
        """
        lab = self.torso_lab(obj, frame)
        if lab is None:
            return None
        return self._dominant_lab(lab)

    def _dominant_lab(self, lab):
        """
        The larger of two k-means clusters over (N, 3) Lab pixels.
        The split itself is on chroma only
        """
        if len(lab) < 2:
            return lab.mean(axis=0)

        if len(lab) > self.max_samples:
            rng = np.random.default_rng(0)
            lab = lab[rng.choice(len(lab), self.max_samples, replace=False)]

        labels, _ = self._kmeans(lab[:, 1:], 2, attempts=3)
        counts = np.bincount(labels, minlength=2)
        return lab[labels == counts.argmax()].mean(axis=0)

    def _kmeans(self, data, k, attempts):
        """ ``cv2.kmeans``, made reproducible """
        cv2.setRNGSeed(0)
        _, labels, centres = cv2.kmeans(
            np.ascontiguousarray(data, np.float32),
            k,
            None,
            self._criteria,
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        return labels.ravel(), centres

    # ---------------------------------------------------------------- stage 2

    def cluster_colours(self, colours, k: int = 2):
        """K-means over one colour vector per player.

        Args:
            colours: (N, D) array, one row per player.
            k: number of clusters.
        Returns:
            ``(labels (N,), centres (k, D))``;
            With fewer than ``k`` players every player is its own cluster
        """
        data = np.ascontiguousarray(colours, dtype=np.float32)
        if len(data) < k:
            return np.arange(len(data)), data.copy()

        return self._kmeans(data, k, attempts=10)


    # ------------------------------------------------------------- per frame

    def update(self, frame, tracked_objects, K=None, R=None, t=None, positions=None):
        """With history data and new detections, re-cluster every known player.
        So it can handle players that are occluded or motion-blurred.
        Only deal with people on pitch — ``filter_on_pitch`` runs first

        Args:
            frame: current frame
            tracked_objects: as returned by ``PlayerTracker.update``.
            K, R, t: pitch calibration, for position information
            positions: ``{id: (x, y)}`` world metres, already filtered. When
                given it is used instead of projecting the raw boxes, so the
                verdict here matches the BEV the caller draws.

        Returns:
            ``{id: {"team": TEAM_A | TEAM_B | None, "role": str}}``
        """
        on_pitch, self.off_pitch = self.filter_on_pitch(
            tracked_objects, K, R, t, positions
        )

        for obj in on_pitch:
            colour = self.identify_jersey_colour(obj, frame)
            if colour is None:
                continue
            self.colour_acc[obj["id"]].append(colour)
        self.team_colours = {}

        # Mean the history once here, not on every read: it is the same dict
        self.voted_colours = {
            tid: np.mean(np.stack(list(c)), axis=0).astype(np.float32)
            for tid, c in self.colour_acc.items()
        }

        ppl_on_pitch = {obj["id"] for obj in on_pitch}
        detected = {obj["id"] for obj in tracked_objects}
        # players detected on-pitch, or not detected in this frame
        ids = [tid for tid in self.voted_colours if tid in ppl_on_pitch or tid not in detected]
        if len(ids) < 2: return None # one player only

        colours = np.stack([self.voted_colours[tid] for tid in ids]) # tid stands for track-id
        features = self.lab_cv2kmeans(colours)
        labels, centres = self.cluster_colours(features)
        labels, centres = self.sort_label(labels, centres, features)

        dist = np.linalg.norm(features[:, None, :] - centres[None, :, :], axis=2)
        nearest = dist.min(axis=1)

        # print("=== Player ID and Nearest Distance ===")
        # for i, tid in enumerate(ids):
        #     print(f"ID: {tid}, Nearest Distance: {nearest[i]:.4f}")
        # print("=" * 40)

        spreads = []
        for cluster in range(len(centres)):
            members = features[labels == cluster]
            if len(members):
                spreads.append(
                    np.median(np.linalg.norm(members - centres[cluster], axis=1))
                )
        threshold = max(
            REF_DIST_FLOOR,
            REF_DIST_FACTOR * (min(spreads) if spreads else 0.0),
        )

        pitch_obj_map = {obj["id"]: obj for obj in on_pitch}
        outliers = {}  # tid -> ground_position

        for tid, label, dist in zip(ids, labels, nearest):
            if dist <= threshold:
                self.assign_acc[tid].append(Assignment(int(label), PLAYER))
                continue

            obj = pitch_obj_map.get(tid)
            if obj is None: # not detected in this frame
                continue

            position = self.ground_position(obj, K, R, t, positions)
            if position is not None:  # the ray can miss the ground plane
                outliers[tid] = position
        if outliers:
            goal = np.array([PITCH_LENGTH / 2, 0.0])

            gk_id = min(outliers,
                key=lambda tid: np.linalg.norm(outliers[tid] - goal),
            )
            self.assign_acc[gk_id].append(Assignment(None, GOALKEEPER))

            for tid in outliers:
                if tid != gk_id:
                    self.assign_acc[tid].append(Assignment(None, REFEREE))

        # voting the current assignments
        self.voted_assignments = {
            tid: Counter(history).most_common(1)[0][0]
            for tid, history in self.assign_acc.items() if history
        }

        # Each team's jersey colour for drawing, remove L's weight
        for team in (TEAM_A, TEAM_B):
            members = [
                self.voted_colours[tid]
                for tid, a in self.voted_assignments.items()
                if a.team == team
            ]
            if members:
                self.team_colours[team] = np.mean(members, axis=0)

    # ---------------------------------------------------------------- drawing

    def draw_colour(self, tid):
        """BGR swatch for a track id. """
        assignment = self.voted_assignments.get(tid)
        if assignment is None:
            return UNKNOWN_COLOUR
        if assignment.team is not None:
            lab = self.team_colours.get(assignment.team)
            if lab is not None:
                return lab_to_bgr(lab)
        return ROLE_COLOURS.get(assignment.role, UNKNOWN_COLOUR)

    @staticmethod
    def ground_position(obj, K, R, t, positions=None):
        """Bottom-centre of each box on the ground plane, in world metres.

        Bottom-centre because that is where a player meets the pitch.

        ``positions`` (from ``core.player_position.PlayerPositionFilter``) is the
        same point after temporal filtering. Prefer it when available: the raw
        box moves the ground point by up to 0.6 m per pixel at the far goal
        line, and ``inside_pitch`` judges against a +/-1 m margin, so an
        unfiltered foot flips players across the touchline frame by frame.
        """
        if positions is not None:
            point = positions.get(obj["id"])
            if point is not None:
                return float(point[0]), float(point[1])
        if K is None or R is None or t is None:
            return None
        point = pixel_to_ground(*foot_point(obj), K, R, t)
        if point is None:
            return None
        return float(point[0]), float(point[1])

    @staticmethod
    def lab_cv2kmeans(colours):
        """Clustering ``(a, b, L * L_WEIGHT)``"""
        features = np.empty_like(colours, dtype=np.float32)
        features[:, 0] = colours[:, 1]  # a
        features[:, 1] = colours[:, 2]  # b
        features[:, 2] = colours[:, 0] * L_WEIGHT  # L, weighted
        return features

    @staticmethod
    def sort_label(labels, centres, features):
        """ kmeans produces random label, so we need to sort it for binding teams.
            Team A always has lower `a`
        """
        keys = []
        for cluster in range(len(centres)):
            members = features[labels == cluster]
            keys.append(tuple(np.median(members, axis=0)) if len(members) else (np.inf,) * 3)

        order = sorted(range(len(centres)), key=lambda cluster: keys[cluster])
        remap = np.empty(len(order), dtype=int)
        remap[order] = np.arange(len(order))
        return remap[labels], centres[order]

    @staticmethod
    def inside_pitch(point):
        if point is None:
            return False
        return (
            abs(float(point[0])) <= PITCH_LENGTH / 2 + PITCH_MARGIN
            and abs(float(point[1])) <= PITCH_WIDTH / 2 + PITCH_MARGIN
        )

def lab_to_bgr(lab):
    """The BGR triple a float CIE-Lab colour corresponds to.

    OpenCV's 8-bit Lab: L, a and b all span 0-255, with a = b = 128 neutral.
    """
    pixel = np.rint(np.clip(lab, 0, 255)).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)