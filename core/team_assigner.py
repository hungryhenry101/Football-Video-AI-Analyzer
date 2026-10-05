"""Cluster tracked people into 2 teams + referee and goalkeepers

Input is the return of ``PlayerTracker.update`` — ``[{id, bbox, cls, conf, mask}]``
where ``mask`` is a 0/1 raster local to ``bbox``

Two stages:
1. ``identify_jersey_colour`` extract a single colour from every mask
2. ``cluster_colours`` finds the two teams among those per-player colours

Colour is in CIE-Lab. Clustering uses ``a``/``b`` plus ``L`` at a third of its
scale. Chroma identifies a kit; ``L`` is what separates a black shirt from a
white one — and a referee from a white-shirted team — because both are
achromatic and share the same ``a``/``b``. ``L`` enters muted rather than raw
because it also swings with sun and shadow across the pitch.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.projection_utils import pixel_to_ground

TEAM_A = 0
TEAM_B = 1

PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"

# Third kits keep fixed swatches; teams are drawn in their own jersey colour.
ROLE_COLOURS = {REFEREE: (0, 255, 255), GOALKEEPER: (255, 0, 255)}
UNKNOWN_COLOUR = (128, 128, 128)

# How much of L enters the team split. A black kit and a white one sit at the
# same chroma (a = b = 128) and differ by ~210 in L, so L has to be present;
# but sun and shadow move L across the pitch far more than they move chroma, so
# it has to be quiet. A third keeps black-vs-white (~70 in feature units) well
# clear of a shade boundary (~20-27) while leaving chroma dominant.
L_WEIGHT = 1.0 / 2.5

REF_DIST_FACTOR = 2.4 # distance from the nearest team colour, relative to the spread of the tighter team
REF_DIST_FLOOR = 25.0 # floor for ref dist

TORSO_TOP = 0.20 # head and hair
TORSO_BOTTOM = 0.55 # shorts and socks

# A keeper inside their six-yard box: within 5.5 m of the goal line, and inside
# the 18.32 m goal-area width. The referee works the central corridor instead.
GOAL_LINE_X = 47.0
GOAL_MOUTH_Y = 9.2

def lab_to_bgr(lab):
    """The BGR triple a float CIE-Lab colour corresponds to.

    OpenCV's 8-bit Lab: L, a and b all span 0-255, with a = b = 128 neutral.
    """
    pixel = np.rint(np.clip(lab, 0, 255)).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


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

        self.colours: dict[int, np.ndarray] = {}      # id -> (3,) mean [L, a, b]
        self.assignments: dict[int, dict] = {}        # id -> {"team", "role"}
        self.team_colours: dict[int, np.ndarray] = {} # team -> (3,) mean [L, a, b]
        self._acc: dict[int, list] = {}               # id -> [sum of LAB, count]

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

    @staticmethod
    def _lab_cv2kmeans(colours):
        """Clustering wants: ``(a, b, L * L_WEIGHT)``"""
        features = np.empty_like(colours, dtype=np.float32)
        features[:, 0] = colours[:, 1]             # a
        features[:, 1] = colours[:, 2]             # b
        features[:, 2] = colours[:, 0] * L_WEIGHT  # L, weighted
        return features

    @staticmethod
    def _sort_label(labels, centres, features):
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

    # ------------------------------------------------------------- per frame

    def update(self, frame, tracked_objects, K=None, R=None, t=None):
        """With history data and new detections, re-cluster every known player.

        So it can handle players that are occluded or motion-blurred.
        Re-clustering each frame is cheap: it runs over one colour vector per player, not over pixels.

        Args:
            frame: current frame
            tracked_objects: as returned by ``PlayerTracker.update``.
            K, R, t: pitch calibration, for position information

        Returns:
            ``{id: {"team": TEAM_A | TEAM_B | None, "role": str}}``.
            Players with no readable colour are absent.
            Outliers get ``team=None``.
        """
        for obj in tracked_objects:
            colour = self.identify_jersey_colour(obj, frame)
            if colour is None:
                continue
            acc = self._acc.setdefault(obj["id"], [np.zeros(3, np.float64), 0])
            acc[0] += colour
            acc[1] += 1

        self.colours = {
            tid: (total / count).astype(np.float32)
            for tid, (total, count) in self._acc.items()
        }
        self.assignments = {}
        self.team_colours = {}

        # One player is not two teams.
        if len(self.colours) < 2:
            return self.assignments

        ids = list(self.colours)
        colours = np.stack([self.colours[tid] for tid in ids])
        features = self._lab_cv2kmeans(colours)
        labels, centres = self.cluster_colours(features)
        labels, centres = self._sort_label(labels, centres, features)

        # A kit far from both team colours is a third kit: keeper or referee.
        dist = np.linalg.norm(features[:, None, :] - centres[None, :, :], axis=2)
        nearest = dist.min(axis=1)

        # Print each person's id and nearest distance
        print("=== Player ID and Nearest Distance ===")
        for i, tid in enumerate(ids):
            print(f"ID: {tid}, Nearest Distance: {nearest[i]:.4f}")
        print("=" * 40)

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

        positions = self._ground_positions(tracked_objects, K, R, t)
        for i, tid in enumerate(ids):
            if nearest[i] > threshold:
                self.assignments[tid] = {
                    "team": None,
                    "role": self._outlier_role(positions.get(tid)),
                }
            else:
                self.assignments[tid] = {"team": int(labels[i]), "role": PLAYER}

        # Each team's jersey colour, for drawing. remove L's weight
        for team in (TEAM_A, TEAM_B):
            members = [
                self.colours[tid]
                for tid, a in self.assignments.items()
                if a["team"] == team
            ]
            if members:
                self.team_colours[team] = np.mean(members, axis=0)

        return self.assignments

    def _ground_positions(self, tracked_objects, K, R, t):
        """Bottom-centre of each box on the ground plane, in world metres.

        Bottom-centre because that is where a player meets the pitch.
        """
        if K is None or R is None or t is None:
            return {}
        positions = {}
        for obj in tracked_objects:
            x1, _, x2, y2 = obj["bbox"]
            point = pixel_to_ground((x1 + x2) / 2, y2, K, R, t)
            if point is not None:
                positions[obj["id"]] = (float(point[0]), float(point[1]))
        return positions

    @staticmethod
    def _outlier_role(position):
        """A third kit is a keeper if it is in a six-yard box, else the referee."""
        if position is None:
            return REFEREE
        x, y = position
        if abs(x) >= GOAL_LINE_X and abs(y) <= GOAL_MOUTH_Y:
            return GOALKEEPER
        return REFEREE

    # ---------------------------------------------------------------- drawing

    def draw_colour(self, tid):
        """BGR swatch for a track id.

        A team member is drawn in their team's own jersey colour, so the
        overlay tells you which kit is on screen rather than which cluster
        index won. A third kit keeps its fixed role swatch — the point of the
        referee and the keeper is to read as "not a team" at a glance.
        """
        assignment = self.assignments.get(tid)
        if assignment is None:
            return UNKNOWN_COLOUR
        if assignment["team"] is not None:
            lab = self.team_colours.get(assignment["team"])
            if lab is not None:
                return lab_to_bgr(lab)
        return ROLE_COLOURS.get(assignment["role"], UNKNOWN_COLOUR)
