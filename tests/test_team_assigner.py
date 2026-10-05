"""Jersey-colour clustering.

Synthetic frames, so these run everywhere — no model weights and no video.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.team_assigner import (
    GOALKEEPER,
    PLAYER,
    REFEREE,
    TEAM_A,
    TEAM_B,
    TeamAssigner,
)

# Kits, in BGR. Grass is what the segmentation mask is meant to exclude, not
# something filtered afterwards.
RED = np.array([40, 40, 200], np.uint8)
BLUE = np.array([200, 60, 40], np.uint8)
YELLOW = np.array([0, 220, 255], np.uint8)   # a third kit: keeper or referee
GREEN = np.array([110, 225, 130], np.uint8)  # the neon kit from test2.mp4
GRASS = np.array([50, 170, 60], np.uint8)
SHADOW = np.array([25, 25, 25], np.uint8)

# The pair chroma alone cannot separate: both are neutral, so their a and b are
# identical and only L differs. This is the referee-on-a-white-team case.
WHITE = np.array([232, 232, 232], np.uint8)
BLACK = np.array([32, 32, 32], np.uint8)

BOX_W, BOX_H = 40, 120
BOX_Y1 = 420
BOX_Y2 = BOX_Y1 + BOX_H

# A nadir camera 20 m above the centre spot, with a focal length that fits the
# 105 m pitch into 1920 px. World x = (u - 960) * 20 / 366, so the +x goal line
# (x = 52.5) sits at u = 1921 and the centre at u = 960.
CAM_K = np.array([[366.0, 0.0, 960.0], [0.0, 366.0, 540.0], [0.0, 0.0, 1.0]])
CAM_R = np.eye(3)
CAM_T = np.array([0.0, 0.0, -20.0])


def _lab(bgr):
    """The CIE-Lab triple of a single BGR colour, in OpenCV channel order."""
    lab = cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


def _frame(h=1080, w=1920):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = GRASS
    return frame


def _add_player(frame, tid, colour, x1, noise=6):
    """Paint a player onto the frame and return their tracker-style object.

    A little per-pixel noise, because a real torso is never one flat colour.
    """
    x2 = x1 + BOX_W
    patch = np.full((BOX_H, BOX_W, 3), colour, np.int16)
    rng = np.random.default_rng(tid)
    patch += rng.integers(-noise, noise + 1, patch.shape)
    frame[BOX_Y1:BOX_Y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)
    return {
        "id": tid,
        "bbox": np.array([x1, BOX_Y1, x2, BOX_Y2], np.float32),
        "cls": 0,
        "conf": 0.9,
        "mask": np.ones((BOX_H, BOX_W), np.uint8),
    }


def _squad(frame, third_kit_x1=1820):
    """Three reds, three blues and one yellow third kit, spread across the pitch.

    ``third_kit_x1`` defaults to the +x six-yard box (world x ~ +48 m); pass
    940 to stand the same player on the centre spot instead.
    """
    objs = []
    for i, x1 in enumerate((100, 180, 260)):
        objs.append(_add_player(frame, 10 + i, RED, x1))
    for i, x1 in enumerate((500, 580, 660)):
        objs.append(_add_player(frame, 20 + i, BLUE, x1))
    objs.append(_add_player(frame, 30, YELLOW, third_kit_x1))
    return objs


def _line_up(frame, kits, x_start=100, spacing=80):
    """One player per kit colour, ids 10, 11, 12... from left to right."""
    return [
        _add_player(frame, 10 + i, kit, x_start + i * spacing)
        for i, kit in enumerate(kits)
    ]


# ------------------------------------------------------------------ stage 1


def test_jersey_colour_reads_the_kit():
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)

    got = TeamAssigner().identify_jersey_colour(obj, frame)

    assert got is not None
    assert len(got) == 3, "the reading carries L as well as a and b"
    assert np.linalg.norm(got - _lab(RED)) < 8, "should land on the red kit"
    assert np.linalg.norm(got - _lab(RED)) < np.linalg.norm(got - _lab(BLUE))


def test_a_grass_leak_does_not_dominate_the_reading():
    """There is no hue filter any more; the reading survives a leak because the
    shirt is still the larger of the two clusters."""
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)

    # Grass inside the torso band, as a leaky mask would produce.
    frame[BOX_Y1 + 30 : BOX_Y1 + 60, 100:116] = GRASS

    got = TeamAssigner().identify_jersey_colour(obj, frame)

    assert got is not None
    assert np.linalg.norm(got - _lab(RED)) < 8, "grass pulled the colour off the kit"


def test_a_shadowed_edge_does_not_dominate_the_reading():
    """A torso with one edge in shadow must still read as the shirt."""
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    frame[BOX_Y1 + 24 : BOX_Y1 + 66, 100:113] = SHADOW  # a strip of the band

    got = TeamAssigner().identify_jersey_colour(obj, frame)

    assert got is not None
    assert np.linalg.norm(got - _lab(RED)) < 10, "the shadowed strip took over"


def test_missing_mask_is_skipped():
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    obj["mask"] = None
    assert TeamAssigner().identify_jersey_colour(obj, frame) is None


def test_a_distant_player_is_too_small_to_read():
    """A handful of pixels is noise, not a kit colour."""
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    obj["bbox"] = np.array([100, 420, 106, 436], np.float32)   # 6 x 16 px
    obj["mask"] = np.ones((16, 6), np.uint8)
    assert TeamAssigner().identify_jersey_colour(obj, frame) is None


def test_torso_lab_returns_the_pixels_the_reading_is_taken_from():
    """The distribution and the mean behind it must be the same pixels."""
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    assigner = TeamAssigner()

    lab = assigner.torso_lab(obj, frame)
    reading = assigner.identify_jersey_colour(obj, frame)

    assert lab is not None
    assert lab.ndim == 2 and lab.shape[1] == 3, "one row per pixel"
    assert np.linalg.norm(lab.mean(axis=0) - _lab(RED)) < 8
    assert np.linalg.norm(reading - _lab(RED)) < 8, "a lookalike set, not the same one"


def test_torso_lab_keeps_the_head_and_the_legs_out():
    """Only the band between TORSO_TOP and TORSO_BOTTOM is sampled, so a black
    head or a black pair of shorts cannot vote on the kit."""
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    frame[BOX_Y1 : BOX_Y1 + 18, 100:140] = BLACK        # head and hair
    frame[BOX_Y1 + 100 : BOX_Y2, 100:140] = BLACK       # shorts and socks

    assigner = TeamAssigner()
    lab = assigner.torso_lab(obj, frame)

    assert lab is not None
    assert np.linalg.norm(lab.mean(axis=0) - _lab(RED)) < 8
    assert np.linalg.norm(assigner.identify_jersey_colour(obj, frame) - _lab(RED)) < 8


def test_torso_lab_refuses_exactly_what_the_reading_refuses():
    frame = _frame()
    obj = _add_player(frame, 1, RED, 100)
    obj["mask"] = None
    assigner = TeamAssigner()

    assert assigner.torso_lab(obj, frame) is None
    assert assigner.identify_jersey_colour(obj, frame) is None


def test_uniform_colour_does_not_break_the_split():
    """k-means with k=2 over identical pixels leaves a cluster empty."""
    assigner = TeamAssigner()
    lab = np.tile(_lab(RED), (500, 1)).astype(np.float32)
    got = assigner._dominant_lab(lab)
    assert np.linalg.norm(got - _lab(RED)) < 1


# ------------------------------------------------------------------ stage 2


def test_cluster_colours_splits_two_kits():
    assigner = TeamAssigner()
    colours = assigner._lab_cv2kmeans(np.stack([_lab(RED)] * 3 + [_lab(BLUE)] * 3))
    labels, centres = assigner.cluster_colours(colours)

    assert len(centres) == 2
    assert len(set(labels[:3])) == 1, "the three reds must share a cluster"
    assert len(set(labels[3:])) == 1, "the three blues must share a cluster"
    assert labels[0] != labels[3]


def test_cluster_colours_with_fewer_players_than_clusters():
    assigner = TeamAssigner()
    labels, centres = assigner.cluster_colours(assigner._lab_cv2kmeans(np.stack([_lab(RED)])), k=2)
    assert list(labels) == [0]
    assert len(centres) == 1


def test_labels_do_not_depend_on_the_order_players_arrive():
    """Same kits, reversed detection order, same team numbers."""
    frame = _frame()
    first = TeamAssigner().update(frame, _squad(frame))
    reversed_objs = list(reversed(_squad(frame)))
    second = TeamAssigner().update(frame, reversed_objs)

    for tid in first:
        assert first[tid]["team"] == second[tid]["team"]


# --------------------------------------------------------------- per frame


def test_update_assigns_two_teams():
    frame = _frame()
    assignments = TeamAssigner().update(frame, _squad(frame))

    reds = {assignments[t]["team"] for t in (10, 11, 12)}
    blues = {assignments[t]["team"] for t in (20, 21, 22)}

    assert len(reds) == 1 and len(blues) == 1, "a team must not be split"
    assert reds != blues
    assert reds.pop() in (TEAM_A, TEAM_B)
    assert all(assignments[t]["role"] == PLAYER for t in (10, 20))


def test_black_and_white_kits_are_separated():
    """The case chroma alone cannot do.

    Both kits are neutral, so their a and b are identical and only L tells them
    apart — which is why L is in the clustering at all.
    """
    assert np.allclose(_lab(WHITE)[1:], _lab(BLACK)[1:], atol=1), (
        "premise of this test: these two kits differ only in lightness"
    )

    frame = _frame()
    objs = _line_up(frame, [WHITE] * 3 + [BLACK] * 3)
    assignments = TeamAssigner().update(frame, objs)

    whites = {assignments[t]["team"] for t in (10, 11, 12)}
    blacks = {assignments[t]["team"] for t in (13, 14, 15)}

    assert len(whites) == 1 and len(blacks) == 1, "a team must not be split"
    assert whites != blacks, "white and black collapsed into one team"


def test_a_black_referee_is_not_folded_into_the_white_team():
    """A black official against a white team: same chroma, so before L was in
    the clustering he was indistinguishable from the team he officiates."""
    frame = _frame()
    objs = _line_up(frame, [WHITE] * 3 + [GREEN] * 3 + [BLACK])
    assignments = TeamAssigner().update(frame, objs)

    assert assignments[16]["team"] is None, "the official joined a team"
    assert assignments[16]["role"] == REFEREE


def test_teams_are_drawn_in_their_own_jersey_colour():
    """The overlay carries the kit that is actually on the pitch."""
    frame = _frame()
    assigner = TeamAssigner()
    assigner.update(frame, _squad(frame))     # reds at 10-12, blues at 20-22

    red = np.array(assigner.draw_colour(10))
    blue = np.array(assigner.draw_colour(20))

    assert np.linalg.norm(red - RED.astype(int)) < 25, f"red team drawn as {red}"
    assert np.linalg.norm(blue - BLUE.astype(int)) < 25, f"blue team drawn as {blue}"


def test_a_third_kit_is_an_outlier_not_a_team_member():
    frame = _frame()
    assignments = TeamAssigner().update(frame, _squad(frame))

    assert assignments[30]["team"] is None
    assert assignments[30]["role"] == REFEREE


def test_two_players_alone_are_not_flagged_as_outliers():
    """The median-based threshold must not fire on a small, uniform set."""
    frame = _frame()
    objs = [_add_player(frame, 1, RED, 100), _add_player(frame, 2, RED, 180)]
    assignments = TeamAssigner().update(frame, objs)

    assert assignments[1]["role"] == PLAYER
    assert assignments[2]["role"] == PLAYER


def test_goalkeeper_is_told_from_the_referee_by_position():
    frame = _frame()
    objs = _squad(frame)                       # yellow at x ~ +48 m, in the box
    assignments = TeamAssigner().update(frame, objs, CAM_K, CAM_R, CAM_T)

    assert assignments[30]["role"] == GOALKEEPER


def test_a_third_kit_in_the_middle_is_the_referee():
    frame = _frame()
    objs = _squad(frame, third_kit_x1=940)     # the same yellow, on the centre spot

    assignments = TeamAssigner().update(frame, objs, CAM_K, CAM_R, CAM_T)

    assert assignments[30]["role"] == REFEREE


@pytest.mark.parametrize(
    "position, expected",
    [
        ((50.0, 0.0), GOALKEEPER),      # in the six-yard box
        ((-50.0, 3.0), GOALKEEPER),     # the other end
        ((50.0, 20.0), REFEREE),        # wide of the goal area
        ((0.0, 0.0), REFEREE),          # centre of the pitch
        ((30.0, 0.0), REFEREE),         # upfield
        (None, REFEREE),                # no calibration available
    ],
)
def test_outlier_role_rule(position, expected):
    assert TeamAssigner._outlier_role(position) == expected


def test_colours_accumulate_so_an_occluded_frame_keeps_its_labels():
    assigner = TeamAssigner()
    frame = _frame()
    objs = _squad(frame)
    first = assigner.update(frame, objs)

    # Next frame: everyone but one player is off camera. The frame must be
    # repainted — an empty frame with the old bboxes still in it would feed
    # grass into their running means, which is a mis-detection, not an occlusion.
    second_frame = _frame()
    second = assigner.update(second_frame, [_add_player(second_frame, 10, RED, 100)])

    assert set(second) == set(first), "known players must survive a missing frame"
    for tid in first:
        assert first[tid]["team"] == second[tid]["team"]


def test_no_players_is_not_an_error():
    assert TeamAssigner().update(_frame(), []) == {}


def test_draw_colour_follows_the_assignment():
    frame = _frame()
    assigner = TeamAssigner()
    assignments = assigner.update(frame, _squad(frame))

    assert assigner.draw_colour(10) != assigner.draw_colour(20)
    assert assigner.draw_colour(30) != assigner.draw_colour(10)
    assert assigner.draw_colour(999) == (128, 128, 128)   # never seen
