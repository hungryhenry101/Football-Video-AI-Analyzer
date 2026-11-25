"""The annotator may only offer landmarks the template can actually solve."""

from core.pitchcal.annotate import RECOMMENDED
from core.pitchcal.geometry import PitchTemplate

# Intersections of two painted template lines.  A human can click the visible
# junction accurately; the visible extreme of a projected curve is *not* the
# projected extreme of the world curve, so curve extremes are a separate,
# weaker class.
INTERSECTION_LANDMARKS = {
    "left_goal_area_bottom_inner",
    "right_goal_area_bottom_inner",
    "left_penalty_top_goal_line",
    "right_penalty_top_goal_line",
    "left_penalty_bottom_goal_line",
    "right_penalty_bottom_goal_line",
    "left_goal_area_top_goal_line",
    "right_goal_area_top_goal_line",
    "left_goal_area_bottom_goal_line",
    "right_goal_area_bottom_goal_line",
    "left_penalty_arc_top",
    "right_penalty_arc_top",
    "left_penalty_arc_bottom",
    "right_penalty_arc_bottom",
}


def test_recommended_landmarks_exist_and_are_unique():
    template = PitchTemplate(105.0, 68.0)
    missing = [name for name in RECOMMENDED if name not in template.landmarks]
    assert not missing, f"annotator offers unknown landmarks: {missing}"
    assert len(RECOMMENDED) == len(set(RECOMMENDED))


def test_recommended_includes_the_line_intersections():
    assert INTERSECTION_LANDMARKS <= set(RECOMMENDED)
