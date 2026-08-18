// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
// 
// This is a license agreement ("Agreement") between you (called "Licensee" or "You" in this Agreement) and EVS Broadcast Equipment SA. (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
// 
// RESERVATION OF OWNERSHIP AND GRANT OF LICENSE:
// Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement.  As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
// CONFIDENTIALITY: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials in confidence and use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication.
// COPYRIGHT: The Software is owned by Licensor and is protected by copyright laws and applicable international treaties and/or conventions.
// PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
// DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
// BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
// USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement and may not use the name or mark "EVS" or any renditions thereof without the prior written permission of Licensor.
// You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
// ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
// TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
// The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
// FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
// DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
// SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement. 
// EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
// EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs.
// SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
// NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
// GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Belgium without reference to conflict of laws principles.  You consent to the exclusive jurisdiction of the courts of Liège.
// ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.


#pragma once

#include "core.h"

#include <opencv2/opencv.hpp>


#include <memory>
#include <set>
#include <array>
#include <map>
#include <fstream>

class SoccerPitch3D;

class SoccerPitch3D
{
public:
    static constexpr double MINIMUM_LENGTH = 90.0;
    static constexpr double MAXIMUM_LENGTH = 120.0;
    static constexpr double DEFAULT_LENGTH = 105.0;
    static constexpr double MINIMUM_WIDTH = 45.0;
    static constexpr double MAXIMUM_WIDTH = 90.0;
    static constexpr double DEFAULT_WIDTH = 68.0;
    static constexpr double CENTER_CIRCLE_RADIUS = 9.15;
    static constexpr double PENALTY_CIRCLE_RADIUS = 9.15;
    static constexpr double CORNER_CIRCLE_RADIUS = 1.0;
    static constexpr double GOAL_LINE_TO_PENALTY_MARK = 11.0;
    static constexpr double GOALPOST_TO_GOALPOST = 7.32;
    static constexpr double GROUND_TO_CROSSBAR = 2.44;
    static constexpr double PENALTY_AREA_WIDTH = 40.32;
    static constexpr double PENALTY_AREA_LENGTH = 16.5;
    static constexpr double GOAL_AREA_WIDTH = 18.32;
    static constexpr double GOAL_AREA_LENGTH = 5.5;
    static constexpr double MAX_LINE_WIDTH = 0.12;

    enum PointID
    {
        UNDEFINED_POINT = 0,
        CENTER_MARK = 101,
        L_PENALTY_MARK = 102,
        R_PENALTY_MARK = 103,
        TL_PITCH_CORNER = 104,
        TR_PITCH_CORNER = 105,
        BL_PITCH_CORNER = 106,
        BR_PITCH_CORNER = 107,
        L_PENALTY_AREA_TL_CORNER = 108,
        L_PENALTY_AREA_TR_CORNER = 109,
        L_PENALTY_AREA_BL_CORNER = 110,
        L_PENALTY_AREA_BR_CORNER = 111,
        R_PENALTY_AREA_TL_CORNER = 112,
        R_PENALTY_AREA_TR_CORNER = 113,
        R_PENALTY_AREA_BL_CORNER = 114,
        R_PENALTY_AREA_BR_CORNER = 115,
        L_GOAL_AREA_TL_CORNER = 116,
        L_GOAL_AREA_TR_CORNER = 117,
        L_GOAL_AREA_BL_CORNER = 118,
        L_GOAL_AREA_BR_CORNER = 119,
        R_GOAL_AREA_TL_CORNER = 120,
        R_GOAL_AREA_TR_CORNER = 121,
        R_GOAL_AREA_BL_CORNER = 122,
        R_GOAL_AREA_BR_CORNER = 123,
        T_TOUCH_AND_HALFWAY_LINES_INTERSECTION = 124,
        B_TOUCH_AND_HALFWAY_LINES_INTERSECTION = 125,
        T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION = 126,
        B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION = 127,
        TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION = 128,
        TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION = 129,
        BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION = 130,
        BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION = 131,
        L1 = TL_PITCH_CORNER,
        L2 = L_PENALTY_AREA_TL_CORNER,
        L3 = L_GOAL_AREA_TL_CORNER,
        L5 = L_GOAL_AREA_BL_CORNER,
        L6 = L_PENALTY_AREA_BL_CORNER,
        L7 = BL_PITCH_CORNER,
        L8 = L_GOAL_AREA_TR_CORNER,
        L9 = L_GOAL_AREA_BR_CORNER,
        L10 = L_PENALTY_MARK,
        L11 = L_PENALTY_AREA_TR_CORNER,
        L12 = TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        L13 = BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        L14 = L_PENALTY_AREA_BR_CORNER,
        C1 = T_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        C3 = T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION,
        C0 = CENTER_MARK,
        C4 = B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION,
        C6 = B_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        R1 = TR_PITCH_CORNER,
        R2 = R_PENALTY_AREA_TR_CORNER,
        R3 = R_GOAL_AREA_TR_CORNER,
        R5 = R_GOAL_AREA_BR_CORNER,
        R6 = R_PENALTY_AREA_BR_CORNER,
        R7 = BR_PITCH_CORNER,
        R8 = R_GOAL_AREA_TL_CORNER,
        R9 = R_GOAL_AREA_BL_CORNER,
        R10 = R_PENALTY_MARK,
        R11 = R_PENALTY_AREA_TL_CORNER,
        R12 = TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        R13 = BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        R14 = R_PENALTY_AREA_BL_CORNER,
    };

    enum LineID
    {
        UNDEFINED_LINE = 150,
        T_TOUCH_LINE = 151,
        B_TOUCH_LINE = 152,
        L_GOAL_LINE = 153,
        HALFWAY_LINE = 154,
        R_GOAL_LINE = 155,
        L_PENALTY_AREA_T_SIDE = 156,
        L_PENALTY_AREA_B_SIDE = 157,
        L_PENALTY_AREA_R_SIDE = 158,
        R_PENALTY_AREA_T_SIDE = 159,
        R_PENALTY_AREA_B_SIDE = 160,
        R_PENALTY_AREA_L_SIDE = 161,
        L_GOAL_AREA_T_SIDE = 162,
        L_GOAL_AREA_B_SIDE = 163,
        L_GOAL_AREA_R_SIDE = 164,
        R_GOAL_AREA_T_SIDE = 165,
        R_GOAL_AREA_B_SIDE = 166,
        R_GOAL_AREA_L_SIDE = 167,
        L_PENALTY_ARC = 168,
        CENTER_CIRCLE = 169,
        R_PENALTY_ARC = 170,
        TL_CORNER_ARC = 171,
        TR_CORNER_ARC = 172,
        BL_CORNER_ARC = 173,
        BR_CORNER_ARC = 174
    };

    static std::map<PointID, std::string> POINT_ID_DESCRIPTIONS;

    static std::map<LineID, std::string> LINE_ID_DESCRIPTIONS;

    static std::vector<PointID> DEFINED_LANDMARK_POINTS;

    static std::vector<LineID> DEFINED_LANDMARK_LINES;

    static std::vector<PointID> BOUNDING_LANDMARK_POINTS;

    static std::map<LineID, std::vector<PointID>> POINTS_ON_LINE;

    static PointID pointIDCast(int integer);

    static LineID fromPointIDPair(PointID pointID1, PointID pointID2);

    static PointID symmetricPointID(PointID pointID);

    static PointID centralSymmetricPointID(PointID pointID);

    static LineID symmetricLineID(LineID lineID);

    static LineID centralSymmetricLineID(LineID lineID);

    static bool isStraightLine(LineID lineID);

    static bool isCircularCurve(LineID lineID);

    explicit SoccerPitch3D(double length = DEFAULT_LENGTH, double width = DEFAULT_WIDTH);

    bool setLength(double length);

    bool setWidth(double width);

    double getLength() const;

    double getWidth() const;

    Point2D getPoint2D(PointID pointID) const;

    LineSegment2D getLineSegment2D(LineID lineID) const;

    Line2D getLine2D(LineID lineID) const;

    Circle2D getCircle2D(LineID lineID) const;

    CircleArc2D getCircleArc2D(LineID lineID) const;

    Point3D getPoint3D(PointID pointID) const;

    Point3D getPoint3D(const Point2D &point2D) const;

    Plane3D getSurface() const;

    Polyline3D getPolyline3D(LineID lineID, double delta) const;

    Polyline3D getPolyline3D(const std::vector<Point2D> &points2D, double delta) const;

    Polyline3D getPolyline3D(const LineSegment2D &lineSegment2D, double delta) const;

    Polyline3D getPolyline3D(const CircleArc2D &circleArc2D, double delta) const;

    Polyline2D getPolyline2D(LineID lineID, double delta) const;

    Polyline2D getPolyline2D(const LineSegment2D &lineSegment2D, double delta) const;

    Polyline2D getPolyline2D(const CircleArc2D &circleArc2D, double delta) const;

    std::vector<Polyline3D> getWireframe(double delta) const;

    std::vector<Polyline2D> getWireframe2D(double delta) const;

private:
    Point2D getRegularPoint2D(PointID pointID) const;

    double _regularLength;
    double _regularWidth;
};
