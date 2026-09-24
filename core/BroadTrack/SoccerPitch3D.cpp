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


#include "SoccerPitch3D.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <iterator>
#include <limits>
#include <algorithm>

using namespace std;
using namespace cv;

constexpr double SoccerPitch3D::MINIMUM_LENGTH;
constexpr double SoccerPitch3D::MAXIMUM_LENGTH;
constexpr double SoccerPitch3D::DEFAULT_LENGTH;
constexpr double SoccerPitch3D::MINIMUM_WIDTH;
constexpr double SoccerPitch3D::MAXIMUM_WIDTH;
constexpr double SoccerPitch3D::DEFAULT_WIDTH;
constexpr double SoccerPitch3D::CENTER_CIRCLE_RADIUS;
constexpr double SoccerPitch3D::PENALTY_CIRCLE_RADIUS;
constexpr double SoccerPitch3D::CORNER_CIRCLE_RADIUS;
constexpr double SoccerPitch3D::GOAL_LINE_TO_PENALTY_MARK;
constexpr double SoccerPitch3D::GOALPOST_TO_GOALPOST;
constexpr double SoccerPitch3D::GROUND_TO_CROSSBAR;
constexpr double SoccerPitch3D::PENALTY_AREA_WIDTH;
constexpr double SoccerPitch3D::PENALTY_AREA_LENGTH;
constexpr double SoccerPitch3D::GOAL_AREA_WIDTH;
constexpr double SoccerPitch3D::GOAL_AREA_LENGTH;
constexpr double SoccerPitch3D::MAX_LINE_WIDTH;

map<SoccerPitch3D::PointID, string> SoccerPitch3D::POINT_ID_DESCRIPTIONS =
    {
        {UNDEFINED_POINT, "UNDEFINED_POINT"},
        {CENTER_MARK, "CENTER_MARK"},
        {L_PENALTY_MARK, "L_PENALTY_MARK"},
        {R_PENALTY_MARK, "R_PENALTY_MARK"},
        {TL_PITCH_CORNER, "TL_PITCH_CORNER"},
        {TR_PITCH_CORNER, "TR_PITCH_CORNER"},
        {BL_PITCH_CORNER, "BL_PITCH_CORNER"},
        {BR_PITCH_CORNER, "BR_PITCH_CORNER"},
        {L_PENALTY_AREA_TL_CORNER, "L_PENALTY_AREA_TL_CORNER"},
        {L_PENALTY_AREA_TR_CORNER, "L_PENALTY_AREA_TR_CORNER"},
        {L_PENALTY_AREA_BL_CORNER, "L_PENALTY_AREA_BL_CORNER"},
        {L_PENALTY_AREA_BR_CORNER, "L_PENALTY_AREA_BR_CORNER"},
        {R_PENALTY_AREA_TL_CORNER, "R_PENALTY_AREA_TL_CORNER"},
        {R_PENALTY_AREA_TR_CORNER, "R_PENALTY_AREA_TR_CORNER"},
        {R_PENALTY_AREA_BL_CORNER, "R_PENALTY_AREA_BL_CORNER"},
        {R_PENALTY_AREA_BR_CORNER, "R_PENALTY_AREA_BR_CORNER"},
        {L_GOAL_AREA_TL_CORNER, "L_GOAL_AREA_TL_CORNER"},
        {L_GOAL_AREA_TR_CORNER, "L_GOAL_AREA_TR_CORNER"},
        {L_GOAL_AREA_BL_CORNER, "L_GOAL_AREA_BL_CORNER"},
        {L_GOAL_AREA_BR_CORNER, "L_GOAL_AREA_BR_CORNER"},
        {R_GOAL_AREA_TL_CORNER, "R_GOAL_AREA_TL_CORNER"},
        {R_GOAL_AREA_TR_CORNER, "R_GOAL_AREA_TR_CORNER"},
        {R_GOAL_AREA_BL_CORNER, "R_GOAL_AREA_BL_CORNER"},
        {R_GOAL_AREA_BR_CORNER, "R_GOAL_AREA_BR_CORNER"},
        {T_TOUCH_AND_HALFWAY_LINES_INTERSECTION, "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"},
        {B_TOUCH_AND_HALFWAY_LINES_INTERSECTION, "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"},
        {T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION, "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"},
        {B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION, "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"},
        {TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION, "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"},
        {TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION, "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"},
        {BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION, "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"},
        {BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION, "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"}};

map<SoccerPitch3D::LineID, string> SoccerPitch3D::LINE_ID_DESCRIPTIONS =
    {
        {UNDEFINED_LINE, "UNDEFINED_LINE"},
        {T_TOUCH_LINE, "T_TOUCH_LINE"},
        {B_TOUCH_LINE, "B_TOUCH_LINE"},
        {L_GOAL_LINE, "L_GOAL_LINE"},
        {HALFWAY_LINE, "HALFWAY_LINE"},
        {R_GOAL_LINE, "R_GOAL_LINE"},
        {L_PENALTY_AREA_T_SIDE, "L_PENALTY_AREA_T_SIDE"},
        {L_PENALTY_AREA_B_SIDE, "L_PENALTY_AREA_B_SIDE"},
        {L_PENALTY_AREA_R_SIDE, "L_PENALTY_AREA_R_SIDE"},
        {R_PENALTY_AREA_T_SIDE, "R_PENALTY_AREA_T_SIDE"},
        {R_PENALTY_AREA_B_SIDE, "R_PENALTY_AREA_B_SIDE"},
        {R_PENALTY_AREA_L_SIDE, "R_PENALTY_AREA_L_SIDE"},
        {L_GOAL_AREA_T_SIDE, "L_GOAL_AREA_T_SIDE"},
        {L_GOAL_AREA_B_SIDE, "L_GOAL_AREA_B_SIDE"},
        {L_GOAL_AREA_R_SIDE, "L_GOAL_AREA_R_SIDE"},
        {R_GOAL_AREA_T_SIDE, "R_GOAL_AREA_T_SIDE"},
        {R_GOAL_AREA_B_SIDE, "R_GOAL_AREA_B_SIDE"},
        {R_GOAL_AREA_L_SIDE, "R_GOAL_AREA_L_SIDE"},
        {L_PENALTY_ARC, "L_PENALTY_ARC"},
        {CENTER_CIRCLE, "CENTER_CIRCLE"},
        {R_PENALTY_ARC, "R_PENALTY_ARC"},
        {TL_CORNER_ARC, "TL_CORNER_ARC"},
        {TR_CORNER_ARC, "TR_CORNER_ARC"},
        {BL_CORNER_ARC, "BL_CORNER_ARC"},
        {BR_CORNER_ARC, "BR_CORNER_ARC"}};

vector<SoccerPitch3D::PointID> SoccerPitch3D::DEFINED_LANDMARK_POINTS =
    {
        SoccerPitch3D::CENTER_MARK,
        SoccerPitch3D::L_PENALTY_MARK,
        SoccerPitch3D::R_PENALTY_MARK,
        SoccerPitch3D::TL_PITCH_CORNER,
        SoccerPitch3D::TR_PITCH_CORNER,
        SoccerPitch3D::BL_PITCH_CORNER,
        SoccerPitch3D::BR_PITCH_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_TL_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_TR_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_BL_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_BR_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_TL_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_TR_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_BL_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_BR_CORNER,
        SoccerPitch3D::L_GOAL_AREA_TL_CORNER,
        SoccerPitch3D::L_GOAL_AREA_TR_CORNER,
        SoccerPitch3D::L_GOAL_AREA_BL_CORNER,
        SoccerPitch3D::L_GOAL_AREA_BR_CORNER,
        SoccerPitch3D::R_GOAL_AREA_TL_CORNER,
        SoccerPitch3D::R_GOAL_AREA_TR_CORNER,
        SoccerPitch3D::R_GOAL_AREA_BL_CORNER,
        SoccerPitch3D::R_GOAL_AREA_BR_CORNER,
        SoccerPitch3D::T_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        SoccerPitch3D::B_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        SoccerPitch3D::T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION,
        SoccerPitch3D::B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION,
        SoccerPitch3D::TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        SoccerPitch3D::TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        SoccerPitch3D::BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
        SoccerPitch3D::BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION,
};

vector<SoccerPitch3D::LineID> SoccerPitch3D::DEFINED_LANDMARK_LINES =
    {
        SoccerPitch3D::T_TOUCH_LINE,
        SoccerPitch3D::B_TOUCH_LINE,
        SoccerPitch3D::L_GOAL_LINE,
        SoccerPitch3D::HALFWAY_LINE,
        SoccerPitch3D::R_GOAL_LINE,
        SoccerPitch3D::L_PENALTY_AREA_T_SIDE,
        SoccerPitch3D::L_PENALTY_AREA_B_SIDE,
        SoccerPitch3D::L_PENALTY_AREA_R_SIDE,
        SoccerPitch3D::R_PENALTY_AREA_T_SIDE,
        SoccerPitch3D::R_PENALTY_AREA_B_SIDE,
        SoccerPitch3D::R_PENALTY_AREA_L_SIDE,
        SoccerPitch3D::L_GOAL_AREA_T_SIDE,
        SoccerPitch3D::L_GOAL_AREA_B_SIDE,
        SoccerPitch3D::L_GOAL_AREA_R_SIDE,
        SoccerPitch3D::R_GOAL_AREA_T_SIDE,
        SoccerPitch3D::R_GOAL_AREA_B_SIDE,
        SoccerPitch3D::R_GOAL_AREA_L_SIDE,
        SoccerPitch3D::L_PENALTY_ARC,
        SoccerPitch3D::CENTER_CIRCLE,
        SoccerPitch3D::R_PENALTY_ARC,
        SoccerPitch3D::TL_CORNER_ARC,
        SoccerPitch3D::TR_CORNER_ARC,
        SoccerPitch3D::BL_CORNER_ARC,
        SoccerPitch3D::BR_CORNER_ARC};

vector<SoccerPitch3D::PointID> SoccerPitch3D::BOUNDING_LANDMARK_POINTS =
    {
        SoccerPitch3D::TL_PITCH_CORNER,
        SoccerPitch3D::T_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        SoccerPitch3D::TR_PITCH_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_TR_CORNER,
        SoccerPitch3D::R_GOAL_AREA_TR_CORNER,
        SoccerPitch3D::R_GOAL_AREA_BR_CORNER,
        SoccerPitch3D::R_PENALTY_AREA_BR_CORNER,
        SoccerPitch3D::BR_PITCH_CORNER,
        SoccerPitch3D::B_TOUCH_AND_HALFWAY_LINES_INTERSECTION,
        SoccerPitch3D::BL_PITCH_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_BL_CORNER,
        SoccerPitch3D::L_GOAL_AREA_BL_CORNER,
        SoccerPitch3D::L_GOAL_AREA_TL_CORNER,
        SoccerPitch3D::L_PENALTY_AREA_TL_CORNER};

map<SoccerPitch3D::LineID, vector<SoccerPitch3D::PointID>> SoccerPitch3D::POINTS_ON_LINE =
    {
        {UNDEFINED_LINE, {}},
        {T_TOUCH_LINE, {L1, C1, R1}},
        {B_TOUCH_LINE, {L7, C6, R7}},
        {L_GOAL_LINE, {L1, L2, L3, L5, L6, L7}},
        {HALFWAY_LINE, {C1, C3, C0, C4, C6}},
        {R_GOAL_LINE, {R1, R2, R3, R5, R6, R7}},
        {L_PENALTY_AREA_T_SIDE, {L2, L11}},
        {L_PENALTY_AREA_B_SIDE, {L6, L14}},
        {L_PENALTY_AREA_R_SIDE, {L11, L12, L13, L14}},
        {R_PENALTY_AREA_T_SIDE, {R2, R11}},
        {R_PENALTY_AREA_B_SIDE, {R6, R14}},
        {R_PENALTY_AREA_L_SIDE, {R11, R12, R13, R14}},
        {L_GOAL_AREA_T_SIDE, {L3, L8}},
        {L_GOAL_AREA_B_SIDE, {L5, L9}},
        {L_GOAL_AREA_R_SIDE, {L8, L9}},
        {R_GOAL_AREA_T_SIDE, {R3, R8}},
        {R_GOAL_AREA_B_SIDE, {R5, R9}},
        {R_GOAL_AREA_L_SIDE, {R8, R9}},
        {L_PENALTY_ARC, {L12, L13}},
        {CENTER_CIRCLE, {C3, C4}},
        {R_PENALTY_ARC, {R12, R13}},
        {TL_CORNER_ARC, {}},
        {TR_CORNER_ARC, {}},
        {BL_CORNER_ARC, {}},
        {BR_CORNER_ARC, {}}};

SoccerPitch3D::PointID SoccerPitch3D::pointIDCast(int integer)
{
    auto pointID = UNDEFINED_POINT;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = CENTER_MARK;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_PENALTY_MARK;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_PENALTY_MARK;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = TL_PITCH_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = TR_PITCH_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = BL_PITCH_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = BR_PITCH_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_PENALTY_AREA_TL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_PENALTY_AREA_TR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_PENALTY_AREA_BL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_PENALTY_AREA_BR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_PENALTY_AREA_TL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_PENALTY_AREA_TR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_PENALTY_AREA_BL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_PENALTY_AREA_BR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_GOAL_AREA_TL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_GOAL_AREA_TR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_GOAL_AREA_BL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = L_GOAL_AREA_BR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_GOAL_AREA_TL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_GOAL_AREA_TR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_GOAL_AREA_BL_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = R_GOAL_AREA_BR_CORNER;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = T_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = B_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }
    pointID = BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    if (integer == static_cast<int>(pointID))
    {
        return pointID;
    }

    return UNDEFINED_POINT;
}

SoccerPitch3D::LineID SoccerPitch3D::fromPointIDPair(SoccerPitch3D::PointID pointID1, SoccerPitch3D::PointID pointID2)
{
    vector<PointID> pointIDPair = {pointID1, pointID2};
    if (count(pointIDPair.begin(), pointIDPair.end(), TL_PITCH_CORNER) &&
        count(pointIDPair.begin(), pointIDPair.end(), TR_PITCH_CORNER))
    {
        return T_TOUCH_LINE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), BL_PITCH_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), BR_PITCH_CORNER))
    {
        return B_TOUCH_LINE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), TL_PITCH_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), BL_PITCH_CORNER))
    {
        return L_GOAL_LINE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), T_TOUCH_AND_HALFWAY_LINES_INTERSECTION) &&
             count(pointIDPair.begin(), pointIDPair.end(), B_TOUCH_AND_HALFWAY_LINES_INTERSECTION))
    {
        return HALFWAY_LINE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), TR_PITCH_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), BR_PITCH_CORNER))
    {
        return R_GOAL_LINE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_TR_CORNER))
    {
        return L_PENALTY_AREA_T_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_BL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_BR_CORNER))
    {
        return L_PENALTY_AREA_B_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_TR_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_PENALTY_AREA_BR_CORNER))
    {
        return L_PENALTY_AREA_R_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_TR_CORNER))
    {
        return R_PENALTY_AREA_T_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_BL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_BR_CORNER))
    {
        return R_PENALTY_AREA_B_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_PENALTY_AREA_BL_CORNER))
    {
        return R_PENALTY_AREA_L_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_TR_CORNER))
    {
        return L_GOAL_AREA_T_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_BL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_BR_CORNER))
    {
        return L_GOAL_AREA_B_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_TR_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), L_GOAL_AREA_BR_CORNER))
    {
        return L_GOAL_AREA_R_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_TR_CORNER))
    {
        return R_GOAL_AREA_T_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_BL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_BR_CORNER))
    {
        return R_GOAL_AREA_B_SIDE;
    }
    else if (count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_TL_CORNER) &&
             count(pointIDPair.begin(), pointIDPair.end(), R_GOAL_AREA_BL_CORNER))
    {
        return R_GOAL_AREA_L_SIDE;
    }

    return UNDEFINED_LINE;
}

SoccerPitch3D::PointID SoccerPitch3D::symmetricPointID(PointID pointID)
{
    switch (pointID)
    {
    case CENTER_MARK:
        return CENTER_MARK;
    case L_PENALTY_MARK:
        return R_PENALTY_MARK;
    case R_PENALTY_MARK:
        return L_PENALTY_MARK;
    case TL_PITCH_CORNER:
        return TR_PITCH_CORNER;
    case TR_PITCH_CORNER:
        return TL_PITCH_CORNER;
    case BL_PITCH_CORNER:
        return BR_PITCH_CORNER;
    case BR_PITCH_CORNER:
        return BL_PITCH_CORNER;
    case L_PENALTY_AREA_TL_CORNER:
        return R_PENALTY_AREA_TR_CORNER;
    case L_PENALTY_AREA_TR_CORNER:
        return R_PENALTY_AREA_TL_CORNER;
    case L_PENALTY_AREA_BL_CORNER:
        return R_PENALTY_AREA_BR_CORNER;
    case L_PENALTY_AREA_BR_CORNER:
        return R_PENALTY_AREA_BL_CORNER;
    case R_PENALTY_AREA_TL_CORNER:
        return L_PENALTY_AREA_TR_CORNER;
    case R_PENALTY_AREA_TR_CORNER:
        return L_PENALTY_AREA_TL_CORNER;
    case R_PENALTY_AREA_BL_CORNER:
        return L_PENALTY_AREA_BR_CORNER;
    case R_PENALTY_AREA_BR_CORNER:
        return L_PENALTY_AREA_BL_CORNER;
    case L_GOAL_AREA_TL_CORNER:
        return R_GOAL_AREA_TR_CORNER;
    case L_GOAL_AREA_TR_CORNER:
        return R_GOAL_AREA_TL_CORNER;
    case L_GOAL_AREA_BL_CORNER:
        return R_GOAL_AREA_BR_CORNER;
    case L_GOAL_AREA_BR_CORNER:
        return R_GOAL_AREA_BL_CORNER;
    case R_GOAL_AREA_TL_CORNER:
        return L_GOAL_AREA_TR_CORNER;
    case R_GOAL_AREA_TR_CORNER:
        return L_GOAL_AREA_TL_CORNER;
    case R_GOAL_AREA_BL_CORNER:
        return L_GOAL_AREA_BR_CORNER;
    case R_GOAL_AREA_BR_CORNER:
        return L_GOAL_AREA_BL_CORNER;
    case T_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
        return T_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    case B_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
        return B_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    case T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
        return T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    case B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
        return B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    case TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    default:
        return UNDEFINED_POINT;
    }
}

SoccerPitch3D::PointID SoccerPitch3D::centralSymmetricPointID(PointID pointID)
{
    switch (pointID)
    {
    case CENTER_MARK:
        return CENTER_MARK;
    case L_PENALTY_MARK:
        return R_PENALTY_MARK;
    case R_PENALTY_MARK:
        return L_PENALTY_MARK;
    case TL_PITCH_CORNER:
        return BR_PITCH_CORNER;
    case TR_PITCH_CORNER:
        return BL_PITCH_CORNER;
    case BL_PITCH_CORNER:
        return TR_PITCH_CORNER;
    case BR_PITCH_CORNER:
        return TL_PITCH_CORNER;
    case L_PENALTY_AREA_TL_CORNER:
        return R_PENALTY_AREA_BR_CORNER;
    case L_PENALTY_AREA_TR_CORNER:
        return R_PENALTY_AREA_BL_CORNER;
    case L_PENALTY_AREA_BL_CORNER:
        return R_PENALTY_AREA_TR_CORNER;
    case L_PENALTY_AREA_BR_CORNER:
        return R_PENALTY_AREA_TL_CORNER;
    case R_PENALTY_AREA_TL_CORNER:
        return L_PENALTY_AREA_BR_CORNER;
    case R_PENALTY_AREA_TR_CORNER:
        return L_PENALTY_AREA_BL_CORNER;
    case R_PENALTY_AREA_BL_CORNER:
        return L_PENALTY_AREA_TR_CORNER;
    case R_PENALTY_AREA_BR_CORNER:
        return L_PENALTY_AREA_TL_CORNER;
    case L_GOAL_AREA_TL_CORNER:
        return R_GOAL_AREA_BR_CORNER;
    case L_GOAL_AREA_TR_CORNER:
        return R_GOAL_AREA_BL_CORNER;
    case L_GOAL_AREA_BL_CORNER:
        return R_GOAL_AREA_TR_CORNER;
    case L_GOAL_AREA_BR_CORNER:
        return R_GOAL_AREA_TL_CORNER;
    case R_GOAL_AREA_TL_CORNER:
        return L_GOAL_AREA_BR_CORNER;
    case R_GOAL_AREA_TR_CORNER:
        return L_GOAL_AREA_BL_CORNER;
    case R_GOAL_AREA_BL_CORNER:
        return L_GOAL_AREA_TR_CORNER;
    case R_GOAL_AREA_BR_CORNER:
        return L_GOAL_AREA_TL_CORNER;
    case T_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
        return B_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    case B_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
        return T_TOUCH_AND_HALFWAY_LINES_INTERSECTION;
    case T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
        return B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    case B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
        return T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION;
    case TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    case BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
        return TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION;
    default:
        return UNDEFINED_POINT;
    }
}

SoccerPitch3D::LineID SoccerPitch3D::symmetricLineID(LineID lineID)
{
    switch (lineID)
    {
    case T_TOUCH_LINE:
        return T_TOUCH_LINE;
    case B_TOUCH_LINE:
        return B_TOUCH_LINE;
    case L_GOAL_LINE:
        return R_GOAL_LINE;
    case HALFWAY_LINE:
        return HALFWAY_LINE;
    case R_GOAL_LINE:
        return L_GOAL_LINE;
    case L_PENALTY_AREA_T_SIDE:
        return R_PENALTY_AREA_T_SIDE;
    case L_PENALTY_AREA_B_SIDE:
        return R_PENALTY_AREA_B_SIDE;
    case L_PENALTY_AREA_R_SIDE:
        return R_PENALTY_AREA_L_SIDE;
    case R_PENALTY_AREA_T_SIDE:
        return L_PENALTY_AREA_T_SIDE;
    case R_PENALTY_AREA_B_SIDE:
        return L_PENALTY_AREA_B_SIDE;
    case R_PENALTY_AREA_L_SIDE:
        return L_PENALTY_AREA_R_SIDE;
    case L_GOAL_AREA_T_SIDE:
        return R_GOAL_AREA_T_SIDE;
    case L_GOAL_AREA_B_SIDE:
        return R_GOAL_AREA_B_SIDE;
    case L_GOAL_AREA_R_SIDE:
        return R_GOAL_AREA_L_SIDE;
    case R_GOAL_AREA_T_SIDE:
        return L_GOAL_AREA_T_SIDE;
    case R_GOAL_AREA_B_SIDE:
        return L_GOAL_AREA_B_SIDE;
    case R_GOAL_AREA_L_SIDE:
        return L_GOAL_AREA_R_SIDE;
    case L_PENALTY_ARC:
        return R_PENALTY_ARC;
    case CENTER_CIRCLE:
        return CENTER_CIRCLE;
    case R_PENALTY_ARC:
        return L_PENALTY_ARC;
    case TL_CORNER_ARC:
        return TR_CORNER_ARC;
    case TR_CORNER_ARC:
        return TL_CORNER_ARC;
    case BL_CORNER_ARC:
        return BR_CORNER_ARC;
    case BR_CORNER_ARC:
        return BL_CORNER_ARC;
    default:
        return UNDEFINED_LINE;
    }
}

SoccerPitch3D::LineID SoccerPitch3D::centralSymmetricLineID(LineID lineID)
{
    switch (lineID)
    {
    case T_TOUCH_LINE:
        return B_TOUCH_LINE;
    case B_TOUCH_LINE:
        return T_TOUCH_LINE;
    case L_GOAL_LINE:
        return R_GOAL_LINE;
    case HALFWAY_LINE:
        return HALFWAY_LINE;
    case R_GOAL_LINE:
        return L_GOAL_LINE;
    case L_PENALTY_AREA_T_SIDE:
        return R_PENALTY_AREA_B_SIDE;
    case L_PENALTY_AREA_B_SIDE:
        return R_PENALTY_AREA_T_SIDE;
    case L_PENALTY_AREA_R_SIDE:
        return R_PENALTY_AREA_L_SIDE;
    case R_PENALTY_AREA_T_SIDE:
        return L_PENALTY_AREA_B_SIDE;
    case R_PENALTY_AREA_B_SIDE:
        return L_PENALTY_AREA_T_SIDE;
    case R_PENALTY_AREA_L_SIDE:
        return L_PENALTY_AREA_R_SIDE;
    case L_GOAL_AREA_T_SIDE:
        return R_GOAL_AREA_B_SIDE;
    case L_GOAL_AREA_B_SIDE:
        return R_GOAL_AREA_T_SIDE;
    case L_GOAL_AREA_R_SIDE:
        return R_GOAL_AREA_L_SIDE;
    case R_GOAL_AREA_T_SIDE:
        return L_GOAL_AREA_B_SIDE;
    case R_GOAL_AREA_B_SIDE:
        return L_GOAL_AREA_T_SIDE;
    case R_GOAL_AREA_L_SIDE:
        return L_GOAL_AREA_R_SIDE;
    case L_PENALTY_ARC:
        return R_PENALTY_ARC;
    case CENTER_CIRCLE:
        return CENTER_CIRCLE;
    case R_PENALTY_ARC:
        return L_PENALTY_ARC;
    case TL_CORNER_ARC:
        return BR_CORNER_ARC;
    case TR_CORNER_ARC:
        return BL_CORNER_ARC;
    case BL_CORNER_ARC:
        return TR_CORNER_ARC;
    case BR_CORNER_ARC:
        return TL_CORNER_ARC;
    default:
        return UNDEFINED_LINE;
    }
}

bool SoccerPitch3D::isStraightLine(LineID lineID)
{
    return !isCircularCurve(lineID);
}

bool SoccerPitch3D::isCircularCurve(LineID lineID)
{
    switch (lineID)
    {
    case CENTER_CIRCLE:
    case L_PENALTY_ARC:
    case R_PENALTY_ARC:
    case TL_CORNER_ARC:
    case TR_CORNER_ARC:
    case BL_CORNER_ARC:
    case BR_CORNER_ARC:
    {
        return true;
    }
    default:
    {
        return false;
    }
    }
}

SoccerPitch3D::SoccerPitch3D(double length, double width)
{
    _regularLength = length;
    _regularWidth = width;
}

bool SoccerPitch3D::setLength(double length)
{

    _regularLength = min(max(length, SoccerPitch3D::MINIMUM_LENGTH), SoccerPitch3D::MAXIMUM_LENGTH);

    return true;
}

bool SoccerPitch3D::setWidth(double width)
{

    _regularWidth = min(max(width, SoccerPitch3D::MINIMUM_WIDTH), SoccerPitch3D::MAXIMUM_WIDTH);

    return true;
}

double SoccerPitch3D::getLength() const
{

    return _regularLength;
}

double SoccerPitch3D::getWidth() const
{

    return _regularWidth;
}

Plane3D SoccerPitch3D::getSurface() const
{
    return Plane3D(Point3D(0., 0., 0.), Vector3x1(0., 0., 1.));
}

Point2D SoccerPitch3D::getRegularPoint2D(PointID pointID) const
{
    double x;
    double y;

    switch (pointID)
    {
    case CENTER_MARK:
    {
        x = 0.0;
        y = 0.0;
        break;
    }
    case L_PENALTY_MARK:
    {
        x = -_regularLength / 2.0 + GOAL_LINE_TO_PENALTY_MARK;
        y = 0.0;
        break;
    }
    case R_PENALTY_MARK:
    {
        x = _regularLength / 2.0 - GOAL_LINE_TO_PENALTY_MARK;
        y = 0.0;
        break;
    }
    case TL_PITCH_CORNER:
    {
        x = -_regularLength / 2.0;
        y = -_regularWidth / 2.0;
        break;
    }
    case TR_PITCH_CORNER:
    {
        x = _regularLength / 2.0;
        y = -_regularWidth / 2.0;
        break;
    }
    case BL_PITCH_CORNER:
    {
        x = -_regularLength / 2.0;
        y = _regularWidth / 2.0;
        break;
    }
    case BR_PITCH_CORNER:
    {
        x = _regularLength / 2.0;
        y = _regularWidth / 2.0;
        break;
    }
    case L_PENALTY_AREA_TL_CORNER:
    {
        x = -_regularLength / 2.0;
        y = -PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case L_PENALTY_AREA_TR_CORNER:
    {
        x = -_regularLength / 2.0 + PENALTY_AREA_LENGTH;
        y = -PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case L_PENALTY_AREA_BL_CORNER:
    {
        x = -_regularLength / 2.0;
        y = PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case L_PENALTY_AREA_BR_CORNER:
    {
        x = -_regularLength / 2.0 + PENALTY_AREA_LENGTH;
        y = PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case R_PENALTY_AREA_TL_CORNER:
    {
        x = _regularLength / 2.0 - PENALTY_AREA_LENGTH;
        y = -PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case R_PENALTY_AREA_TR_CORNER:
    {
        x = _regularLength / 2.0;
        y = -PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case R_PENALTY_AREA_BL_CORNER:
    {
        x = _regularLength / 2.0 - PENALTY_AREA_LENGTH;
        y = PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case R_PENALTY_AREA_BR_CORNER:
    {
        x = _regularLength / 2.0;
        y = PENALTY_AREA_WIDTH / 2.0;
        break;
    }
    case L_GOAL_AREA_TL_CORNER:
    {
        x = -_regularLength / 2.0;
        y = -GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case L_GOAL_AREA_TR_CORNER:
    {
        x = -_regularLength / 2.0 + GOAL_AREA_LENGTH;
        y = -GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case L_GOAL_AREA_BL_CORNER:
    {
        x = -_regularLength / 2.0;
        y = GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case L_GOAL_AREA_BR_CORNER:
    {
        x = -_regularLength / 2.0 + GOAL_AREA_LENGTH;
        y = GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case R_GOAL_AREA_TL_CORNER:
    {
        x = _regularLength / 2.0 - GOAL_AREA_LENGTH;
        y = -GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case R_GOAL_AREA_TR_CORNER:
    {
        x = _regularLength / 2.0;
        y = -GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case R_GOAL_AREA_BL_CORNER:
    {
        x = _regularLength / 2.0 - GOAL_AREA_LENGTH;
        y = GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case R_GOAL_AREA_BR_CORNER:
    {
        x = _regularLength / 2.0;
        y = GOAL_AREA_WIDTH / 2.0;
        break;
    }
    case T_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
    {
        x = 0.0;
        y = -_regularWidth / 2.0;
        break;
    }
    case B_TOUCH_AND_HALFWAY_LINES_INTERSECTION:
    {
        x = 0.0;
        y = _regularWidth / 2.0;
        break;
    }
    case T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
    {
        x = 0.0;
        y = -CENTER_CIRCLE_RADIUS;
        break;
    }
    case B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION:
    {
        x = 0.0;
        y = CENTER_CIRCLE_RADIUS;
        break;
    }
    case TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
    {
        x = -_regularLength / 2.0 + PENALTY_AREA_LENGTH;
        double dx = PENALTY_AREA_LENGTH - GOAL_LINE_TO_PENALTY_MARK;
        y = -sqrt(PENALTY_CIRCLE_RADIUS * PENALTY_CIRCLE_RADIUS - dx * dx);
        break;
    }
    case TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
    {
        x = _regularLength / 2.0 - PENALTY_AREA_LENGTH;
        double dx = PENALTY_AREA_LENGTH - GOAL_LINE_TO_PENALTY_MARK;
        y = -sqrt(PENALTY_CIRCLE_RADIUS * PENALTY_CIRCLE_RADIUS - dx * dx);
        break;
    }
    case BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
    {
        x = -_regularLength / 2.0 + PENALTY_AREA_LENGTH;
        double dx = PENALTY_AREA_LENGTH - GOAL_LINE_TO_PENALTY_MARK;
        y = sqrt(PENALTY_CIRCLE_RADIUS * PENALTY_CIRCLE_RADIUS - dx * dx);
        break;
    }
    case BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION:
    {
        x = _regularLength / 2.0 - PENALTY_AREA_LENGTH;
        double dx = PENALTY_AREA_LENGTH - GOAL_LINE_TO_PENALTY_MARK;
        y = sqrt(PENALTY_CIRCLE_RADIUS * PENALTY_CIRCLE_RADIUS - dx * dx);
        break;
    }
    default:
    {
        x = numeric_limits<double>::signaling_NaN();
        y = numeric_limits<double>::signaling_NaN();
        break;
    }
    }

    return Point2D(x, y);
}

Point2D SoccerPitch3D::getPoint2D(PointID pointID) const
{

    return getRegularPoint2D(pointID);
}

LineSegment2D SoccerPitch3D::getLineSegment2D(LineID lineID) const
{
    Point2D p1;
    Point2D p2;

    switch (lineID)
    {
    case T_TOUCH_LINE:
    {
        p1 = getPoint2D(TL_PITCH_CORNER);
        p2 = getPoint2D(TR_PITCH_CORNER);
        break;
    }
    case B_TOUCH_LINE:
    {
        p1 = getPoint2D(BL_PITCH_CORNER);
        p2 = getPoint2D(BR_PITCH_CORNER);
        break;
    }
    case L_GOAL_LINE:
    {
        p1 = getPoint2D(TL_PITCH_CORNER);
        p2 = getPoint2D(BL_PITCH_CORNER);
        break;
    }
    case HALFWAY_LINE:
    {
        p1 = getPoint2D(T_TOUCH_AND_HALFWAY_LINES_INTERSECTION);
        p2 = getPoint2D(B_TOUCH_AND_HALFWAY_LINES_INTERSECTION);
        break;
    }
    case R_GOAL_LINE:
    {
        p1 = getPoint2D(TR_PITCH_CORNER);
        p2 = getPoint2D(BR_PITCH_CORNER);
        break;
    }
    case L_PENALTY_AREA_T_SIDE:
    {
        p1 = getPoint2D(L_PENALTY_AREA_TL_CORNER);
        p2 = getPoint2D(L_PENALTY_AREA_TR_CORNER);
        break;
    }
    case L_PENALTY_AREA_B_SIDE:
    {
        p1 = getPoint2D(L_PENALTY_AREA_BL_CORNER);
        p2 = getPoint2D(L_PENALTY_AREA_BR_CORNER);
        break;
    }
    case L_PENALTY_AREA_R_SIDE:
    {
        p1 = getPoint2D(L_PENALTY_AREA_TR_CORNER);
        p2 = getPoint2D(L_PENALTY_AREA_BR_CORNER);
        break;
    }
    case R_PENALTY_AREA_T_SIDE:
    {
        p1 = getPoint2D(R_PENALTY_AREA_TL_CORNER);
        p2 = getPoint2D(R_PENALTY_AREA_TR_CORNER);
        break;
    }
    case R_PENALTY_AREA_B_SIDE:
    {
        p1 = getPoint2D(R_PENALTY_AREA_BL_CORNER);
        p2 = getPoint2D(R_PENALTY_AREA_BR_CORNER);
        break;
    }
    case R_PENALTY_AREA_L_SIDE:
    {
        p1 = getPoint2D(R_PENALTY_AREA_TL_CORNER);
        p2 = getPoint2D(R_PENALTY_AREA_BL_CORNER);
        break;
    }
    case L_GOAL_AREA_T_SIDE:
    {
        p1 = getPoint2D(L_GOAL_AREA_TL_CORNER);
        p2 = getPoint2D(L_GOAL_AREA_TR_CORNER);
        break;
    }
    case L_GOAL_AREA_B_SIDE:
    {
        p1 = getPoint2D(L_GOAL_AREA_BL_CORNER);
        p2 = getPoint2D(L_GOAL_AREA_BR_CORNER);
        break;
    }
    case L_GOAL_AREA_R_SIDE:
    {
        p1 = getPoint2D(L_GOAL_AREA_TR_CORNER);
        p2 = getPoint2D(L_GOAL_AREA_BR_CORNER);
        break;
    }
    case R_GOAL_AREA_T_SIDE:
    {
        p1 = getPoint2D(R_GOAL_AREA_TL_CORNER);
        p2 = getPoint2D(R_GOAL_AREA_TR_CORNER);
        break;
    }
    case R_GOAL_AREA_B_SIDE:
    {
        p1 = getPoint2D(R_GOAL_AREA_BL_CORNER);
        p2 = getPoint2D(R_GOAL_AREA_BR_CORNER);
        break;
    }
    case R_GOAL_AREA_L_SIDE:
    {
        p1 = getPoint2D(R_GOAL_AREA_TL_CORNER);
        p2 = getPoint2D(R_GOAL_AREA_BL_CORNER);
        break;
    }
    default:
    {
        p1 = getPoint2D(UNDEFINED_POINT);
        p2 = getPoint2D(UNDEFINED_POINT);
        break;
    }
    }

    return LineSegment2D(p1, p2);
}

Line2D SoccerPitch3D::getLine2D(LineID lineID) const
{
    LineSegment2D lineSegment2D = getLineSegment2D(lineID);

    return Line2D(lineSegment2D.start(), lineSegment2D.end());
}

Circle2D SoccerPitch3D::getCircle2D(LineID lineID) const
{
    double radius;
    Point2D center;

    switch (lineID)
    {
    case CENTER_CIRCLE:
    {

        radius = CENTER_CIRCLE_RADIUS;
        center = getPoint2D(CENTER_MARK);

        break;
    }
    case L_PENALTY_ARC:
    {

        radius = PENALTY_CIRCLE_RADIUS;
        center = getPoint2D(L_PENALTY_MARK);

        break;
    }
    case R_PENALTY_ARC:
    {

        radius = PENALTY_CIRCLE_RADIUS;
        center = getPoint2D(R_PENALTY_MARK);

        break;
    }
    case TL_CORNER_ARC:
    {
        radius = CORNER_CIRCLE_RADIUS;
        center = getPoint2D(TL_PITCH_CORNER);
        break;
    }
    case TR_CORNER_ARC:
    {
        radius = CORNER_CIRCLE_RADIUS;
        center = getPoint2D(TR_PITCH_CORNER);
        break;
    }
    case BL_CORNER_ARC:
    {
        radius = CORNER_CIRCLE_RADIUS;
        center = getPoint2D(BL_PITCH_CORNER);
        break;
    }
    case BR_CORNER_ARC:
    {
        radius = CORNER_CIRCLE_RADIUS;
        center = getPoint2D(BR_PITCH_CORNER);
        break;
    }
    default:
    {
        radius = numeric_limits<double>::signaling_NaN();
        center = getPoint2D(UNDEFINED_POINT);
        break;
    }
    }

    return Circle2D(center, radius);
}

CircleArc2D SoccerPitch3D::getCircleArc2D(LineID lineID) const
{
    Circle2D circle = getCircle2D(lineID);
    double fromAngle;
    double toAngle;

    switch (lineID)
    {
    case CENTER_CIRCLE:
    {
        fromAngle = 0.0;
        toAngle = CV_2PI;
        break;
    }
    case L_PENALTY_ARC:
    {
        Point2D topIntersection = getPoint2D(TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION);
        Point2D bottomIntersection = getPoint2D(BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION);
        fromAngle = atan2(topIntersection.y() - circle.center().y(), topIntersection.x() - circle.center().x()) + CV_2PI;
        toAngle = atan2(bottomIntersection.y() - circle.center().y(), bottomIntersection.x() - circle.center().x()) + CV_2PI;
        break;
    }
    case R_PENALTY_ARC:
    {
        Point2D topIntersection = getPoint2D(TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION);
        Point2D bottomIntersection = getPoint2D(BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION);
        fromAngle = atan2(bottomIntersection.y() - circle.center().y(), bottomIntersection.x() - circle.center().x());
        toAngle = atan2(topIntersection.y() - circle.center().y(), topIntersection.x() - circle.center().x()) + CV_2PI;
        break;
    }
    case TL_CORNER_ARC:
    {
        fromAngle = 0.0;
        toAngle = CV_PI / 2.0;
        break;
    }
    case TR_CORNER_ARC:
    {
        fromAngle = CV_PI / 2.0;
        toAngle = CV_PI;
        break;
    }
    case BL_CORNER_ARC:
    {
        fromAngle = 3.0 * CV_PI / 2.0;
        toAngle = CV_2PI;
        break;
    }
    case BR_CORNER_ARC:
    {
        fromAngle = CV_PI;
        toAngle = 3.0 * CV_PI / 2.0;
        break;
    }
    default:
    {
        fromAngle = numeric_limits<double>::signaling_NaN();
        toAngle = numeric_limits<double>::signaling_NaN();
        break;
    }
    }

    return CircleArc2D(circle, fromAngle, toAngle);
}

Point3D SoccerPitch3D::getPoint3D(PointID pointID) const
{
    Point2D point2D = getPoint2D(pointID);

    return getPoint3D(point2D);
}

Point3D SoccerPitch3D::getPoint3D(const Point2D &point2D) const
{
    double x = point2D.x();
    double y = point2D.y();
    double z = 0.;

    return Point3D(x, y, z);
}

Polyline3D SoccerPitch3D::getPolyline3D(LineID lineID, double delta) const
{
    Polyline3D polyline3D;

    switch (lineID)
    {
    case T_TOUCH_LINE:
    case B_TOUCH_LINE:
    case L_GOAL_LINE:
    case HALFWAY_LINE:
    case R_GOAL_LINE:
    case L_PENALTY_AREA_T_SIDE:
    case L_PENALTY_AREA_B_SIDE:
    case L_PENALTY_AREA_R_SIDE:
    case R_PENALTY_AREA_T_SIDE:
    case R_PENALTY_AREA_B_SIDE:
    case R_PENALTY_AREA_L_SIDE:
    case L_GOAL_AREA_T_SIDE:
    case L_GOAL_AREA_B_SIDE:
    case L_GOAL_AREA_R_SIDE:
    case R_GOAL_AREA_T_SIDE:
    case R_GOAL_AREA_B_SIDE:
    case R_GOAL_AREA_L_SIDE:
    {
        LineSegment2D lineSegment2D = getLineSegment2D(lineID);
        polyline3D = getPolyline3D(lineSegment2D, delta);
        break;
    }
    case CENTER_CIRCLE:
    case L_PENALTY_ARC:
    case R_PENALTY_ARC:
    case TL_CORNER_ARC:
    case TR_CORNER_ARC:
    case BL_CORNER_ARC:
    case BR_CORNER_ARC:
    {
        CircleArc2D circleArc2D = getCircleArc2D(lineID);
        polyline3D = getPolyline3D(circleArc2D, delta);
        break;
    }
    default:
    {
        break;
    }
    }

    return polyline3D;
}

Polyline2D SoccerPitch3D::getPolyline2D(LineID lineID, double delta) const
{
    Polyline2D polyline2D;

    switch (lineID)
    {
    case T_TOUCH_LINE:
    case B_TOUCH_LINE:
    case L_GOAL_LINE:
    case HALFWAY_LINE:
    case R_GOAL_LINE:
    case L_PENALTY_AREA_T_SIDE:
    case L_PENALTY_AREA_B_SIDE:
    case L_PENALTY_AREA_R_SIDE:
    case R_PENALTY_AREA_T_SIDE:
    case R_PENALTY_AREA_B_SIDE:
    case R_PENALTY_AREA_L_SIDE:
    case L_GOAL_AREA_T_SIDE:
    case L_GOAL_AREA_B_SIDE:
    case L_GOAL_AREA_R_SIDE:
    case R_GOAL_AREA_T_SIDE:
    case R_GOAL_AREA_B_SIDE:
    case R_GOAL_AREA_L_SIDE:
    {
        LineSegment2D lineSegment2D = getLineSegment2D(lineID);
        polyline2D = getPolyline2D(lineSegment2D, delta);
        break;
    }
    case CENTER_CIRCLE:
    case L_PENALTY_ARC:
    case R_PENALTY_ARC:
    case TL_CORNER_ARC:
    case TR_CORNER_ARC:
    case BL_CORNER_ARC:
    case BR_CORNER_ARC:
    {
        CircleArc2D circleArc2D = getCircleArc2D(lineID);
        polyline2D = getPolyline2D(circleArc2D, delta);
        break;
    }
    default:
    {
        break;
    }
    }

    return polyline2D;
}

Polyline3D SoccerPitch3D::getPolyline3D(const vector<Point2D> &points2D, double delta) const
{
    Polyline3D polyline3D;

    if (points2D.empty())
    {
        return polyline3D;
    }

    double x1 = points2D[0].x();
    double y1 = points2D[0].y();
    double z1 = 0.;
    polyline3D.emplace_back(Point3D(x1, y1, z1));

    for (int i = 1; i < points2D.size(); i++)
    {
        double x2 = points2D[i].x();
        double y2 = points2D[i].y();
        double z2 = 0.;

        double lx = x2 - x1;
        double ly = y2 - y1;
        double length = sqrt(lx * lx + ly * ly);

        if (delta > 0.0 && length > delta)
        {
            auto numInnerVertices = static_cast<int>(floor(length / delta));
            if (length - (numInnerVertices * delta) < delta / 2.0)
            {
                numInnerVertices--;
            }

            double dx = delta * lx / length;
            double dy = delta * ly / length;

            for (int j = 1; j <= numInnerVertices; j++)
            {
                double x = x1 + j * dx;
                double y = y1 + j * dy;
                double z = 0.;
                polyline3D.emplace_back(Point3D(x, y, z));
            }
        }

        polyline3D.emplace_back(Point3D(x2, y2, z2));

        x1 = x2;
        y1 = y2;
        z1 = z2;
    }

    return polyline3D;
}

Polyline3D SoccerPitch3D::getPolyline3D(const LineSegment2D &lineSegment2D, double delta) const
{
    double x1 = lineSegment2D.start().x();
    double y1 = lineSegment2D.start().y();
    double z1 = 0.;

    double x2 = lineSegment2D.end().x();
    double y2 = lineSegment2D.end().y();
    double z2 = 0.;

    Polyline3D polyline3D;
    polyline3D.emplace_back(Point3D(x1, y1, z1));

    double lx = x2 - x1;
    double ly = y2 - y1;
    double length = sqrt(lx * lx + ly * ly);

    if (delta > 0.0 && length > delta)
    {
        auto numInnerVertices = static_cast<int>(floor(length / delta));
        if (length - (numInnerVertices * delta) < delta / 2.0)
        {
            numInnerVertices--;
        }

        double dx = delta * lx / length;
        double dy = delta * ly / length;

        for (int i = 1; i <= numInnerVertices; i++)
        {
            double x = x1 + i * dx;
            double y = y1 + i * dy;
            double z = 0.;
            polyline3D.emplace_back(Point3D(x, y, z));
        }
    }

    polyline3D.emplace_back(Point3D(x2, y2, z2));

    return polyline3D;
}

Polyline2D SoccerPitch3D::getPolyline2D(const LineSegment2D &lineSegment2D, double delta) const
{
    double x1 = lineSegment2D.start().x();
    double y1 = lineSegment2D.start().y();

    double x2 = lineSegment2D.end().x();
    double y2 = lineSegment2D.end().y();

    Polyline2D polyline2D;
    polyline2D.emplace_back(Point2D(x1, y1));

    double lx = x2 - x1;
    double ly = y2 - y1;
    double length = sqrt(lx * lx + ly * ly);

    if (delta > 0.0 && length > delta)
    {
        auto numInnerVertices = static_cast<int>(floor(length / delta));
        if (length - (numInnerVertices * delta) < delta / 2.0)
        {
            numInnerVertices--;
        }

        double dx = delta * lx / length;
        double dy = delta * ly / length;

        for (int i = 1; i <= numInnerVertices; i++)
        {
            double x = x1 + i * dx;
            double y = y1 + i * dy;
            polyline2D.emplace_back(Point2D(x, y));
        }
    }

    polyline2D.emplace_back(Point2D(x2, y2));

    return polyline2D;
}

Polyline2D SoccerPitch3D::getPolyline2D(const CircleArc2D &circleArc2D, double delta) const
{
    double cx = circleArc2D.circle().center().x();
    double cy = circleArc2D.circle().center().y();
    double r = circleArc2D.circle().radius();
    double t1 = circleArc2D.startAngle();
    double t2 = circleArc2D.endAngle();

    while (t2 < t1)
    {
        t2 += CV_2PI;
    }

    double x1 = cx + cos(t1) * r;
    double y1 = cy + sin(t1) * r;

    double x2 = cx + cos(t2) * r;
    double y2 = cy + sin(t2) * r;

    Polyline2D polyline2D;
    polyline2D.emplace_back(Point2D(x1, y1));

    double length = r * (t2 - t1);

    if (delta > 0.0 && length > delta)
    {
        auto numInnerVertices = static_cast<int>(floor(length / delta));
        if (length - (numInnerVertices * delta) < delta / 2.0)
        {
            numInnerVertices--;
        }

        double dt = delta / r;

        for (int i = 1; i <= numInnerVertices; i++)
        {
            double t = t1 + i * dt;
            double x = cx + cos(t) * r;
            double y = cy + sin(t) * r;
            polyline2D.emplace_back(Point2D(x, y));
        }
    }

    polyline2D.emplace_back(Point2D(x2, y2));

    return polyline2D;
}

Polyline3D SoccerPitch3D::getPolyline3D(const CircleArc2D &circleArc2D, double delta) const
{
    double cx = circleArc2D.circle().center().x();
    double cy = circleArc2D.circle().center().y();
    double r = circleArc2D.circle().radius();
    double t1 = circleArc2D.startAngle();
    double t2 = circleArc2D.endAngle();

    while (t2 < t1)
    {
        t2 += CV_2PI;
    }

    double x1 = cx + cos(t1) * r;
    double y1 = cy + sin(t1) * r;
    double z1 = 0.;

    double x2 = cx + cos(t2) * r;
    double y2 = cy + sin(t2) * r;
    double z2 = 0.;

    Polyline3D polyline3D;
    polyline3D.emplace_back(Point3D(x1, y1, z1));

    double length = r * (t2 - t1);

    if (delta > 0.0 && length > delta)
    {
        auto numInnerVertices = static_cast<int>(floor(length / delta));
        if (length - (numInnerVertices * delta) < delta / 2.0)
        {
            numInnerVertices--;
        }

        double dt = delta / r;

        for (int i = 1; i <= numInnerVertices; i++)
        {
            double t = t1 + i * dt;
            double x = cx + cos(t) * r;
            double y = cy + sin(t) * r;
            double z = 0.;
            polyline3D.emplace_back(Point3D(x, y, z));
        }
    }

    polyline3D.emplace_back(Point3D(x2, y2, z2));

    return polyline3D;
}

vector<Polyline2D> SoccerPitch3D::getWireframe2D(double delta) const
{
    vector<Polyline2D> polylines2D;

    polylines2D.emplace_back(getPolyline2D(T_TOUCH_LINE, delta));
    polylines2D.emplace_back(getPolyline2D(B_TOUCH_LINE, delta));
    polylines2D.emplace_back(getPolyline2D(L_GOAL_LINE, delta));
    polylines2D.emplace_back(getPolyline2D(HALFWAY_LINE, delta));
    polylines2D.emplace_back(getPolyline2D(R_GOAL_LINE, delta));
    polylines2D.emplace_back(getPolyline2D(L_PENALTY_AREA_T_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(L_PENALTY_AREA_B_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(L_PENALTY_AREA_R_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_PENALTY_AREA_T_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_PENALTY_AREA_B_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_PENALTY_AREA_L_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(L_GOAL_AREA_T_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(L_GOAL_AREA_B_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(L_GOAL_AREA_R_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_GOAL_AREA_T_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_GOAL_AREA_B_SIDE, delta));
    polylines2D.emplace_back(getPolyline2D(R_GOAL_AREA_L_SIDE, delta));

    polylines2D.emplace_back(getPolyline2D(L_PENALTY_ARC, delta));
    polylines2D.emplace_back(getPolyline2D(CENTER_CIRCLE, delta));
    polylines2D.emplace_back(getPolyline2D(R_PENALTY_ARC, delta));
    polylines2D.emplace_back(getPolyline2D(TL_CORNER_ARC, delta));
    polylines2D.emplace_back(getPolyline2D(TR_CORNER_ARC, delta));
    polylines2D.emplace_back(getPolyline2D(BL_CORNER_ARC, delta));
    polylines2D.emplace_back(getPolyline2D(BR_CORNER_ARC, delta));

    Polyline2D centerMark;
    centerMark.emplace_back(getPoint2D(CENTER_MARK));
    polylines2D.emplace_back(centerMark);

    Polyline2D leftPenaltyMark;
    leftPenaltyMark.emplace_back(getPoint2D(L_PENALTY_MARK));
    polylines2D.emplace_back(leftPenaltyMark);

    Polyline2D rightPenaltyMark;
    rightPenaltyMark.emplace_back(getPoint2D(R_PENALTY_MARK));
    polylines2D.emplace_back(rightPenaltyMark);

    return polylines2D;
}

vector<Polyline3D> SoccerPitch3D::getWireframe(double delta) const
{
    vector<Polyline3D> polylines3D;

    polylines3D.emplace_back(getPolyline3D(T_TOUCH_LINE, delta));
    polylines3D.emplace_back(getPolyline3D(B_TOUCH_LINE, delta));
    polylines3D.emplace_back(getPolyline3D(L_GOAL_LINE, delta));
    polylines3D.emplace_back(getPolyline3D(HALFWAY_LINE, delta));
    polylines3D.emplace_back(getPolyline3D(R_GOAL_LINE, delta));
    polylines3D.emplace_back(getPolyline3D(L_PENALTY_AREA_T_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(L_PENALTY_AREA_B_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(L_PENALTY_AREA_R_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_PENALTY_AREA_T_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_PENALTY_AREA_B_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_PENALTY_AREA_L_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(L_GOAL_AREA_T_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(L_GOAL_AREA_B_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(L_GOAL_AREA_R_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_GOAL_AREA_T_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_GOAL_AREA_B_SIDE, delta));
    polylines3D.emplace_back(getPolyline3D(R_GOAL_AREA_L_SIDE, delta));

    polylines3D.emplace_back(getPolyline3D(L_PENALTY_ARC, delta));
    polylines3D.emplace_back(getPolyline3D(CENTER_CIRCLE, delta));
    polylines3D.emplace_back(getPolyline3D(R_PENALTY_ARC, delta));
    polylines3D.emplace_back(getPolyline3D(TL_CORNER_ARC, delta));
    polylines3D.emplace_back(getPolyline3D(TR_CORNER_ARC, delta));
    polylines3D.emplace_back(getPolyline3D(BL_CORNER_ARC, delta));
    polylines3D.emplace_back(getPolyline3D(BR_CORNER_ARC, delta));

    Polyline3D centerMark;
    centerMark.emplace_back(getPoint3D(CENTER_MARK));
    polylines3D.emplace_back(centerMark);

    Polyline3D leftPenaltyMark;
    leftPenaltyMark.emplace_back(getPoint3D(L_PENALTY_MARK));
    polylines3D.emplace_back(leftPenaltyMark);

    Polyline3D rightPenaltyMark;
    rightPenaltyMark.emplace_back(getPoint3D(R_PENALTY_MARK));
    polylines3D.emplace_back(rightPenaltyMark);

    return polylines3D;
}
