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
#include "PointExtractor.h"
#include "core.h"
#include "Camera.h"
#include "SoccerPitch3D.h"

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <deque>
#include <tuple>

namespace fs = boost::filesystem;

class CameraTracker
{

public:
    CameraTracker();

    std::tuple<double, Camera> update(const cv::Mat &semLinesMask,
                                      const std::vector<std::pair<Point3D, Point2D>> &pitchProjections,
                                      bool softPosition = false,
                                      bool ptz = false,
                                      bool lensDistortion = true,
                                      int cauchyParameter = 10);

    void reinit(const std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &detectedPoints, int threshold = 2, int n_iterations = 20);

    std::tuple<double, double> evaluateReprojectionError(const std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &points, int threshold, std::vector<bool> &outInliers, const Camera &camera);

    std::tuple<double, double> estimatePanTilt(const std::vector<std::pair<Point3D, Point2D>> &detectedPoints,
                                               double focal,
                                               Point3D position,
                                               size_t iwidth,
                                               size_t iheight);

    double evaluate(const cv::Mat &semLinesMask);

    double evaluateWith(const cv::Mat &semLinesMask, const Camera &camera);

    Matrix3x3 homographyOf(const Camera &camera, double scale = 0.5);

    double scoreCameraWithMask(const cv::Mat &rawLines, const Camera &camera);

    Camera getCamera() const;

    void setCamera(const Camera &camera);

    void setTripodInfo(const Point3D &center, double radius);

private:
    Camera _camera;
    SoccerPitch3D _soccerPitch3D;

    Point3D _tripodCenter;
    PointExtractor _pointExtractor;
    double _tripodRadius;
    double _lambda;
    // Camera-position anchor to prevent gauge drift. The line-mask fit leaves
    // a position<->focal<->k1 gauge: a free position lets the per-frame solve
    // slowly walk the camera off the pitch lines while the line reprojection
    // error stays small (measured here: ~2m position drift with a ~0.15 line-IoU
    // drop over a sequence). For a tripod-mounted camera the position is
    // constant, so it is pinned to the value averaged over the first few free
    // solves; focal/zoom stay free to follow the lens.
    bool _positionAnchored;
    Point3D _referencePosition;
    // k1 radial distortion is a lens constant, so it is pinned to the
    // averaged anchor too. Once the position is pinned, the line-mask fit
    // still has a focal<->k1 gauge: on occluded/degraded masks the optimizer
    // can jump k1 to a large value (+0.05 measured) with a ~0.18 score drop,
    // then snap back. Pinning k1 leaves pan/tilt/roll + focal as the only free
    // DOFs, which are exactly the natural DOFs of a PTZ tripod camera.
    bool _k1Anchored;
    double _referenceK1;
    // Roll anchor (optional). Roll is constant for a tripod-mounted head, so
    // the anchor captures it like position/k1 and RollPinningResidual can pin
    // it (BROADTRACK_ROLL_WEIGHT). It is DISABLED by default: experiments on
    // this sequence showed the line-mask genuinely fits better with a slowly
    // drifting roll (pinning it lowered the mask fit at every frame), so the
    // roll drift is what keeps the overlay aligned and is not the cause of the
    // visible drift. The visible drift is instead a pan lag that the line-IoU
    // pan refinement in update() corrects (see BROADTRACK_NO_PAN_REFINEMENT).
    bool _rollAnchored;
    double _referenceRoll;
    // Accumulators for the anchor: the first few well-fitting free-position
    // solves are averaged before pinning (see the capture block in update()).
    int _anchorFrameCount;
    double _anchorPositionSumX, _anchorPositionSumY, _anchorPositionSumZ;
    double _anchorK1Sum;
    double _anchorRollSum;
    // History of per-frame line-IoU refined pans for the causal median filter
    // in update() (see BROADTRACK_PAN_SMOOTH_WINDOW).
    std::deque<double> _refinedPanHistory;
};
