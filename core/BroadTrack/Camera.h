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
#include "SoccerPitch3D.h"

#include "opencv2/opencv.hpp"

#include <vector>

class Camera
{

public:
    static constexpr double GIMBAL_LOCK_EPSILON = 1.0e-9;

    static constexpr double HD_WIDTH_PIXELS = 1920.0;

    static constexpr double HD_HEIGHT_PIXELS = 1080.0;

    static Matrix3x3 getOrientationFromPanTiltRoll(const Vector3x1 &cameraPanTiltRoll);

    static std::vector<Vector3x1> getPanTiltRollSolutionsFromOrientation(const Matrix3x3 &cameraOrientation);

    static Vector3x1 getPanTiltRollFromOrientation(const Matrix3x3 &cameraOrientation, double rollHint = 0.0);

    explicit Camera(const cv::Size &pixelResolution = cv::Size(HD_WIDTH_PIXELS, HD_HEIGHT_PIXELS));

    /* EXTRINSIC PARAMETERS */

    Camera &setRotation(const Matrix3x3 &rotationMatrix);

    Camera &setOrientation(const Matrix3x3 &orientationMatrix);

    Camera &setPanTiltRoll(const Vector3x1 &panTiltRoll);

    Camera &setTranslation(const Vector3x1 &translationVector);

    Camera &setPosition(const Vector3x1 &positionVector);

    Camera &setRadialDistortion(const std::vector<double> &distortionCoefficients);

    Camera &setRadialCorrection(const std::vector<double> &correctionCoefficients);

    Camera &setNormalizedRadialDistortion(const std::vector<double> &normalizedDistortionCoefficients);

    Camera &setNormalizedRadialCorrection(const std::vector<double> &normalizedCorrectionCoefficients);

    Camera &setFocalLength(double focalLength);

    Camera &setVerticalFieldOfView(double verticalFieldOfView);

    Camera &setHorizontalFieldOfView(double horizontalFieldOfView);

    Camera &setPrincipalPoint(const Point2D &principalPoint);

    Camera &setPixelResolution(const cv::Size &pixelResolution);

    Camera &setCalibration(const Matrix3x3 &calibrationMatrix);

    Camera &setPinhole(const Matrix3x3 &groundPlaneHomography);

    Camera &setPose(const Vector3x1 &targetPoint, const Vector3x1 &upVector, const Vector3x1 &position);

    Camera &setPinhole(const Vector3x1 &targetPoint, const Vector3x1 &upVector, const Vector3x1 &position, double vFov);

    Matrix3x3 getRotation() const;

    Vector3x1 getAngleAxis() const;

    Matrix3x3 getOrientation() const;

    Vector3x1 getPanTiltRoll(double rollHint = 0.0) const;

    Vector3x1 getTargetPoint() const;

    Vector3x1 getUpVector() const;

    Vector3x1 getTranslation() const;

    Vector3x1 getPosition() const;

    std::vector<double> getRadialDistortion() const;

    std::vector<double> getRadialCorrection() const;

    std::vector<double> getNormalizedRadialDistortion() const;

    std::vector<double> getNormalizedRadialCorrection() const;

    double getFocalLength() const;

    double getVerticalFieldOfView() const;

    double getHorizontalFieldOfView() const;

    Point2D getPrincipalPoint() const;

    cv::Size getPixelResolution() const;

    Matrix3x3 getCalibration() const;

    Matrix3x3 getGroundPlaneHomography() const;

    // Get the vector linking the camera position and the unprojected point
    Vector3x1 getWorldVector(const Point2D &point) const;

    Line3D getRay(const Point2D &pixelCoordinates) const;

    bool transformToCameraCoordinates(const Point3D &worldCoordinates, Point3D &cameraCoordinates) const;

    bool projectToNormalizedImagePlane(const Point3D &cameraCoordinates, Point2D &normalizedImageCoordinates) const;

    bool distort(const Point2D &normalizedImageCoordinates, Point2D &distortedNormalizedImageCoordinates) const;

    bool undistort(const Point2D &distortedNormalizedImageCoordinates, Point2D &normalizedImageCoordinates) const;

    bool transformToPixelCoordinates(const Point2D &normalizedImageCoordinates, Point2D &pixelCoordinates) const;

    bool transformToNormalizedImageCoordinates(const Point2D &pixelCoordinates,
                                               Point2D &normalizedImageCoordinates) const;

    bool project(const Point3D &worldCoordinates, Point2D &pixelCoordinates, bool clip = true) const;

    std::vector<Polyline2D> project(const Polyline3D &worldCurve, bool clip = true) const;

    std::vector<Polyline2D> project(const std::vector<Polyline3D> &worldWireframe, bool clip = true) const;

    std::vector<Polyline2D> projectWireframe(const SoccerPitch3D &object3D, bool clip = true) const;

    cv::Mat &drawWireframe(const SoccerPitch3D &object3D,
                           cv::Mat &image,
                           const cv::Scalar &color,
                           int lineThickness = 1,
                           bool antiAliasing = true) const;

    cv::Mat &draw(const std::vector<Polyline2D> &pixelCurves,
                  cv::Mat &image,
                  const cv::Scalar &color,
                  int lineThickness = 1,
                  bool antiAliasing = true) const;

    cv::Mat &draw(const Point3D &point3D,
                  cv::Mat &image,
                  const cv::Scalar &color,
                  int lineThickness = 1,
                  bool antiAliasing = true) const;

    std::string toJSONString() const;

    Camera &fromJSONString(std::string JSONString);

private:
    double getLargestRadiusInNormalizedImagePlane() const;

    Matrix3x3 _rotationMatrix = Matrix3x3::identity();
    Vector3x1 _translationVector = Vector3x1::zero();
    std::vector<double> _distortionCoefficients = {};
    std::vector<double> _correctionCoefficients = {};
    double _focalLength = 1.0;
    cv::Size _pixelResolution;
    double _clippingCircleRadius;
};

std::ostream &operator<<(std::ostream &os, const Camera &c);
