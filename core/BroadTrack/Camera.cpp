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

#include "Camera.h"
#include "ceres/rotation.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

using namespace cv;
using namespace std;
using namespace rapidjson;

#define CV_DRAW_SHIFT 2

constexpr double Camera::GIMBAL_LOCK_EPSILON;

constexpr double Camera::HD_WIDTH_PIXELS;

constexpr double Camera::HD_HEIGHT_PIXELS;

Matrix3x3 Camera::getOrientationFromPanTiltRoll(const Vector3x1 &cameraPanTiltRoll)
{
    double pan = cameraPanTiltRoll[0];
    double tilt = cameraPanTiltRoll[1];
    double roll = cameraPanTiltRoll[2];

    Matrix3x3 Rt(1.0, 0.0, 0.0,
                 0.0, cos(tilt), -sin(tilt),
                 0.0, sin(tilt), cos(tilt));
    Matrix3x3 Rp(cos(pan), -sin(pan), 0.0,
                 sin(pan), cos(pan), 0.0,
                 0.0, 0.0, 1.0);
    Matrix3x3 Rr(cos(roll), -sin(roll), 0.0,
                 sin(roll), cos(roll), 0.0,
                 0.0, 0.0, 1.0);

    return Rp * Rt * Rr;
}

vector<Vector3x1> Camera::getPanTiltRollSolutionsFromOrientation(const Matrix3x3 &cameraOrientation)
{
    vector<Vector3x1> solutions;

    bool gimbalLock = fabs(fabs(cameraOrientation[8]) - 1.0) < GIMBAL_LOCK_EPSILON;

    if (gimbalLock)
    {
        double tilt;
        double rollCoefficientForPanEquation;
        double constantTermForPanEquation;

        if (cameraOrientation[8] > 0.0)
        {
            tilt = 0.0;
            rollCoefficientForPanEquation = -1.0;
            constantTermForPanEquation = atan2(cameraOrientation[3], cameraOrientation[4]);
        }
        else
        {
            tilt = CV_PI;
            rollCoefficientForPanEquation = 1.0;
            constantTermForPanEquation = atan2(cameraOrientation[1], cameraOrientation[0]);
        }

        Vector3x1 generalSolution(tilt, rollCoefficientForPanEquation, constantTermForPanEquation);

        solutions.emplace_back(generalSolution);
    }
    else
    {
        double firstTilt = acos(cameraOrientation[8]);
        double secondTilt = -firstTilt;
        double signSinFirstTilt = sin(firstTilt) > 0.0 ? 1.0 : -1.0;
        double signSinSecondTilt = sin(secondTilt) > 0.0 ? 1.0 : -1.0;
        double firstPan = atan2(signSinFirstTilt * cameraOrientation[2], signSinFirstTilt * -cameraOrientation[5]);
        double secondPan = atan2(signSinSecondTilt * cameraOrientation[2], signSinSecondTilt * -cameraOrientation[5]);
        double firstRoll = atan2(signSinFirstTilt * cameraOrientation[6], signSinFirstTilt * cameraOrientation[7]);
        double secondRoll = atan2(signSinSecondTilt * cameraOrientation[6], signSinSecondTilt * cameraOrientation[7]);

        Vector3x1 firstSolution(firstPan, firstTilt, firstRoll);
        Vector3x1 secondSolution(secondPan, secondTilt, secondRoll);

        solutions.emplace_back(firstSolution);
        solutions.emplace_back(secondSolution);
    }

    return solutions;
}

Vector3x1 Camera::getPanTiltRollFromOrientation(const Matrix3x3 &cameraOrientation, double rollHint)
{
    vector<Vector3x1> solutions = getPanTiltRollSolutionsFromOrientation(cameraOrientation);

    double particularPan = 0.0;
    double particularTilt = 0.0;
    double particularRoll = 0.0;

    if (solutions.size() == 2)
    {
        double firstRoll = solutions[0][2];
        double firstRollDifference = fmod(fabs(firstRoll - rollHint), CV_2PI);
        double firstSmallestRollDifference = min(CV_2PI - firstRollDifference, firstRollDifference);

        double secondRoll = solutions[1][2];
        double secondRollDifference = fmod(fabs(secondRoll - rollHint), CV_2PI);
        double secondSmallestRollDifference = min(CV_2PI - secondRollDifference, secondRollDifference);

        int preferredSolutionIndex = firstSmallestRollDifference < secondSmallestRollDifference ? 0 : 1;

        particularPan = solutions[preferredSolutionIndex][0];
        particularTilt = solutions[preferredSolutionIndex][1];
        particularRoll = solutions[preferredSolutionIndex][2];
    }
    else if (solutions.size() == 1)
    {
        particularTilt = solutions[0][0];
        particularRoll = rollHint;
        double rollCoefficientForPanEquation = solutions[0][1];
        double constantTermForPanEquation = solutions[0][2];
        particularPan = rollCoefficientForPanEquation * particularRoll + constantTermForPanEquation;
    }

    Vector3x1 particularSolution;
    particularSolution[0] = particularPan;
    particularSolution[1] = particularTilt;
    particularSolution[2] = particularRoll;

    return particularSolution;
}

Camera::Camera(const cv::Size &pixelResolution)
{
    _pixelResolution = pixelResolution;
    _clippingCircleRadius = getLargestRadiusInNormalizedImagePlane();
}

Camera &Camera::setRotation(const Matrix3x3 &rotationMatrix)
{
    _rotationMatrix = rotationMatrix;

    return *this;
}

Camera &Camera::setOrientation(const Matrix3x3 &orientationMatrix)
{
    _rotationMatrix = orientationMatrix.transpose();

    return *this;
}

Camera &Camera::setPanTiltRoll(const Vector3x1 &panTiltRoll)
{
    setRotation(getOrientationFromPanTiltRoll(panTiltRoll).transpose());

    return *this;
}

Camera &Camera::setPosition(const Vector3x1 &positionVector)
{
    _translationVector = -_rotationMatrix * positionVector;

    return *this;
}

Camera &Camera::setRadialDistortion(const vector<double> &distortionCoefficients)
{
    _distortionCoefficients = distortionCoefficients;

    return *this;
}

Camera &Camera::setFocalLength(double focalLength)
{
    _focalLength = focalLength;

    return *this;
}

Camera &Camera::setVerticalFieldOfView(double verticalFieldOfView)
{
    _focalLength = _pixelResolution.height / (2.0 * tan(verticalFieldOfView / 2.0));

    return *this;
}

Camera &Camera::setHorizontalFieldOfView(double horizontalFieldOfView)
{
    _focalLength = _pixelResolution.width / (2.0 * tan(horizontalFieldOfView / 2.0));

    return *this;
}

Camera &Camera::setPrincipalPoint(const Point2D &principalPoint)
{
    _pixelResolution = cv::Size(2.0 * principalPoint.x(), 2.0 * principalPoint.y());

    return *this;
}

Camera &Camera::setPixelResolution(const cv::Size &pixelResolution)
{
    _pixelResolution = pixelResolution;

    return *this;
}

Camera &Camera::setCalibration(const Matrix3x3 &calibrationMatrix)
{
    _focalLength = (calibrationMatrix[0] + calibrationMatrix[4]) / (2.0 * calibrationMatrix[8]);

    Point2D principalPoint(calibrationMatrix[2], calibrationMatrix[5], calibrationMatrix[8]);
    _pixelResolution = cv::Size(2.0 * principalPoint.x(), 2.0 * principalPoint.y());

    return *this;
}

Camera &Camera::setPose(const Vector3x1 &targetPoint,
                        const Vector3x1 &upVector,
                        const Vector3x1 &position)
{

    Vector3x1 Rc3 = (targetPoint - position).unitLength();
    Vector3x1 Rc1 = Rc3.crossProduct(upVector).unitLength();
    Vector3x1 Rc2 = Rc3.crossProduct(Rc1);
    Matrix3x3 Rc(Rc1, Rc2, Rc3);

    setOrientation(Rc);
    setPosition(position);

    return *this;
}

Camera &Camera::setPinhole(const Vector3x1 &targetPoint,
                           const Vector3x1 &upVector,
                           const Vector3x1 &position,
                           double vFov)
{
    setPose(targetPoint, upVector, position);
    setVerticalFieldOfView(vFov);

    return *this;
}

Vector3x1 Camera::getAngleAxis() const
{
    Vector3x1 angleAxisVector;
    // TODO: ceres expects R, not Rc, but expects column major (therefore Rc in row major is provided)
    ceres::RotationMatrixToAngleAxis(getOrientation().data(), angleAxisVector.data());

    return angleAxisVector;
}

Matrix3x3 Camera::getRotation() const
{
    return _rotationMatrix;
}

Matrix3x3 Camera::getOrientation() const
{
    return _rotationMatrix.transpose();
}

Vector3x1 Camera::getPanTiltRoll(double rollHint) const
{
    return getPanTiltRollFromOrientation(getOrientation(), rollHint);
}

Vector3x1 Camera::getTargetPoint() const
{
    return getPosition() + getOrientation().column(2);
}

Vector3x1 Camera::getUpVector() const
{
    return -getOrientation().column(1);
}

Vector3x1 Camera::getTranslation() const
{
    return _translationVector;
}

Vector3x1 Camera::getPosition() const
{
    return -getOrientation() * _translationVector;
}

vector<double> Camera::getRadialDistortion() const
{
    return _distortionCoefficients;
}

vector<double> Camera::getRadialCorrection() const
{
    return _correctionCoefficients;
}

vector<double> Camera::getNormalizedRadialDistortion() const
{
    vector<double> normalizedDistortionCoefficients;
    double normalizedFocalLength = _pixelResolution.height / _focalLength;
    for (int i = 0; i < _distortionCoefficients.size(); i++)
    {
        normalizedDistortionCoefficients.emplace_back(
            _distortionCoefficients[i] * pow(normalizedFocalLength, 2.0 * (i + 1)));
    }

    return normalizedDistortionCoefficients;
}

vector<double> Camera::getNormalizedRadialCorrection() const
{
    vector<double> normalizedCorrectionCoefficients;
    double normalizedFocalLength = _pixelResolution.height / _focalLength;
    for (int i = 0; i < _correctionCoefficients.size(); i++)
    {
        normalizedCorrectionCoefficients.emplace_back(
            _correctionCoefficients[i] * pow(normalizedFocalLength, 2.0 * (i + 1)));
    }

    return normalizedCorrectionCoefficients;
}

double Camera::getFocalLength() const
{
    return _focalLength;
}

double Camera::getVerticalFieldOfView() const
{
    return 2.0 * atan2(_pixelResolution.height / 2.0, _focalLength);
}

double Camera::getHorizontalFieldOfView() const
{
    return 2.0 * atan2(_pixelResolution.width / 2.0, _focalLength);
}

Point2D Camera::getPrincipalPoint() const
{
    return Point2D(_pixelResolution.width / 2.0, _pixelResolution.height / 2.0);
}

cv::Size Camera::getPixelResolution() const
{
    return _pixelResolution;
}

Matrix3x3 Camera::getCalibration() const
{
    return Matrix3x3(_focalLength, 0.0, _pixelResolution.width / 2.0,
                     0.0, _focalLength, _pixelResolution.height / 2.0,
                     0.0, 0.0, 1.0);
}

Matrix3x3 Camera::getGroundPlaneHomography() const
{
    Matrix3x3 K = getCalibration();
    Matrix3x3 KR = K * _rotationMatrix;
    Vector3x1 Kt = K * _translationVector;

    return Matrix3x3(KR[0], KR[1], Kt[0],
                     KR[3], KR[4], Kt[1],
                     KR[6], KR[7], Kt[2]);
}

Vector3x1 Camera::getWorldVector(const Point2D &point) const
{
    return getOrientation() * getCalibration().inverse() * point;
}

Line3D Camera::getRay(const Point2D &pixelCoordinates) const
{
    Point2D distortedNormalizedImageCoordinates;
    transformToNormalizedImageCoordinates(pixelCoordinates, distortedNormalizedImageCoordinates);

    Point2D normalizedImageCoordinates;
    undistort(distortedNormalizedImageCoordinates, normalizedImageCoordinates);

    Vector3x1 raySlope = getOrientation() * Vector3x1(normalizedImageCoordinates.x(),
                                                      normalizedImageCoordinates.y(),
                                                      1.0);

    Point3D rayOrigin(getPosition());

    Line3D ray(rayOrigin, raySlope);

    return ray;
}

bool Camera::transformToCameraCoordinates(const Point3D &worldCoordinates, Point3D &cameraCoordinates) const
{
    cameraCoordinates = _rotationMatrix * worldCoordinates + Point3D(_translationVector);

    return cameraCoordinates.z() > 0.0;
}

bool Camera::projectToNormalizedImagePlane(const Point3D &cameraCoordinates, Point2D &normalizedImageCoordinates) const
{
    normalizedImageCoordinates = cameraCoordinates.normalizedCentralProjection();

    return !(cameraCoordinates.hx() == 0.0 && cameraCoordinates.hy() == 0.0 && cameraCoordinates.hz() == 0.0);
}

bool Camera::distort(const Point2D &normalizedImageCoordinates, Point2D &distortedNormalizedImageCoordinates) const
{
    distortedNormalizedImageCoordinates = normalizedImageCoordinates;

    double squaredRadius = 0.0;

    if (!_distortionCoefficients.empty())
    {
        squaredRadius = normalizedImageCoordinates.squaredNorm();

        double distortion = 1.0;

        for (int k = 0; k < _distortionCoefficients.size(); k++)
        {
            distortion += _distortionCoefficients[k] * pow(squaredRadius, k + 1);
        }

        distortedNormalizedImageCoordinates = normalizedImageCoordinates * distortion;
    }

    return sqrt(squaredRadius) < _clippingCircleRadius;
}

bool Camera::undistort(const Point2D &distortedNormalizedImageCoordinates, Point2D &normalizedImageCoordinates) const
{
    normalizedImageCoordinates = distortedNormalizedImageCoordinates;

    double squaredDistortedRadius = 0.0;

    if (!_correctionCoefficients.empty())
    {
        squaredDistortedRadius = distortedNormalizedImageCoordinates.squaredNorm();

        double correction = 1.0;

        for (int k = 0; k < _correctionCoefficients.size(); k++)
        {
            correction += _correctionCoefficients[k] * pow(squaredDistortedRadius, k + 1);
        }

        normalizedImageCoordinates = distortedNormalizedImageCoordinates * correction;
    }

    double distortedClippingCircleRadius = getLargestRadiusInNormalizedImagePlane();

    return sqrt(squaredDistortedRadius) < distortedClippingCircleRadius;
}

bool Camera::transformToPixelCoordinates(const Point2D &normalizedImageCoordinates, Point2D &pixelCoordinates) const
{
    pixelCoordinates = Point2D(getCalibration() * normalizedImageCoordinates);

    return pixelCoordinates.isWithin(_pixelResolution);
}

bool Camera::transformToNormalizedImageCoordinates(const Point2D &pixelCoordinates,
                                                   Point2D &normalizedImageCoordinates) const
{
    normalizedImageCoordinates = Point2D(getCalibration().inverse() * pixelCoordinates);

    return pixelCoordinates.isWithin(_pixelResolution);
}

bool Camera::project(const Point3D &worldCoordinates, Point2D &pixelCoordinates, bool clip) const
{
    Point3D cameraCoordinates;
    bool inFrontOfCamera = transformToCameraCoordinates(worldCoordinates, cameraCoordinates);

    Point2D normalizedImageCoordinates;
    bool notTheOrigin = projectToNormalizedImagePlane(cameraCoordinates, normalizedImageCoordinates);

    Point2D distortedNormalizedImageCoordinates;
    bool withinDistortionDomain = distort(normalizedImageCoordinates, distortedNormalizedImageCoordinates);

    bool withinPixelResolution = transformToPixelCoordinates(distortedNormalizedImageCoordinates, pixelCoordinates);

    return !clip || (inFrontOfCamera && notTheOrigin && withinDistortionDomain && withinPixelResolution);
}

vector<Polyline2D> Camera::project(const Polyline3D &worldCurve, bool clip) const
{
    vector<Polyline2D> pixelCurves;

    Polyline2D currentPixelCurve;
    for (const auto &worldPoint : worldCurve)
    {
        Point2D pixelPoint;
        bool validProjection = project(worldPoint, pixelPoint, clip);
        if (validProjection)
        {
            currentPixelCurve.emplace_back(pixelPoint);
        }
        else if (!currentPixelCurve.empty())
        {
            pixelCurves.emplace_back(currentPixelCurve);
            currentPixelCurve.clear();
        }
    }

    if (!currentPixelCurve.empty())
    {
        pixelCurves.emplace_back(currentPixelCurve);
    }

    return pixelCurves;
}

vector<Polyline2D> Camera::project(const vector<Polyline3D> &worldWireframe, bool clip) const
{
    vector<Polyline2D> pixelLocationsPolylines;

    for (const auto &worldPointsPolyline : worldWireframe)
    {
        vector<Polyline2D> currentPixelLocationsPolylines = project(worldPointsPolyline, clip);
        pixelLocationsPolylines.insert(pixelLocationsPolylines.end(),
                                       currentPixelLocationsPolylines.begin(),
                                       currentPixelLocationsPolylines.end());
    }

    return pixelLocationsPolylines;
}

vector<Polyline2D> Camera::projectWireframe(const SoccerPitch3D &object3D, bool clip) const
{
    vector<Polyline3D> wireframe3D = object3D.getWireframe(0.1);

    vector<Polyline2D> wireframe2D = project(wireframe3D, clip);

    return wireframe2D;
}

cv::Mat &Camera::drawWireframe(const SoccerPitch3D &object3D,
                               cv::Mat &image,
                               const cv::Scalar &color,
                               int lineThickness,
                               bool antiAliasing) const
{
    vector<Polyline2D> wireframe2D = projectWireframe(object3D);
    draw(wireframe2D, image, color, lineThickness, antiAliasing);

    return image;
}

cv::Mat &Camera::draw(const vector<Polyline2D> &pixelCurves,
                      cv::Mat &cvImage,
                      const cv::Scalar &cvColor,
                      int lineThickness,
                      bool antiAliasing) const
{

    int lineType = antiAliasing ? LINE_AA : LINE_8;

    for (auto polyline : pixelCurves)
    {
        if (polyline.size() > 1)
        {

            for (int i = 0; i < polyline.size() - 1; i++)
            {
                Point2D p1 = polyline[i];
                Point2D p2 = polyline[i + 1];

                Point cvP1(static_cast<int>(round(p1.x() * pow(2, CV_DRAW_SHIFT))),
                           static_cast<int>(round(p1.y() * pow(2, CV_DRAW_SHIFT))));

                Point cvP2(static_cast<int>(round(p2.x() * pow(2, CV_DRAW_SHIFT))),
                           static_cast<int>(round(p2.y() * pow(2, CV_DRAW_SHIFT))));

                line(cvImage, cvP1, cvP2, cvColor, lineThickness, lineType, CV_DRAW_SHIFT);
            }
        }
        else if (polyline.size() == 1)
        {
            Point2D p = polyline[0];

            Point cvP(static_cast<int>(round(p.x() * pow(2, CV_DRAW_SHIFT))),
                      static_cast<int>(round(p.y() * pow(2, CV_DRAW_SHIFT))));

            circle(cvImage, cvP, 2 * lineThickness, cvColor, -1, lineType, CV_DRAW_SHIFT);
        }
    }

    return cvImage;
}

cv::Mat &Camera::draw(const Point3D &point3D,
                      cv::Mat &cvImage,
                      const cv::Scalar &cvColor,
                      int lineThickness,
                      bool antiAliasing) const
{

    int lineType = antiAliasing ? LINE_AA : LINE_8;

    Point2D point2D;
    project(point3D, point2D);

    Point cvP(static_cast<int>(round(point2D.x() * pow(2, CV_DRAW_SHIFT))),
              static_cast<int>(round(point2D.y() * pow(2, CV_DRAW_SHIFT))));

    circle(cvImage, cvP, 2 * lineThickness, cvColor, -1, lineType, CV_DRAW_SHIFT);

    return cvImage;
}

/* PRIVATE CLASSES AND FUNCTIONS */

double Camera::getLargestRadiusInNormalizedImagePlane() const
{
    Matrix3x3 invK = getCalibration().inverse();
    Point2D imageTopLeft = Point2D(0.0, 0.0).projected(invK).normalized();
    Point2D imageTopRight = Point2D(_pixelResolution.width, 0.0).projected(invK).normalized();
    Point2D imageBottomLeft = Point2D(0.0, _pixelResolution.height).projected(invK).normalized();
    Point2D imageBottomRight = Point2D(_pixelResolution.width, _pixelResolution.height).projected(invK).normalized();

    return max<double>({imageTopLeft.norm(), imageTopRight.norm(), imageBottomLeft.norm(), imageBottomRight.norm()});
}

string Camera::toJSONString() const
{
    Document document;
    document.SetObject();
    Document::AllocatorType &allocator = document.GetAllocator();

    cv::Size sensorResolution = getPixelResolution();
    document.AddMember("sensorResolutionWidthPixels", Value().SetDouble(sensorResolution.width), allocator);
    document.AddMember("sensorResolutionHeightPixels", Value().SetDouble(sensorResolution.height), allocator);

    document.AddMember("horizontalFieldOfViewDegrees",
                       Value().SetDouble(getHorizontalFieldOfView() * 180.0 / CV_PI), allocator);

    Vector3x1 rotationAngles = getPanTiltRoll();
    document.AddMember("panDegrees", Value().SetDouble(rotationAngles[0] * 180.0 / CV_PI), allocator);
    document.AddMember("tiltDegrees", Value().SetDouble(rotationAngles[1] * 180.0 / CV_PI), allocator);
    document.AddMember("rollDegrees", Value().SetDouble(rotationAngles[2] * 180.0 / CV_PI), allocator);

    Vector3x1 position = getPosition();
    document.AddMember("positionXMeters", Value().SetDouble(position[0]), allocator);
    document.AddMember("positionYMeters", Value().SetDouble(position[1]), allocator);
    document.AddMember("positionZMeters", Value().SetDouble(position[2]), allocator);

    // TODO: write accessors/mutators of camera class that deal with normalized coefficients
    double normalizedFocalLength = getPixelResolution().height / getFocalLength();
    Value normalizedRadialDistortionCoefficients(Type::kArrayType);
    vector<double> radialDistortionCoefficients = getRadialDistortion();
    for (int k = 0; k < radialDistortionCoefficients.size(); k++)
    {
        double normalizedCoefficient = radialDistortionCoefficients[k] * pow(normalizedFocalLength, 2.0 * (k + 1));
        normalizedRadialDistortionCoefficients.PushBack(Value().SetDouble(normalizedCoefficient), allocator);
    }
    document.AddMember("normalizedRadialDistortionCoefficients", normalizedRadialDistortionCoefficients, allocator);

    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    document.Accept(writer);

    return buffer.GetString();
}

Camera &Camera::fromJSONString(string JSONString)
{
    Document document;
    StringStream stream(JSONString.c_str());
    document.ParseStream(stream);

    cv::Size sensorResolution;
    Value::ConstMemberIterator iterator = document.FindMember("sensorResolutionWidthPixels");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        sensorResolution.width = iterator->value.GetDouble();
    }
    else
    {
        // TODO: log the problem
        return *this;
    }
    iterator = document.FindMember("sensorResolutionHeightPixels");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        sensorResolution.height = iterator->value.GetDouble();
    }
    else
    {
        // TODO: log the problem
        return *this;
    }

    this->setPixelResolution(sensorResolution);

    double horizontalFOV;
    iterator = document.FindMember("horizontalFieldOfViewDegrees");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        horizontalFOV = iterator->value.GetDouble() * CV_PI / 180.0;
    }
    else
    {
        // TODO: log the problem
        return *this;
    }

    this->setHorizontalFieldOfView(horizontalFOV);

    Vector3x1 rotationAngles;
    iterator = document.FindMember("panDegrees");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        rotationAngles[0] = iterator->value.GetDouble() * CV_PI / 180.0;
    }
    else
    {
        // TODO: log the problem
        return *this;
    }
    iterator = document.FindMember("tiltDegrees");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        rotationAngles[1] = iterator->value.GetDouble() * CV_PI / 180.0;
    }
    else
    {
        // TODO: log the problem
        return *this;
    }
    iterator = document.FindMember("rollDegrees");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        rotationAngles[2] = iterator->value.GetDouble() * CV_PI / 180.0;
    }
    else
    {
        // TODO: log the problem
        return *this;
    }

    this->setPanTiltRoll(rotationAngles);

    Vector3x1 position;
    iterator = document.FindMember("positionXMeters");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        position[0] = iterator->value.GetDouble();
    }
    else
    {
        // TODO: log the problem
        return *this;
    }
    iterator = document.FindMember("positionYMeters");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        position[1] = iterator->value.GetDouble();
    }
    else
    {
        // TODO: log the problem
        return *this;
    }
    iterator = document.FindMember("positionZMeters");
    if (iterator != document.MemberEnd() && iterator->value.IsNumber())
    {
        position[2] = iterator->value.GetDouble();
    }
    else
    {
        // TODO: log the problem
        return *this;
    }

    this->setPosition(position);

    vector<double> radialDistortionCoefficients(3, 0.0);
    iterator = document.FindMember("normalizedRadialDistortionCoefficients");
    if (iterator != document.MemberEnd() && iterator->value.IsArray() && iterator->value.Size() <= 3)
    {
        double normalizedFocalLength = getPixelResolution().height / getFocalLength();
        for (int k = 0; k < iterator->value.Size(); k++)
        {
            if (iterator->value[k].IsNumber())
            {
                radialDistortionCoefficients[k] = iterator->value[k].GetDouble() / pow(normalizedFocalLength, 2.0 * (k + 1));
            }
            else
            {
                // TODO: log the problem
                return *this;
            }
        }
    }
    else
    {
        // TODO: log the problem
        return *this;
    }

    this->setRadialDistortion(radialDistortionCoefficients);

    return *this;
}
