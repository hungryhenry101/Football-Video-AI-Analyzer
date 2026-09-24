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


#include "JsonCalibDict.h"
#include "CameraTracker.h"
#include "Chronometer.h"
#include "LineSegmentationModel.h"
#include "KeypointDetectionModel.h"

#include "Camera.h"
#include <opencv2/opencv.hpp>
// #include <opencv2/cudaoptflow.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <filesystem>
#include <iostream>
namespace fs = boost::filesystem;

#include <iostream>
#include <fstream>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

using namespace rapidjson;
double MASK_TO_HD = 2.0;

SoccerPitch3D::LineID mySymmetricLineID(SoccerPitch3D::LineID lineID)
{
    switch (lineID)
    {
    case SoccerPitch3D::LineID::T_TOUCH_LINE:
        return SoccerPitch3D::LineID::B_TOUCH_LINE;
    case SoccerPitch3D::LineID::B_TOUCH_LINE:
        return SoccerPitch3D::LineID::T_TOUCH_LINE;
    case SoccerPitch3D::LineID::L_GOAL_LINE:
        return SoccerPitch3D::LineID::R_GOAL_LINE;
    case SoccerPitch3D::LineID::HALFWAY_LINE:
        return SoccerPitch3D::LineID::HALFWAY_LINE;
    case SoccerPitch3D::LineID::R_GOAL_LINE:
        return SoccerPitch3D::LineID::L_GOAL_LINE;
    case SoccerPitch3D::LineID::L_PENALTY_AREA_T_SIDE:
        return SoccerPitch3D::LineID::R_PENALTY_AREA_B_SIDE;
    case SoccerPitch3D::LineID::L_PENALTY_AREA_B_SIDE:
        return SoccerPitch3D::LineID::R_PENALTY_AREA_T_SIDE;
    case SoccerPitch3D::LineID::L_PENALTY_AREA_R_SIDE:
        return SoccerPitch3D::LineID::R_PENALTY_AREA_L_SIDE;
    case SoccerPitch3D::LineID::R_PENALTY_AREA_T_SIDE:
        return SoccerPitch3D::LineID::L_PENALTY_AREA_B_SIDE;
    case SoccerPitch3D::LineID::R_PENALTY_AREA_B_SIDE:
        return SoccerPitch3D::LineID::L_PENALTY_AREA_T_SIDE;
    case SoccerPitch3D::LineID::R_PENALTY_AREA_L_SIDE:
        return SoccerPitch3D::LineID::L_PENALTY_AREA_R_SIDE;
    case SoccerPitch3D::LineID::L_GOAL_AREA_T_SIDE:
        return SoccerPitch3D::LineID::R_GOAL_AREA_B_SIDE;
    case SoccerPitch3D::LineID::L_GOAL_AREA_B_SIDE:
        return SoccerPitch3D::LineID::R_GOAL_AREA_T_SIDE;
    case SoccerPitch3D::LineID::L_GOAL_AREA_R_SIDE:
        return SoccerPitch3D::LineID::R_GOAL_AREA_L_SIDE;
    case SoccerPitch3D::LineID::R_GOAL_AREA_T_SIDE:
        return SoccerPitch3D::LineID::L_GOAL_AREA_B_SIDE;
    case SoccerPitch3D::LineID::R_GOAL_AREA_B_SIDE:
        return SoccerPitch3D::LineID::L_GOAL_AREA_T_SIDE;
    case SoccerPitch3D::LineID::R_GOAL_AREA_L_SIDE:
        return SoccerPitch3D::LineID::L_GOAL_AREA_R_SIDE;
    case SoccerPitch3D::LineID::L_PENALTY_ARC:
        return SoccerPitch3D::LineID::R_PENALTY_ARC;
    case SoccerPitch3D::LineID::CENTER_CIRCLE:
        return SoccerPitch3D::LineID::CENTER_CIRCLE;
    case SoccerPitch3D::LineID::R_PENALTY_ARC:
        return SoccerPitch3D::LineID::L_PENALTY_ARC;
    default:
        return SoccerPitch3D::LineID::UNDEFINED_LINE;
    }
}

cv::Mat mirrorSoccerPitchLabels(const cv::Mat &mask)
{
    cv::Mat outputImage = mask.clone();
    std::set<uchar> uniqueColors(mask.begin<uchar>(), mask.end<uchar>());

    // Iterate over the gray map and apply the mapping using binary indexing
    for (const auto &gray : uniqueColors)
    {

        // Create a mask where the current gray value exists
        cv::Mat binaryMask = (mask == gray);
        if (gray > 171 || gray < 150)
        {

            outputImage.setTo(0, binaryMask);
            continue;
        }

        SoccerPitch3D::LineID newGrayValue = SoccerPitch3D::symmetricLineID(static_cast<SoccerPitch3D::LineID>(gray));
        // Apply the new gray value to the locations where the mask is true

        outputImage.setTo(newGrayValue, binaryMask);
    }

    return outputImage;
}

int loadSemPointsData(std::string filePath, std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &pointDict)
{

    std::ifstream ifs(filePath);
    if (!ifs.is_open())
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return 1;
    }

    // Parse the JSON file
    IStreamWrapper isw(ifs);
    Document document;
    document.ParseStream(isw);

    // Check if parsing succeeded
    if (document.HasParseError())
    {
        std::cerr << "Error parsing JSON file" << std::endl;
        return 1;
    }

    // Iterate over all members of the JSON object
    for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin(); itr != document.MemberEnd(); ++itr)
    {
        std::string key = itr->name.GetString();
        const rapidjson::Value &array = itr->value;
        SoccerPitch3D::PointID id;
        bool found = false;
        for (auto &point : SoccerPitch3D::POINT_ID_DESCRIPTIONS)
        {
            if (key == point.second)
            {
                id = point.first;
                found = true;
                break;
            }
        }
        if (!found)
        {
            // std::cout<<"Unrecognized key "<<key<<std::endl;
            continue;
        }

        std::vector<Point2D> semPoints;

        if (array.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < array.Size(); i++)
            {
                const rapidjson::Value &innerArray = array[i];
                if (innerArray.IsArray() && innerArray.Size() == 2)
                {
                    int x = innerArray[0].GetInt();
                    int y = innerArray[1].GetInt();
                    Point2D point2D(y * MASK_TO_HD,
                                    x * MASK_TO_HD);
                    semPoints.push_back(point2D);
                }
            }
        }

        if (semPoints.size())
        {
            pointDict.push_back(std::make_pair(id, semPoints));
        }
    }
    return 0;
}

std::tuple<std::map<std::string, std::pair<Point3D, double>>, std::map<std::string, Point3D>> loadMatchTripodInfo(const std::string &filePath)
{
    // Open the JSON file
    std::map<std::string, std::pair<Point3D, double>> spheresInfo;
    std::map<std::string, Point3D> intersections;

    std::ifstream ifs(filePath);
    if (!ifs.is_open())
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return std::tie(spheresInfo, intersections);
    }

    // Parse the JSON file
    IStreamWrapper isw(ifs);
    Document d;
    d.ParseStream(isw);

    // Check if the JSON is an object
    if (!d.IsObject())
    {
        std::cerr << "JSON is not an object!" << std::endl;
        return std::tie(spheresInfo, intersections);
    }

    // Iterate over the JSON object
    for (auto &seqMember : d.GetObject())
    {
        std::cout << "SEQ: " << seqMember.name.GetString() << std::endl;
        std::string sequence = seqMember.name.GetString();
        if (!seqMember.value.IsObject())
        {
            std::cerr << "Expected object for SEQ: " << seqMember.name.GetString() << std::endl;
            continue;
        }

        const Value &seqData = seqMember.value;

        // Extract and print "mean_intersection_axis"
        if (seqData.HasMember("mean_intersection_axis") && seqData["mean_intersection_axis"].IsArray())
        {
            std::cout << "  mean_intersection_axis: [";
            Point3D meanIntersection;
            for (SizeType i = 0; i < seqData["mean_intersection_axis"].Size(); ++i)
            {
                std::cout << seqData["mean_intersection_axis"][i].GetDouble();
                if (i < seqData["mean_intersection_axis"].Size() - 1)
                    std::cout << ", ";
                meanIntersection[i] = seqData["mean_intersection_axis"][i].GetDouble();
            }
            std::cout << "]" << std::endl;
            intersections.emplace(std::make_pair(sequence, meanIntersection));
        }

        // Extract and print "sphere"
        if (seqData.HasMember("sphere") && seqData["sphere"].IsObject())
        {
            const Value &sphere = seqData["sphere"];
            Point3D sphereCenter;
            if (sphere.HasMember("center") && sphere["center"].IsArray())
            {
                std::cout << "  sphere center: [";
                for (SizeType i = 0; i < sphere["center"].Size(); ++i)
                {
                    std::cout << sphere["center"][i].GetDouble();
                    if (i < sphere["center"].Size() - 1)
                        std::cout << ", ";
                    sphereCenter[i] = sphere["center"][i].GetDouble();
                }
                std::cout << "]" << std::endl;
            }
            double radius = 0.0;
            if (sphere.HasMember("radius") && sphere["radius"].IsDouble())
            {
                std::cout << "  sphere radius: " << sphere["radius"].GetDouble() << std::endl;
                radius = sphere["radius"].GetDouble();
            }
            spheresInfo.emplace(std::make_pair(sequence, std::make_pair(sphereCenter, radius)));
        }
    }

    return std::tuple(spheresInfo, intersections);
}

struct config
{
    bool opticalFlow;
    bool radial_distortion;
    std::string positionMode;
    std::string output;
};

int main()
{
    std::cout << "Hello VSC" << std::endl;
    fs::path testDir = "/home/fmg/data/SN23/calibration/test/";

    config conf;
    conf.opticalFlow = false;
    conf.radial_distortion = true;
    conf.positionMode = "free"; //"hard"; //"PTZ"; // "free";
    conf.output = "reinit_final_line_sym.json";

    SoccerPitch3D soccerPitch3D;
    auto topLeftPitchCorner = soccerPitch3D.getPoint2D(SoccerPitch3D::TL_PITCH_CORNER);
    auto bottomRightPitchCorner = soccerPitch3D.getPoint2D(SoccerPitch3D::BR_PITCH_CORNER);
    fs::path output_json_path = fs::path(testDir).parent_path() / conf.output; //
    JsonCalibDict outputDict = JsonCalibDict(output_json_path.generic_string(), true);

    Camera cam, cam16ML, cam16MR, camTactical;
    double score, score16ML, score16MR, scoreTactical;

    CameraTracker tracker;
    CameraTracker right16mtracker;
    CameraTracker left16mtracker;
    CameraTracker tacticalTracker;

    int image_width = 1920;
    int image_height = 1080;

    Camera priorCamera;
    priorCamera.setPosition(Vector3x1(0.,
                                      55.,
                                      -12.));
    tracker.setTripodInfo(Point3D(0., 55., -12.), 0);
    priorCamera.setPixelResolution(cv::Size(image_width, image_height));
    priorCamera.setFocalLength(sqrt(pow(image_width, 2) + pow(image_height, 2)));
    tracker.setCamera(priorCamera);

    Camera priorCamera16ML;
    priorCamera16ML.setPosition(Vector3x1(-36.,
                                          55.,
                                          -12.));
    left16mtracker.setTripodInfo(Point3D(-36., 55., -12.), 0);
    priorCamera16ML.setPixelResolution(cv::Size(image_width, image_height));
    priorCamera16ML.setFocalLength(sqrt(pow(image_width, 2) + pow(image_height, 2)));
    left16mtracker.setCamera(priorCamera16ML);

    Camera priorCamera16MR;
    priorCamera16MR.setPosition(Vector3x1(36.,
                                          55.,
                                          -12.));
    right16mtracker.setTripodInfo(Point3D(36., 55., -12.), 0);
    priorCamera16MR.setPixelResolution(cv::Size(image_width, image_height));
    priorCamera16MR.setFocalLength(sqrt(pow(image_width, 2) + pow(image_height, 2)));
    right16mtracker.setCamera(priorCamera16MR);

    Camera priorCameraTactical;
    priorCameraTactical.setPosition(Vector3x1(-65.,
                                              0.,
                                              -15.));
    tacticalTracker.setTripodInfo(Point3D(-65., 0., -15.), 0);
    priorCameraTactical.setPixelResolution(cv::Size(image_width, image_height));
    priorCameraTactical.setFocalLength(sqrt(pow(image_width, 2) + pow(image_height, 2)));
    tacticalTracker.setCamera(priorCameraTactical);

    LineSegmentationModel lineDetector("../tvcalib_model.pt",
                                       std::vector<double>({0.485, 0.456, 0.406}),
                                       std::vector<double>({0.229, 0.224, 0.225}));

    KeypointDetectionModel keypointDetector("../nbjw_keypoint_model.pt");

    for (int i = 0; i < 3141; i++)
    {
        std::cout << "Processing image " << i << " score : ";
        std::string testDirString = testDir.generic_string();
        std::ostringstream oss;
        oss << std::setw(5) << std::setfill('0') << i;
        std::string frame_path = oss.str() + ".jpg";
        std::string full_img_path = testDirString + frame_path;
        cv::Mat image = cv::imread(full_img_path);

        std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> pointDict;
        keypointDetector.computeKeypoints(image, pointDict);
        cv::Mat mask = lineDetector.computeLineMask(image);

        std::map<double, Camera> result;

        if (pointDict.size() >= 2)
        {
            std::vector<std::pair<Point3D, Point2D>> empty;

            tracker.reinit(pointDict, 12);
            std::tie(score, cam) = tracker.update(mask, empty);
            result.insert(std::make_pair(score, cam));

            right16mtracker.reinit(pointDict, 12);
            std::tie(score16MR, cam16MR) = right16mtracker.update(mask, empty);
            result.insert(std::make_pair(score16MR, cam16MR));

            left16mtracker.reinit(pointDict, 12);
            std::tie(score16ML, cam16ML) = left16mtracker.update(mask, empty);
            result.insert(std::make_pair(score16ML, cam16ML));

            tacticalTracker.reinit(pointDict, 12);
            // TVCalib hbg convention is right side camera, while NBJW is left, mirroring of semantic labels is needed
            cv::Mat newMask = mirrorSoccerPitchLabels(mask);
            std::tie(scoreTactical, camTactical) = tacticalTracker.update(newMask, empty);
            result.insert(std::make_pair(scoreTactical, camTactical));

            const auto &maxScoredCam = *result.rbegin();
            std::cout << maxScoredCam.first << std::endl;

            outputDict.addCamera(full_img_path, maxScoredCam.second, maxScoredCam.first, 0, true);
        }
        else
        {
            std::cout << "NO INIT" << std::endl;
        }
    }
    outputDict.flush();
}