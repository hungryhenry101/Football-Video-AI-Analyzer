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
#include "KeypointDetectionModel.h"
#include "LineSegmentationModel.h"

#include "Camera.h"
#include "core.h"
#include <torch/mps.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options.hpp>

#include <filesystem>
#include <iostream>
namespace fs = boost::filesystem;
namespace po = boost::program_options;

#include <fstream>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

using namespace rapidjson;
const double MASK_TO_HD = 2.0;

std::vector<cv::Rect2d> loadHumanBboxes(std::string full_human_bboxes_path)
{
    std::ifstream jsonFile(full_human_bboxes_path);
    std::string jsonString((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
    rapidjson::Document document;
    document.Parse(jsonString.c_str());
    if (document.HasParseError())
    {
        std::cerr << "Error parsing " << full_human_bboxes_path << ": " << document.GetParseError() << std::endl;
    }
    auto documentObject = document.GetObject();
    auto bboxesArray = documentObject["bboxes"].GetArray();
    std::vector<cv::Rect2d> bboxes;
    for (const auto &bboxArray : bboxesArray)
    {
        cv::Rect2d bbox;
        bbox.x = bboxArray[0].GetDouble() - 100;
        bbox.y = bboxArray[1].GetDouble() - 100;
        bbox.width = bboxArray[2].GetDouble() + 100 - bbox.x;
        bbox.height = bboxArray[3].GetDouble() + 100 - bbox.y;
        bboxes.emplace_back(bbox);
    }
    return bboxes;
}

 std::pair<Point3D, double> loadMatchTripodInfo(const std::string &filePath)
{
    // Open the JSON file
    std::pair<Point3D, double> sphereInfo;

    std::ifstream ifs(filePath);
    if (!ifs.is_open())
    {
        std::cout << "Could not open the file: " << filePath << std::endl;
        return sphereInfo;
    }

    // Parse the JSON file
    IStreamWrapper isw(ifs);
    Document d;
    d.ParseStream(isw);

    // Check if the JSON is an object
    if (!d.IsObject())
    {
        std::cerr << "JSON is not an object!" << std::endl;
        return sphereInfo;
    }

    auto seqMember = d.GetObject();
    std::cout << "Loading tripod data" << std::endl;

    // Extract and print "sphere"
    if (seqMember.HasMember("sphere") && seqMember["sphere"].IsObject())
    {
        const Value &sphere = seqMember["sphere"];
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
        sphereInfo = std::make_pair(sphereCenter, radius);
    }
    return sphereInfo;
}

struct Arguments
{
    bool valid = false;
    bool output_frames = false;
    double posX;
    double posY;
    double posZ;
    std::string tvcalib;
    std::string keypoints;
    std::string frames_directory;
    std::string human_bboxes_directory;
    std::string tripod_info_file;
    std::string output_frames_directory;
    std::string output_json;
};
Arguments parseArgs(int argc, char *argv[])
{
    Arguments args;
    try
    {
        // Define the options
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "Show help message")
        ("X", po::value<double>(&args.posX)->default_value(0.), "Tripod X-coordinate")
        ("Y", po::value<double>(&args.posY)->default_value(90.), "Tripod Y-coordinate")
        ("Z", po::value<double>(&args.posZ)->default_value(-18.), "Tripod Z-coordinate")
        ("l", po::value<std::string>(&args.tvcalib)->default_value("/usr/share/broadtrack/tvcalib_model.pt"), "TVCalib Model")
        ("k", po::value<std::string>(&args.keypoints)->default_value("/usr/share/broadtrack/nbjw_keypoint_model.pt"), "Keypoint Model")
        ("f", po::value<std::string>(&args.frames_directory), "Frames directory")
        ("b", po::value<std::string>(&args.human_bboxes_directory), "Human bboxes directory")
        ("t", po::value<std::string>(&args.tripod_info_file), "Tripod info file")
        ("r", po::value<std::string>(&args.output_frames_directory), "Output frames directory")
        ("o", po::value<std::string>(&args.output_json)->default_value("./broadtrack.json"), "Output json");

        // Parse command-line arguments
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if(vm.count("r"))
        {
            args.output_frames = true;
        }
        // Display help if needed
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return args;
        }

        // Output the parsed values
        std::cout << "Provided arguments:" << std::endl;
        std::cout << " - frames_directory: " << args.frames_directory << std::endl;
        std::cout << " - human_bboxes_directory: " << args.human_bboxes_directory << std::endl;
        std::cout << " - tripod_info_file: " << args.tripod_info_file << std::endl;
        std::cout << " - default tripod position: " << args.posX << ", " << args.posY << ", " << args.posZ << std::endl;
        std::cout << " - output_frames_directory: " << args.output_frames_directory << std::endl;
        std::cout << " - output_json: " << args.output_json << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return args;
    }
    args.valid = true;
    return args;
}

struct config
{
    bool opticalFlow;
    bool radial_distortion;
    std::string positionMode;
};

int main(int argc, char *argv[])
{

    Arguments args = parseArgs(argc, argv);
    if (!args.valid)
        return 1;

    std::cout << "torch::mps::is_available() = " << (torch::mps::is_available() ? "true" : "false") << std::endl;

    std::vector<config> configs;

    config conf0;
    conf0.opticalFlow = true;
    conf0.radial_distortion = true;
    if(fs::exists(args.tripod_info_file))
    {
        conf0.positionMode = "soft";
    }
    else 
    {
        conf0.positionMode = "free";
    }
    
    configs.push_back(conf0);

    for (int c = 0; c < configs.size(); c++)
    {
        config conf = configs[c];
        std::cout << "Starting " << args.frames_directory << std::endl;

        std::string tripod_info_dict = args.tripod_info_file; 
        std::pair<Point3D, double> sphereInfo = loadMatchTripodInfo(tripod_info_dict);
        SoccerPitch3D soccerPitch3D;
        auto topLeftPitchCorner = soccerPitch3D.getPoint2D(SoccerPitch3D::TL_PITCH_CORNER);
        auto bottomRightPitchCorner = soccerPitch3D.getPoint2D(SoccerPitch3D::BR_PITCH_CORNER);

        std::string seqStr = args.frames_directory; 
        fs::path sequence(args.frames_directory);
        int frames_number = std::count_if(
            fs::directory_iterator(sequence),
            fs::directory_iterator(),
            static_cast<bool (*)(const fs::path &)>(fs::is_regular_file));

        CameraTracker tracker;
        LineSegmentationModel lineDetector(args.tvcalib,
                                           std::vector<double>({0.485, 0.456, 0.406}),
                                           std::vector<double>({0.229, 0.224, 0.225}));

        KeypointDetectionModel keypointDetector(args.keypoints);

        fs::path img_path = sequence / "000001.jpg"; 
        cv::Mat cv_img = cv::imread(img_path.string());
        int image_width = cv_img.cols;
        int image_height = cv_img.rows;
        fs::path output_json_path = args.output_json;

        fs::path output_img_dir = args.output_frames_directory;
        if (args.output_frames && !fs::is_directory(output_img_dir))
        {
            fs::create_directories(output_img_dir);
        }

        std::cout << "Analyzing sequence " << sequence << std::endl;
        /**** INIT ****/
        JsonCalibDict outputDict(output_json_path.generic_string(), true);

        if (cv_img.empty())
        {
            std::cerr << "Could not open or find the image: " << img_path << std::endl;
            continue;
        }

    
        Camera priorCamera;
        priorCamera.setPanTiltRoll(Vector3x1(0., 80. * CV_PI / 180., 0.));

        if (conf.positionMode != "free")
        {
            std::pair<Point3D, double> sphere = sphereInfo;
            tracker.setTripodInfo(std::get<0>(sphere), std::get<1>(sphere));
            Point3D position0(std::get<0>(sphere));
            priorCamera.setPosition(Vector3x1(position0.x(),
                                              position0.y(),
                                              position0.z()));
        }

        // No prior position
        if (conf.positionMode == "free")
        {
            priorCamera.setPosition(Vector3x1(args.posX,
                                              args.posY,
                                              args.posZ));
            tracker.setTripodInfo(Point3D(args.posX, args.posY, args.posZ), 0); // Point3D(0., 55., -12.)
        }

        priorCamera.setPixelResolution(cv::Size(image_width, image_height));
        priorCamera.setFocalLength(sqrt(pow(image_width, 2) + pow(image_height, 2)));
        tracker.setCamera(priorCamera);

        std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> pointDict;
        keypointDetector.computeKeypoints(cv_img, pointDict);
        if (pointDict.size() >= 2)
        {
            tracker.reinit(pointDict, 12);
            std::cout << tracker.getCamera().toJSONString() << std::endl;
        }
        else
        {
            std::cout << "NO INIT" << std::endl;
        }

        /**** TRACK ****/
        Chronometer chrono("iteration");
        Chronometer opticalFlowTimer("optical flow");

        cv::Mat previousCVFrame;
        std::vector<cv::Point2f> p0, p1;

        int nb_frames_tracking_lost = 0;
        for (int i = 1; i < frames_number; i++)
        {
            auto cam0 = tracker.getCamera();
            Camera cam;
            double score; // = tracker.evaluate(mask);
            bool reinit = false;

            std::ostringstream oss;
            oss << std::setw(6) << std::setfill('0') << i; 
            std::string frame_path = "/" + oss.str() + ".jpg";
            std::string full_img_path = seqStr + frame_path; 

            std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> pointDict;

            std::string full_output_img_path = (output_img_dir / ("t_" + oss.str() + ".jpg")).generic_string();
            std::string full_human_bboxes_path = seqStr + "/human-bboxes/" + oss.str() + ".json";

            cv::Mat currentCVFrame = cv::imread(full_img_path);
            if (currentCVFrame.empty())
            {
                std::cout << "Could not open or find the image: " << full_img_path << std::endl;
                continue;
            }

            double scale = 0.5;

            /**** OPTICAL FLOW ****/
            std::vector<cv::Rect2d> bboxes;
            if (fs::exists(full_human_bboxes_path))
            {
                bboxes = loadHumanBboxes(full_human_bboxes_path);
            }

            std::vector<std::pair<Point3D, Point2D>> pitchProjections;
            if (conf.opticalFlow && !nb_frames_tracking_lost)
            {
                if (i == 1)
                {

                    for (int row = 50; row < image_height - 50; row += 50)
                    {
                        for (int col = 50; col < image_width - 50; col += 50)
                        {
                            p0.emplace_back(cv::Point2f(col, row));
                        }
                    }
                }
                else if (i > 1)
                {
                    opticalFlowTimer.tic();
                    cv::Mat curr;
                    cv::Mat prvs;
                    cvtColor(previousCVFrame, prvs, cv::COLOR_BGR2GRAY);
                    cvtColor(currentCVFrame, curr, cv::COLOR_BGR2GRAY);

                    cv::Mat flow(prvs.size(), CV_32FC2);
                    // optical flow
                    std::vector<uchar> status;
                    std::vector<float> err;
                    cv::TermCriteria criteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
                    cv::calcOpticalFlowPyrLK(prvs, curr, p0, p1, status, err, cv::Size(55, 55), 2, criteria);
                    opticalFlowTimer.toc();

                    std::vector<cv::Point2f> good_new;
                    for (uint i = 0; i < p0.size(); i++)
                    {
                        for (const auto &bbox : bboxes)
                        {
                            if (bbox.contains(p1[i]))
                            {
                                status[i] = 0;
                                break;
                            }
                        }

                        if (status[i] == 1)
                        {
                            auto point2D = Point2D(p1[i].x, p1[i].y);
                            auto point3D = soccerPitch3D.getSurface().intersection(cam0.getRay(Point2D(p0[i].x, p0[i].y)));
                            if (point3D.x() > topLeftPitchCorner.x() &&
                                point3D.x() < bottomRightPitchCorner.x() &&
                                point3D.y() > topLeftPitchCorner.y() &&
                                point3D.y() < bottomRightPitchCorner.y())
                            {
                                pitchProjections.emplace_back(point3D, point2D);
                                good_new.push_back(p1[i]);
                            }
                        }
                    }

                    // Update tracking points for the next frame
                    p0 = good_new;

                    // Reinitialize grid if too many points are lost
                    if (p0.size() < 10)
                    {
                        p0.clear();
                        for (int row = 50; row < image_height - 50; row += 50)
                        {
                            for (int col = 50; col < image_width - 50; col += 50)
                            {
                                p0.emplace_back(cv::Point2f(col, row));
                            }
                        }
                    }
                }
            }

            chrono.tic();

            /**** UPDATE ****/
            cam = tracker.getCamera();

            chrono.tic();

            cv::Mat mask = lineDetector.computeLineMask(currentCVFrame);
            chrono.toc("Compute semlines");

            if (conf.positionMode == "soft")
            {
                std::tie(score, cam) = tracker.update(mask, pitchProjections, true, false, conf.radial_distortion);
            }
            else if (conf.positionMode == "free")
            {
                std::tie(score, cam) = tracker.update(mask, pitchProjections, false, false, conf.radial_distortion);
            }
            else
            {
                std::cerr << "Wrong config mode: unknown positionMode " << conf.positionMode << std::endl;
            }

            if (score < 0.3)
            {
                nb_frames_tracking_lost += 1;
                std::cout << "Reinit needed " << std::endl;
                keypointDetector.computeKeypoints(currentCVFrame, pointDict);

                if (pointDict.size() >= 2)
                {

                    tracker.reinit(pointDict, 12);
                    double newScore = tracker.evaluate(mask);
                    if (newScore < score)
                    {
                        // reset previous camera parameters
                        tracker.setCamera(cam);
                    }
                    else
                    {
                        score = newScore;
                    }
                }
                if (nb_frames_tracking_lost > 5 && score < 0.2)
                {
                    std::cout << priorCamera.toJSONString() << std::endl;
                    tracker.setCamera(priorCamera);
                }
                reinit = true;
                std::vector<std::pair<Point3D, Point2D>> empty;

                if (conf.positionMode == "soft")
                {
                    std::tie(score, cam) = tracker.update(mask, empty, true, false, conf.radial_distortion, 100 * nb_frames_tracking_lost);
                }
                else if (conf.positionMode == "free")
                {

                    std::tie(score, cam) = tracker.update(mask, empty, true, false, conf.radial_distortion, 100 * nb_frames_tracking_lost);
                }
            }
            if (score > 0.5)
            {
                nb_frames_tracking_lost = 0;
            }

            int64_t time = chrono.toc("Score " + frame_path + " " + std::to_string(score));
            previousCVFrame = currentCVFrame;


            if (args.output_frames)
            {
                cv::Mat bgr = currentCVFrame.clone();
                cv::Mat cvResized;
                cv::resize(bgr, cvResized, cv::Size(), scale, scale);
                cam.setPixelResolution(cv::Size(cvResized.cols, cvResized.rows));
                cam.setFocalLength(cam.getFocalLength() * scale);
                cam.drawWireframe(soccerPitch3D, cvResized, cv::Scalar(255, 0, 0));
                cv::imwrite(full_output_img_path, cvResized);
            }

            outputDict.addCamera(full_img_path, cam, score, time, reinit);
        }
        outputDict.flush();
    }
}