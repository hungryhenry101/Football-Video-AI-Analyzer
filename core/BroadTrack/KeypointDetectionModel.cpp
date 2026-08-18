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

#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/mps.h>

#include "core.h"

#include "SoccerPitch3D.h"

#include "KeypointDetectionModel.h"
#include <iostream>
#include <memory>

const std::map<int, SoccerPitch3D::PointID> kpConversionMap = {
    {50, SoccerPitch3D::PointID::CENTER_MARK},
    {44, SoccerPitch3D::PointID::L_PENALTY_MARK},
    {56, SoccerPitch3D::PointID::R_PENALTY_MARK},
    {0, SoccerPitch3D::PointID::TL_PITCH_CORNER},
    {27, SoccerPitch3D::PointID::BL_PITCH_CORNER},
    {2, SoccerPitch3D::PointID::TR_PITCH_CORNER},
    {29, SoccerPitch3D::PointID::BR_PITCH_CORNER},
    {3, SoccerPitch3D::PointID::L_PENALTY_AREA_TL_CORNER},
    {4, SoccerPitch3D::PointID::L_PENALTY_AREA_TR_CORNER},
    {23, SoccerPitch3D::PointID::L_PENALTY_AREA_BL_CORNER},
    {24, SoccerPitch3D::PointID::L_PENALTY_AREA_BR_CORNER},
    {5, SoccerPitch3D::PointID::R_PENALTY_AREA_TL_CORNER},
    {6, SoccerPitch3D::PointID::R_PENALTY_AREA_TR_CORNER},
    {25, SoccerPitch3D::PointID::R_PENALTY_AREA_BL_CORNER},
    {26, SoccerPitch3D::PointID::R_PENALTY_AREA_BR_CORNER},
    {7, SoccerPitch3D::PointID::L_GOAL_AREA_TL_CORNER},
    {8, SoccerPitch3D::PointID::L_GOAL_AREA_TR_CORNER},
    {19, SoccerPitch3D::PointID::L_GOAL_AREA_BL_CORNER},
    {20, SoccerPitch3D::PointID::L_GOAL_AREA_BR_CORNER},
    {9, SoccerPitch3D::PointID::R_GOAL_AREA_TL_CORNER},
    {10, SoccerPitch3D::PointID::R_GOAL_AREA_TR_CORNER},
    {21, SoccerPitch3D::PointID::R_GOAL_AREA_BL_CORNER},
    {22, SoccerPitch3D::PointID::R_GOAL_AREA_BR_CORNER},
    {1, SoccerPitch3D::PointID::T_TOUCH_AND_HALFWAY_LINES_INTERSECTION},
    {28, SoccerPitch3D::PointID::B_TOUCH_AND_HALFWAY_LINES_INTERSECTION},
    {31, SoccerPitch3D::PointID::T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION},
    {34, SoccerPitch3D::PointID::B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION},
    {30, SoccerPitch3D::PointID::TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION},
    {33, SoccerPitch3D::PointID::BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION},
    {32, SoccerPitch3D::PointID::TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION},
    {35, SoccerPitch3D::PointID::BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION}};

KeypointDetectionModel::KeypointDetectionModel(std::string model_path)
{
    torch::DeviceType deviceType = torch::DeviceType::CPU;
    // Check if MPS (Apple Metal) is available and move model to GPU if available
    if (torch::mps::is_available())
    {
        std::cout << "MPS is available! Moving model to GPU." << std::endl;
        deviceType = torch::DeviceType::MPS;
    }
    _model = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, deviceType));
}

ChannelKeypoints KeypointDetectionModel::getKeypointsFromHeatmapBatchMaxpool(
    const torch::Tensor &heatmap,
    int scale = 2,
    int maxKeypoints = 2,
    int minKeypointPixelDistance = 15,
    bool returnScores = true)
{
    // Validate input tensor dimensions (NxCxHxW)
    TORCH_CHECK(heatmap.dim() == 4, "Heatmap must have shape NxCxHxW");

    int batchSize = heatmap.size(0);
    int nChannels = heatmap.size(1);
    int height = heatmap.size(2);
    int width = heatmap.size(3);

    // Define kernel size and padding
    int kernel = minKeypointPixelDistance * 2 + 1;
    int pad = minKeypointPixelDistance;

    // Pad the heatmap to exclude border keypoints
    torch::Tensor paddedHeatmap = torch::nn::functional::pad(
        heatmap, torch::nn::functional::PadFuncOptions({pad, pad, pad, pad}).mode(torch::kConstant).value(1.0));

    // Max pooling to find local maxima
    torch::Tensor maxPooledHeatmap = torch::nn::functional::max_pool2d(
        paddedHeatmap, torch::nn::functional::MaxPool2dFuncOptions(kernel).stride(1).padding(0));

    // Find local maxima
    torch::Tensor localMaxima = maxPooledHeatmap == heatmap; // .narrow(2, pad, height).narrow(3, pad, width)

    // Mask non-local maxima values in the heatmap
    torch::Tensor filteredHeatmap = heatmap * localMaxima;

    // Extract top-k scores and their indices
    auto topkResult = torch::topk(filteredHeatmap.view({batchSize, nChannels, -1}), maxKeypoints, 2, true);
    torch::Tensor scores = std::get<0>(topkResult);  // Scores
    torch::Tensor indices = std::get<1>(topkResult); // Indices

    // Calculate (x, y) coordinates for indices
    torch::Tensor x = indices % width;
    torch::Tensor y = indices / width;
    torch::Tensor coords = torch::stack({x, y}, -1);

    // Move data to CPU and convert to standard containers
    auto coordsCpu = coords.to(torch::kCPU).contiguous();
    auto scoresCpu = scores.to(torch::kCPU).contiguous();
    auto coordsAccessor = coordsCpu.accessor<float, 4>();
    auto scoresAccessor = scoresCpu.accessor<float, 3>();

    // Prepare output container
    std::vector<ChannelKeypoints> output(batchSize, std::vector<std::vector<Keypoint>>(nChannels, std::vector<Keypoint>()));

    // Populate output keypoints
    for (int b = 0; b < batchSize; ++b)
    {
        for (int c = 0; c < nChannels; ++c)
        {
            for (int k = 0; k < maxKeypoints; ++k)
            {
                float x = coordsAccessor[b][c][k][0];
                float y = coordsAccessor[b][c][k][1];
                float score = scoresAccessor[b][c][k];
                if (returnScores)
                {
                    output[b][c].emplace_back(std::tuple(round(x * scale), round(y * scale), score));
                }
                else
                {
                    output[b][c].emplace_back(std::tuple(round(x * scale), round(y * scale), 0.0f));
                }
            }
        }
    }

    return output[0];
}

void KeypointDetectionModel::retrieveSemanticPoints(ChannelKeypoints detectedPoints, std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &outPointDict)
{
    for (int c = 0; c < detectedPoints.size(); c++)
    {
        if (!kpConversionMap.count(c))
        {
            continue;
        }
        auto spClass = kpConversionMap.at(c);
        std::vector<Point2D> pointsVec;

        for (Keypoint &kp : detectedPoints[c])
        {
            if (std::get<2>(kp) > 0.1)
            {
                Point2D point(
                    std::get<0>(kp),
                    std::get<1>(kp));
                pointsVec.push_back(point);
            }
        }
        if (!pointsVec.empty())
        {
            outPointDict.emplace_back(spClass, pointsVec);
        }
    }
}

torch::Tensor KeypointDetectionModel::convertImagesToInputs(std::vector<cv::Mat> const &imageBatch, torch::DeviceType deviceType)
{
    if (imageBatch.empty())
        throw std::invalid_argument("No image supplied");
    auto batchSize = imageBatch.size();

    if ((imageBatch[0].dims != 2) || (imageBatch[0].channels() != 3))
        throw std::invalid_argument("Not a BGR image batch");
    auto height = imageBatch[0].size[0];
    auto width = imageBatch[0].size[1];

    auto cvType = imageBatch[0].type();
    torch::Dtype dataType;
    switch (cvType)
    {
    case CV_8UC3:
        dataType = torch::kUInt8;
        break;

    case CV_32FC3:
        dataType = torch::kFloat32;
        break;

    default:
        throw std::invalid_argument("Unsupported image data type");
    }

    /* Create tensor of suitable dimensions and data type on target device */
    std::vector<int64_t> sizes = {(int64_t)batchSize, height, width, 3};
    auto options = torch::dtype(dataType).device(deviceType).requires_grad(false);
    auto inputs = torch::empty(sizes, options);

    /* Copy image batch contents to input tensor */
    int i = 0;
    for (auto image : imageBatch)
    {
        if ((image.dims != 2) ||
            (image.size[0] != height) ||
            (image.size[1] != width) ||
            (image.channels() != 3))
            throw std::invalid_argument("Inconsistent image batch");

        if (!image.isContinuous())
            throw std::invalid_argument("Non-contiguous image data");

        std::vector<int64_t> sizes = {height, width, 3};
        inputs[i++].copy_(torch::from_blob(image.data, sizes, dataType), true);
    }
    /* Switch from [B, H, W, C] to [B, C, H, W] format
     */
    inputs = inputs.permute({0, 3, 1, 2});

    /* Convert data type as needed */
    inputs = inputs.to(torch::kFloat32, true);
    inputs.div_(255);
    return inputs;
}

void KeypointDetectionModel::computeKeypoints(const cv::Mat &image,
                                              std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &outPointDict)
{
    torch::DeviceType deviceType = torch::DeviceType::CPU;
    // Check if MPS (Apple Metal) is available and move model to GPU if available
    if (torch::mps::is_available())
    {
        // std::cout << "MPS is available! Moving model to GPU." << std::endl;
        deviceType = torch::DeviceType::MPS;
        _model->to(deviceType);
    }
    cv::Mat resized_image;
    int new_height;
    int new_width;
    int smaller_dim = 540;
    if (image.rows > image.cols)
    {
        new_width = smaller_dim;
        new_height = static_cast<int>(smaller_dim * image.rows / image.cols);
    }
    else
    {
        new_height = smaller_dim;
        new_width = static_cast<int>(smaller_dim * image.cols / image.rows);
    }
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    std::vector<cv::Mat> imageBatch = {resized_image};
    auto inputs = convertImagesToInputs(imageBatch, deviceType);
    std::vector<torch::IValue> inputValues;
    inputValues.push_back(inputs.toType(torch::kFloat32));

    // Run inference
    torch::Tensor result = _model->forward(inputValues).toTensor();

    // Detach, squeeze, and find the argmax along dimension 0
    result = result.detach();
    result = result.to(torch::kCPU);
    auto keypoints = getKeypointsFromHeatmapBatchMaxpool(result);

    retrieveSemanticPoints(keypoints, outPointDict);
}
