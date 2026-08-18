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

#include "LineSegmentationModel.h"
#include "SoccerPitch3D.h"

#include <torch/mps.h>

#include <iostream>
#include <memory>

const std::map<int, SoccerPitch3D::LineID> conversionMap = {

    {1, SoccerPitch3D::LineID::L_PENALTY_AREA_B_SIDE}, //'Big rect. left bottom',
    {2, SoccerPitch3D::LineID::L_PENALTY_AREA_R_SIDE}, //'Big rect. left main',
    {3, SoccerPitch3D::LineID::L_PENALTY_AREA_T_SIDE}, //'Big rect. left top',
    {4, SoccerPitch3D::LineID::R_PENALTY_AREA_B_SIDE}, //'Big rect. right bottom',
    {5, SoccerPitch3D::LineID::R_PENALTY_AREA_L_SIDE}, //'Big rect. right main',
    {6, SoccerPitch3D::LineID::R_PENALTY_AREA_T_SIDE}, //'Big rect. right top',
    {7, SoccerPitch3D::LineID::CENTER_CIRCLE},         //'Circle central',
    {8, SoccerPitch3D::LineID::L_PENALTY_ARC},         //'Circle left',
    {9, SoccerPitch3D::LineID::R_PENALTY_ARC},         //'Circle right',
    {10, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal left crossbar',
    {11, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal left post left',
    {12, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal left post right',
    {13, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal right crossbar',
    {14, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal right post left',
    {15, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal right post right',
    {16, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Goal unknown',
    {17, SoccerPitch3D::LineID::UNDEFINED_LINE},       //'Line unknown',
    {18, SoccerPitch3D::LineID::HALFWAY_LINE},         //'Middle line',
    {19, SoccerPitch3D::LineID::B_TOUCH_LINE},         //'Side line bottom',
    {20, SoccerPitch3D::LineID::L_GOAL_LINE},          //'Side line left',
    {21, SoccerPitch3D::LineID::R_GOAL_LINE},          //'Side line right',
    {22, SoccerPitch3D::LineID::T_TOUCH_LINE},         //'Side line top',
    {23, SoccerPitch3D::LineID::L_GOAL_AREA_B_SIDE},   //'Small rect. left bottom',
    {24, SoccerPitch3D::LineID::L_GOAL_AREA_R_SIDE},   //'Small rect. left main',
    {25, SoccerPitch3D::LineID::L_GOAL_AREA_T_SIDE},   //'Small rect. left top',
    {26, SoccerPitch3D::LineID::R_GOAL_AREA_B_SIDE},   //'Small rect. right bottom',
    {27, SoccerPitch3D::LineID::R_GOAL_AREA_L_SIDE},   //'Small rect. right main',
    {28, SoccerPitch3D::LineID::R_GOAL_AREA_T_SIDE},   //'Small rect. right top'
};

void LineSegmentationModel::normalize(torch::Tensor &inputs, torch::DeviceType deviceType)
{
    if ((!_mean.defined()) || (!_std.defined()))
        throw std::logic_error("Normalization parameters not set");

    if ((_mean.dim() != 1) || (_mean.sizes()[0] != 3))
        throw std::invalid_argument("Mean values tensor has wrong size");

    if ((_std.dim() != 1) || (_std.sizes()[0] != 3))
        throw std::invalid_argument("Standard deviations tensor has wrong size");
    if (_std.eq(0).any().item().toBool())
        throw std::invalid_argument("Standard deviation equal to 0");

    auto means_ = _mean.to(deviceType);
    auto standardDeviations_ = _std.to(deviceType);

    /* Reshape tensors to match model input format, i.e. [B, C, W, H] */

    standardDeviations_ = standardDeviations_.unsqueeze(0);
    standardDeviations_ = standardDeviations_.unsqueeze(-1);
    standardDeviations_ = standardDeviations_.unsqueeze(-1);

    means_ = means_.unsqueeze(0);
    means_ = means_.unsqueeze(-1);
    means_ = means_.unsqueeze(-1);

    if ((!means_.defined()) || (!standardDeviations_.defined()))
        throw std::logic_error("Normalization parameters not set");

    inputs.sub_(means_).div_(standardDeviations_);
}

torch::Tensor LineSegmentationModel::convertImagesToInputs(std::vector<cv::Mat> const &imageBatch, torch::DeviceType deviceType)
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

    normalize(inputs, deviceType);

    return inputs;
}

LineSegmentationModel::LineSegmentationModel(std::string model_path, std::vector<double> mean, std::vector<double> std)
{
    torch::DeviceType deviceType = torch::DeviceType::CPU;
    // Check if MPS (Apple Metal) is available and move model to GPU if available
    if (torch::mps::is_available())
    {
        std::cout << "MPS is available! Moving model to GPU." << std::endl;
        deviceType = torch::DeviceType::MPS;
    }
    _model = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, deviceType));
    _mean = torch::tensor(mean);
    _std = torch::tensor(std);
}

cv::Mat LineSegmentationModel::computeLineMask(cv::Mat image)
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

    if (image.rows > image.cols)
    {
        new_width = 256;
        new_height = static_cast<int>(256.0 * image.rows / image.cols);
    }
    else
    {
        new_height = 256;
        new_width = static_cast<int>(256.0 * image.cols / image.rows);
    }
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    std::vector<cv::Mat> imageBatch = {resized_image};
    auto inputs = convertImagesToInputs(imageBatch, deviceType);
    std::vector<torch::IValue> inputValues;
    inputValues.push_back(inputs.toType(torch::kFloat32));

    // Run inference
    auto output = _model->forward(inputValues).toGenericDict();

    torch::Tensor result = output.at("out").toTensor();

    // Detach, squeeze, and find the argmax along dimension 0
    result = result.detach().squeeze(0).argmax(0);

    // Move the result to CPU and convert to numpy-equivalent format (uint8)
    result = result.to(torch::kCPU).to(torch::kUInt8);

    // Convert to OpenCV Mat
    cv::Mat result_mat(result.size(0), result.size(1), CV_8UC1);
    std::vector<int64_t> sizes = {result.size(0), result.size(1)};
    auto imageTensor = torch::from_blob(result_mat.data, sizes, torch::kUInt8);
    imageTensor.copy_(result.contiguous(), true);

    for (auto conversionItem : conversionMap)
    {
        result_mat.setTo(static_cast<int>(conversionItem.second), result_mat == conversionItem.first);
    }
    result_mat.setTo(0, result_mat == static_cast<int>(SoccerPitch3D::LineID::UNDEFINED_LINE));
    return result_mat;
}
