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

#include "LineIoUScore.h"

using namespace cv;

typedef uint8_t rte_v256u8_t __attribute__((vector_size(32), aligned(32)));
/**
 * 256 bits vector size to use with unsigned 64 bits elements.
 *
 * a = (rte_v256u64_t){ a0, a1, a2, a3 }
 */
typedef uint64_t rte_v256u64_t __attribute__((vector_size(32), aligned(32)));
typedef union ptr
{
    uint8_t *u8;
    rte_v256u8_t *v256_u8;
    rte_v256u64_t *v256_u64;
} ptr;

double fastComputeIoU(const Mat_<uchar> &A, const Mat_<uchar> &B)
{
    Mat_<uchar> intersection(A.size());
    Mat_<uchar> union_(A.size());

    const uint8_t *from1 = A.data;
    const uint8_t *from2 = B.data;

    int numElemsA = A.cols * A.rows;
    int complement32A = (32 - numElemsA % 32) % 32;
    std::vector<uint8_t> dataA(static_cast<unsigned long>(numElemsA + 62));
    uint8_t *dataAPtr = &dataA[0];
    // We need aligned memory
    while ((uint64_t)dataAPtr % 32 != 0)
    {
        dataAPtr++;
    }
    memcpy(dataAPtr, from1, static_cast<size_t>(numElemsA));
    memset(&dataAPtr[numElemsA], 0, complement32A);

    int numElemsB = B.cols * B.rows;
    int complement32B = (32 - numElemsB % 32) % 32;
    std::vector<uint8_t> dataB(static_cast<unsigned long>(numElemsB + 62));
    uint8_t *dataBPtr = &dataB[0];
    while ((uint64_t)dataBPtr % 32 != 0)
    {
        dataBPtr++;
    }
    memcpy(dataBPtr, from2, static_cast<size_t>(numElemsB));
    memset(&dataBPtr[numElemsB], 0, complement32B);

    int width = A.size().width;
    int height = A.size().height;

    ptr a, b;
    a.u8 = dataAPtr;
    b.u8 = dataBPtr;

    rte_v256u8_t valOR, valAND;

    ptr orPTr, andPtr;
    orPTr.v256_u8 = &valOR;
    andPtr.v256_u8 = &valAND;

    uint32_t countOR = 0;
    uint32_t countAND = 0;

    auto before = std::chrono::steady_clock::now();

    for (int u = 0; u < (numElemsA + complement32A) / 32; u++)
    {
        *orPTr.v256_u64 = *a.v256_u64 | *b.v256_u64;
        *andPtr.v256_u64 = *a.v256_u64++ & *b.v256_u64++;

        for (int i = 0; i < 32; ++i)
        {
            countOR += orPTr.u8[i];
            countAND += andPtr.u8[i];
        }
    }

    double IoU = (double)countAND / (double)countOR;

    return IoU;
}

double LineIoUScore::evaluateFast(const Matrix3x3 &H,
                                  double pitchLength,
                                  double pitchWidth)
{
    Mat_<uchar> cvRawLineMask(_rawLineMask.rows, _rawLineMask.cols, _rawLineMask.data);
    Mat_<uchar> cvTestLineMap(_rawLineMask.rows, _rawLineMask.cols);

    cv::Scalar white(255, 255, 255);
    cvTestLineMap.setTo(Scalar(0));
    SoccerPitch3D pitch(pitchLength, pitchWidth);

    auto polylines2D = pitch.getWireframe2D(1.);
    for (auto polyline : polylines2D)
    {
        bool first = true;
        Point2D prevPoint;
        for (auto point : polyline)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                Point2D projected = point.projected(H);
                if (
                    0 <= projected.x() && projected.x() < cvTestLineMap.cols &&
                    0 <= prevPoint.x() && prevPoint.x() < cvTestLineMap.cols &&
                    0 <= projected.y() && projected.y() < cvTestLineMap.rows &&
                    0 <= prevPoint.y() && prevPoint.y() < cvTestLineMap.rows)
                {
                    cv::line(cvTestLineMap,
                             cv::Point(prevPoint.x(), prevPoint.y()),
                             cv::Point(projected.x(), projected.y()),
                             white,
                             10, 8, 0);
                }
            }
            prevPoint = point.projected(H);
        }
    }
    return fastComputeIoU(cvRawLineMask, cvTestLineMap);
}
