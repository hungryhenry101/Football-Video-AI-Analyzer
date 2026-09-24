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


#include "PointExtractor.h"
#include <random>

void PointExtractor::setMask(const cv::Mat &mask)
{
    _currMask = mask.clone();
    if (_scaling == 1. && mask.rows != 540)
    {
        _scaling = 540. / mask.rows;
        _radius = round(_radius / _scaling);
        std::cout << "Changed support radius to " << _radius << std::endl;
    }

    _hasRun = false;
}

void PointExtractor::run()
{
    if (_hasRun || _currMask.empty())
        return;

    cv::Mat idx;
    double min;
    double max;
    cv::minMaxLoc(_currMask, &min, &max);

    min = 100;
    cv::Range range(min, max + 1);

    std::mt19937 generator{std::random_device{}()};
    std::map<int, std::vector<cv::Point>> new_points;

    for (int r = range.start; r < range.end; r++)
    {
        cv::Mat idx(_currMask == r);
        if (cv::sum(idx)[0] == 0 || r > 200)
            continue;

        std::vector<cv::Point> points;
        std::vector<cv::Point> finalPoints;
        cv::findNonZero(idx, points);

        if (points.empty())
            continue;

        std::vector<cv::Point> previous_points;

        // Reuse of the previous frame's points is only valid when the points
        // are in the same coordinate space as the current mask. When the mask
        // is not 540 rows tall the extracted points are stored scaled by
        // _scaling (see the end of run()), so reusing them against the
        // unscaled mask would index out of bounds (and silently corrupt the
        // extraction in NDEBUG builds). Fall back to the random-seed path.
        if (_scaling == 1. && _points.find(r) != _points.end() && !_points.at(r).empty())
        {
            previous_points = std::vector<cv::Point>(_points.at(r));
        }

        while (points.size() > 0) // && (double)initialPointSize*0.8/(pow(radius,2)) > finalPoints.size() )
        {
            cv::Point2d candidate;
            cv::Point picked;

            if (!previous_points.empty())
            {
                cv::Point pt = previous_points[previous_points.size() - 1];
                previous_points.pop_back();
                candidate = cv::Point2d(pt.x, pt.y);
                picked = cv::Point(pt);
                if (_currMask.at<unsigned char>(picked.y, picked.x) != r)
                {
                    continue;
                }
            }
            else
            {
                std::vector<int> weights;
                weights.assign(points.size(), 1);
                std::discrete_distribution<int> distribution(weights.begin(),
                                                             weights.end());

                int pick = distribution(generator);
                candidate = cv::Point2d(points[pick].x, points[pick].y);
                picked = cv::Point(points[pick]);
            }

            double dist = 10.;
            while (dist > 1.)
            {

                cv::Point2d center = getSupportCenter(candidate, idx);
                dist = sqrt(pow(center.x - candidate.x, 2) +
                            pow(center.y - candidate.y, 2));
                candidate = center;
            }
            if (candidate != cv::Point2d(0, 0))
            {
                cv::Point p(round(candidate.x), round(candidate.y));
                finalPoints.push_back(p);
                cv::circle(idx, p, 20, cv::Scalar({0, 0, 0}), -1);
            }
            else
            {
                cv::circle(idx, picked, 20, cv::Scalar({0, 0, 0}), -1);
            }
            int nb_deleted = 0;
            std::vector<cv::Point> newPoints = points;
            for (int i = 0; i < points.size(); i++)
            {
                if ((int)idx.at<unsigned char>(points[i].y, points[i].x) == 0)
                {
                    newPoints.erase(newPoints.begin() + i - nb_deleted);
                    nb_deleted++;
                }
            }
            points = newPoints;
        }
        new_points.insert(std::pair<int, std::vector<cv::Point>>(r, finalPoints));
    }
    _hasRun = true;
    if (_scaling != 1.)
    {
        for (auto &pointsId2Vector : new_points)
        {
            for (auto &point : pointsId2Vector.second)
            {
                point *= _scaling;
            }
        }
    }
    _points = new_points;
}

void PointExtractor::getExtractedPoints(
    std::map<int, std::vector<cv::Point>> &map)
{
    if (!_hasRun && !_currMask.empty())
        run();
    map = _points;
}

cv::Point2d
PointExtractor::getSupportCenter(cv::Point2d initialPoint, cv::Mat &img)
{
    int x = initialPoint.x;
    int y = initialPoint.y;
    double centerx = 0.;
    double centery = 0.;
    int supportingPixNb = 0;
    int istart = x - _radius, iend = x + _radius, jstart = y - _radius, jend = y + _radius;

    if (istart < 0)
        istart = 0;
    if (iend >= img.cols)
        iend = img.cols - 1;
    if (jstart < 0)
        jstart = 0;
    if (jend >= img.rows)
        jend = img.rows - 1;

    double ksum = 0;

    for (int i = istart; i <= iend; i++)
    {
        for (int j = jstart; j <= jend; j++)
        {
            double dist = sqrt(pow((i - x), 2) + pow(j - y, 2));
            if (dist <= _radius && (int)img.at<unsigned char>(j, i) != 0)
            {
                double kxy = 1 / (2 * M_PI * pow(_radius, 2)) * exp(-pow(dist, 2) / (2 * pow(_radius, 2)));
                ksum += kxy;
                centerx += (double)kxy * i;
                centery += (double)kxy * j;
                supportingPixNb++;
            }
        }
    }
    if (supportingPixNb < (int)(CV_PI * pow(_radius, 2) * _minSupport))
        return cv::Point2d(0, 0);
    centerx /= ksum;
    centery /= ksum;
    return cv::Point2d(centerx, centery);
}
