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

#include "core.h"
#include <numeric>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Matrix3x3 Matrix3x3::identity()
{
    return Matrix3x3(1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0);
}

Matrix3x3::Matrix3x3()
{
    for (int i = 0; i < 9; i++)
        _data[i] = 0.0;
}

Matrix3x3::Matrix3x3(const double data[9])
{
    for (int i = 0; i < 9; i++)
        _data[i] = data[i];
}

Matrix3x3::Matrix3x3(const Matrix3x3 &other)
{
    for (int i = 0; i < 9; i++)
        _data[i] = other._data[i];
}

Matrix3x3::Matrix3x3(double m11, double m12, double m13,
                     double m21, double m22, double m23,
                     double m31, double m32, double m33)
{
    set(m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33);
}

Matrix3x3::Matrix3x3(const Vector3x1 &c1, const Vector3x1 &c2, const Vector3x1 &c3)
{
    set(c1[0], c2[0], c3[0],
        c1[1], c2[1], c3[1],
        c1[2], c2[2], c3[2]);
}

double *Matrix3x3::data()
{
    return _data;
}

const double *Matrix3x3::data() const
{
    return _data;
}

int Matrix3x3::nbElems() const
{
    return 9;
}

double &Matrix3x3::operator[](int index)
{
    return _data[index];
}

const double &Matrix3x3::operator[](int index) const
{
    return _data[index];
}

Matrix3x3 Matrix3x3::operator*(const Matrix3x3 &mat) const
{
    return multiply(mat);
}

Vector3x1 Matrix3x3::operator*(const Vector3x1 &vec) const
{
    return multiply(vec);
}

Point3D Matrix3x3::operator*(const Point3D &point3D) const
{
    return multiply(point3D);
}

Line3D Matrix3x3::operator*(const Line3D &line3D) const
{
    return multiply(line3D);
}

Matrix3x3 Matrix3x3::operator*(double scalar) const
{
    return multiply(scalar);
}

Matrix3x3 Matrix3x3::operator-() const
{
    return Matrix3x3((-Matx33d(this->data())).val);
}

double &Matrix3x3::operator()(int row, int col)
{
    return _data[row * 3 + col];
}

void Matrix3x3::set(const double data[9])
{
    for (int i = 0; i < 9; i++)
        _data[i] = data[i];
}

void Matrix3x3::set(double m11, double m12, double m13,
                    double m21, double m22, double m23,
                    double m31, double m32, double m33)
{
    _data[0] = m11;
    _data[1] = m12;
    _data[2] = m13;
    _data[3] = m21;
    _data[4] = m22;
    _data[5] = m23;
    _data[6] = m31;
    _data[7] = m32;
    _data[8] = m33;
}

double Matrix3x3::norm() const
{
    return sqrt(_data[0] * _data[0] + _data[1] * _data[1] + _data[2] * _data[2] +
                _data[3] * _data[3] + _data[4] * _data[4] + _data[5] * _data[5] +
                _data[6] * _data[6] + _data[7] * _data[7] + _data[8] * _data[8]);
}

Matrix3x3 Matrix3x3::inverse() const
{
    double *m = (double *)_data;

    Matx33d Minv = Matx33d(m).inv();
    return Matrix3x3(Minv.val);
}

Matrix3x3 Matrix3x3::transpose() const
{
    double *m = (double *)_data;

    Matx33d Mt = Matx33d(m).t();
    return Matrix3x3(Mt.val);
}

Matrix3x3 Matrix3x3::multiply(const Matrix3x3 &mat) const
{
    double *m1 = (double *)_data;
    double *m2 = (double *)mat._data;

    Matx33d M1M2 = Matx33d(m1) * Matx33d(m2);
    return Matrix3x3(M1M2.val);
}

Vector3x1 Matrix3x3::multiply(const Vector3x1 &vec) const
{
    double *m = (double *)_data;
    double *v = (double *)vec.data();

    Matx31d MV = Matx33d(m) * Matx31d(v);
    return Vector3x1(MV.val);
}

Point3D Matrix3x3::multiply(const Point3D &point3D) const
{
    Vector3x1 vec = multiply(Vector3x1(point3D.hx(), point3D.hy(), point3D.hz()));
    return Point3D(vec[0], vec[1], vec[2], point3D.w());
}

Line3D Matrix3x3::multiply(const Line3D &line3D) const
{
    Point3D p1 = multiply(line3D.parametricEquationPoint(0.0));
    Point3D p2 = multiply(line3D.parametricEquationPoint(1.0));

    return Line3D(p1, p2);
}

Matrix3x3 Matrix3x3::multiply(double scalar) const
{
    double *m = (double *)_data;

    Matx33d ms = scalar * Matx33d(m);
    return Matrix3x3(ms.val);
}

Vector3x1 Matrix3x3::column(int index) const
{
    return Vector3x1(_data[index], _data[3 + index], _data[6 + index]);
}

ostream &operator<<(ostream &out, const Matrix3x3 &mat)
{
    return out << "["
               << mat._data[0] << ", " << mat._data[1] << ", " << mat._data[2] << ";" << std::endl
               << mat._data[3] << ", " << mat._data[4] << ", " << mat._data[5] << ";" << std::endl
               << mat._data[6] << ", " << mat._data[7] << ", " << mat._data[8] << "]";
}

Vector4x1::Vector4x1()
{
    for (int i = 0; i < 4; i++)
        _data[i] = 0.;
}

Vector4x1::Vector4x1(const double data[4])
{
    for (int i = 0; i < 4; i++)
        _data[i] = data[i];
}

Vector4x1::Vector4x1(const Vector4x1 &other)
{
    for (int i = 0; i < 4; i++)
        _data[i] = other._data[i];
}

Vector4x1::Vector4x1(double v1, double v2, double v3, double v4)
{
    _data[0] = v1;
    _data[1] = v2;
    _data[2] = v3;
    _data[3] = v4;
}

double *Vector4x1::data()
{
    return _data;
}

const double *Vector4x1::data() const
{
    return _data;
}

int Vector4x1::nbElems() const
{
    return 4;
}

double &Vector4x1::operator[](int index)
{
    return _data[index];
}

const double &Vector4x1::operator[](int index) const
{
    return _data[index];
}

Vector4x1 Vector4x1::operator+(const Vector4x1 &other) const
{
    return Vector4x1(_data[0] + other._data[0], _data[1] + other._data[1],
                     _data[2] + other._data[2], _data[3] + other._data[3]);
}

Vector4x1 Vector4x1::operator-(const Vector4x1 &other) const
{
    return Vector4x1(_data[0] - other._data[0], _data[1] - other._data[1],
                     _data[2] - other._data[2], _data[3] - other._data[3]);
}

Vector4x1 Vector4x1::operator-() const
{
    return Vector4x1((-Matx41d(this->data())).val);
}

double Vector4x1::dotProduct(const Vector4x1 &vec) const
{
    return _data[0] * vec._data[0] + _data[1] * vec._data[1] + _data[2] * vec._data[2] + _data[3] * vec._data[3];
}

double Vector4x1::norm() const
{
    return sqrt(
        _data[0] * _data[0] + _data[1] * _data[1] + _data[2] * _data[2] + _data[3] * _data[3]);
}

void Vector4x1::scale(double s)
{
    _data[0] *= s;
    _data[1] *= s;
    _data[2] *= s;
    _data[3] *= s;
}

ostream &operator<<(ostream &out, const Vector4x1 &vec)
{
    return out << "[" << vec._data[0]
               << "; " << vec._data[1]
               << "; " << vec._data[2]
               << "; " << vec._data[3] << "]";
}

Vector3x1 Vector3x1::zero()
{
    return Vector3x1(0.0, 0.0, 0.0);
}

Vector3x1::Vector3x1()
{
    for (int i = 0; i < 3; i++)
        _data[i] = 0.;
}

Vector3x1::Vector3x1(const double data[3])
{
    for (int i = 0; i < 3; i++)
        _data[i] = data[i];
}

Vector3x1::Vector3x1(const Vector3x1 &other)
{
    for (int i = 0; i < 3; i++)
        _data[i] = other._data[i];
}

Vector3x1::Vector3x1(double v1, double v2, double v3)
{
    _data[0] = v1;
    _data[1] = v2;
    _data[2] = v3;
}

double *Vector3x1::data()
{
    return _data;
}

const double *Vector3x1::data() const
{
    return _data;
}

int Vector3x1::nbElems() const
{
    return 3;
}

double &Vector3x1::operator[](int index)
{
    return _data[index];
}

const double &Vector3x1::operator[](int index) const
{
    return _data[index];
}

Vector3x1 Vector3x1::operator+(const Vector3x1 &other) const
{
    return Vector3x1(_data[0] + other._data[0], _data[1] + other._data[1], _data[2] + other._data[2]);
}

Vector3x1 Vector3x1::operator-(const Vector3x1 &other) const
{
    return Vector3x1(_data[0] - other._data[0], _data[1] - other._data[1], _data[2] - other._data[2]);
}

Vector3x1 Vector3x1::operator-() const
{
    return Vector3x1((-Matx31d(this->data())).val);
}

Vector3x1 Vector3x1::crossProduct(const Vector3x1 &vec) const
{
    Vector3x1 cp;
    cp[0] = _data[1] * vec._data[2] - _data[2] * vec._data[1];
    cp[1] = _data[2] * vec._data[0] - _data[0] * vec._data[2];
    cp[2] = _data[0] * vec._data[1] - _data[1] * vec._data[0];
    return cp;
}

double Vector3x1::dotProduct(const Vector3x1 &vec) const
{
    return _data[0] * vec._data[0] + _data[1] * vec._data[1] + _data[2] * vec._data[2];
}

double Vector3x1::norm() const
{
    return sqrt(_data[0] * _data[0] + _data[1] * _data[1] + _data[2] * _data[2]);
}

void Vector3x1::scale(double s)
{
    _data[0] *= s;
    _data[1] *= s;
    _data[2] *= s;
}

Vector3x1 Vector3x1::operator*(double factor) const
{
    Vector3x1 v = *this;
    v.scale(factor);

    return v;
}

Vector3x1 Vector3x1::operator/(double factor) const
{
    Vector3x1 v = *this;
    v.scale(1.0 / factor);

    return v;
}

Vector3x1 Vector3x1::unitLength() const
{
    return *this / this->norm();
}

ostream &operator<<(ostream &out, const Vector3x1 &vec)
{
    return out << "[" << vec._data[0]
               << "; " << vec._data[1]
               << "; " << vec._data[2] << "]";
}

Line3D::Line3D()
{
    first.set(0.0, 0.0, 0.0);
    second = {0.0, 0.0, 1.0};
}

Line3D::Line3D(const Point3D &firstPoint, const Point3D &secondPoint)
{
    if (secondPoint.atInfinity())
    {
        first = secondPoint.normalized();
    }
    else
    {
        first = firstPoint.normalized();
    }

    Point3D A = firstPoint.normalized();
    Point3D B = secondPoint.normalized();

    if (A.atInfinity())
    {
        A.set(A.hx(), A.hy(), A.hz(), 1.0);
    }

    if (B.atInfinity())
    {
        B.set(B.hx(), B.hy(), B.hz(), 1.0);
    }

    Vector3x1 slope(B.x() - A.x(), B.y() - A.y(), B.z() - A.z());
    double slopeNorm = slope.norm();
    if (slopeNorm < numeric_limits<double>::epsilon())
    {
        second = {0.0, 0.0, 1.0};
    }
    else
    {
        second = slope;
        second.scale(1.0 / slopeNorm);
    }
}

Line3D::Line3D(const Point3D &parametricEquationPoint, const Vector3x1 &parametricEquationSlope)
{
    first = parametricEquationPoint;
    double slopeNorm = parametricEquationSlope.norm();
    if (slopeNorm < numeric_limits<double>::epsilon())
    {
        second = {0.0, 0.0, 1.0};
    }
    else
    {
        second = parametricEquationSlope;
        second.scale(1.0 / slopeNorm);
    }
}

Line3D::Line3D(const Line3D &other)
{
    first = other.first;
    second = other.second;
}

Point3D Line3D::parametricEquationPoint(double t) const
{
    if (first.atInfinity())
    {
        return Point3D(first.hx() + t * second[0], first.hy() + t * second[1], first.hz() + t * second[2], 0.0);
    }

    return Point3D(first.x() + t * second[0], first.y() + t * second[1], first.z() + t * second[2], 1.0);
}

Vector3x1 Line3D::parametricEquationSlope() const
{
    return second;
}

Point3D Line3D::intersection(const Plane3D &plane) const
{
    return plane.intersection(*this);
}

bool Line3D::atInfinity() const
{
    return first.atInfinity();
}

Point2D::Point2D()
{
    _data[0] = 0.0;
    _data[1] = 0.0;
    _data[2] = 1.0;
}

Point2D::Point2D(const double *data) : Vector3x1(data)
{
}

Point2D::Point2D(const Point2D &other) : Vector3x1(other)
{
}

Point2D::Point2D(const Vector3x1 &vec)
{
    _data[0] = vec[0];
    _data[1] = vec[1];
    _data[2] = vec[2];
}

Point2D::Point2D(double x, double y)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = 1.0;
}

Point2D::Point2D(double hx, double hy, double w)
{
    _data[0] = hx;
    _data[1] = hy;
    _data[2] = w;
}

Point2D::Point2D(const Line2D &l1, const Line2D &l2)
{
    Vector3x1 cp = l1.normalized().crossProduct(l2.normalized());
    _data[0] = cp[0];
    _data[1] = cp[1];
    _data[2] = cp[2];

    *this = this->normalized();
}

double Point2D::x() const
{
    return _data[0] / _data[2];
}

double Point2D::y() const
{
    return _data[1] / _data[2];
}

double Point2D::hx() const
{
    return _data[0];
}

double Point2D::hy() const
{
    return _data[1];
}

double Point2D::w() const
{
    return _data[2];
}

void Point2D::setX(double value)
{
    _data[0] = value * _data[2];
}

void Point2D::setY(double value)
{
    _data[1] = value * _data[2];
}

void Point2D::set(double hx, double hy, double w)
{
    _data[0] = hx;
    _data[1] = hy;
    _data[2] = w;
}

bool Point2D::operator==(const Point2D &other) const
{
    if ((_data[2] == 0.0 && other._data[2] != 0.0) ||
        (_data[2] != 0.0 && other._data[2] == 0.0))
        return false;

    double w1 = (_data[2] == 0.0) ? 1.0 : _data[2];
    double w2 = (other._data[2] == 0.0) ? 1.0 : other._data[2];

    return (_data[0] * w2 == other._data[0] * w1) &&
           (_data[1] * w2 == other._data[1] * w1);
}

bool Point2D::operator!=(const Point2D &other) const
{
    return !operator==(other);
}

// TODO: infinity case(s)
Point2D Point2D::operator+(const Point2D &other) const
{
    return Point2D(this->x() + other.x(), this->y() + other.y());
}

// TODO: infinity case(s)
Point2D Point2D::operator-(const Point2D &other) const
{
    return Point2D(this->x() - other.x(), this->y() - other.y());
}

Point2D Point2D::operator-() const
{
    return Point2D(-_data[0], -_data[1], _data[2]);
}

Point2D Point2D::operator*(double scale) const
{
    return Point2D(scale * _data[0], scale * _data[1], _data[2]);
}

// TODO: infinity case(s)
Point2D Point2D::operator/(double scale) const
{
    return Point2D(1.0 / scale * _data[0], 1.0 / scale * _data[1], _data[2]);
}

double Point2D::squaredNorm() const
{
    return x() * x() + y() * y();
};

double Point2D::norm() const
{
    return sqrt(squaredNorm());
}

void Point2D::scale(double s)
{
    _data[0] *= s;
    _data[1] *= s;
}

double Point2D::squaredDistance(const Point2D &point) const
{
    double dx = x() - point.x();
    double dy = y() - point.y();
    return dx * dx + dy * dy;
}

double Point2D::distance(const Point2D &point) const
{
    return sqrt(squaredDistance(point));
}

double Point2D::signedDistance(const Line2D &line) const
{
    if (line.atInfinity())
    {
        return this->atInfinity() ? 0.0 : numeric_limits<double>::infinity();
    }

    Line2D l = line.normalized();
    Point2D p = this->normalized();

    return l.a() * p.hx() + l.b() * p.hy() + l.c() * p.w();
}

double Point2D::distance(const Line2D &line) const
{
    return fabs(signedDistance(line));
}

double Point2D::distance(const LineSegment2D &lineSegment)
{
    Line2D line(lineSegment.start(), lineSegment.end());
    Point2D closestPointOnLine = this->projected(line);

    double ax = lineSegment.start().x();
    double ay = lineSegment.start().y();
    double bx = lineSegment.end().x();
    double by = lineSegment.end().y();
    double cx = closestPointOnLine.x();
    double cy = closestPointOnLine.y();

    Point2D closestPointOnLineSegment;
    double t = ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) /
               ((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    if (t < 0.0)
    {
        closestPointOnLineSegment = lineSegment.start();
    }
    else if (t > 1.0)
    {
        closestPointOnLineSegment = lineSegment.end();
    }
    else
    {
        closestPointOnLineSegment = closestPointOnLine;
    }

    return this->distance(closestPointOnLineSegment);
}

double Point2D::distance(const std::vector<Point2D> pointList)
{
    double minDist = numeric_limits<double>::infinity();
    for (const auto &point : pointList)
    {
        double dist = this->distance(point);
        if (dist < minDist)
        {
            minDist = dist;
        }
    }

    return minDist;
}

double Point2D::distance(const Polyline2D &polyline2D) const
{
    double minDist = numeric_limits<double>::infinity();
    for (int i = 0; i < polyline2D.size() - 1; i++)
    {
        Point2D closestPointOnLineSegment;

        Matx21d A(polyline2D[i].x(), polyline2D[i].y());
        Matx21d B(polyline2D[i + 1].x(), polyline2D[i + 1].y());
        Matx21d AB = B - A;
        double ABdotAB = AB.dot(AB);
        if (ABdotAB < numeric_limits<double>::epsilon())
        {
            closestPointOnLineSegment.set(A(0), A(1));
        }
        else
        {
            Matx21d P(this->x(), this->y());

            Matx21d AP = P - A;
            Matx21d C = A + AP.dot(AB) / ABdotAB * AB;
            Matx21d AC = C - A;

            double t = AC.dot(AB) / ABdotAB;
            if (t < 0.0)
            {
                closestPointOnLineSegment.set(A(0), A(1));
            }
            else if (t > 1.0)
            {
                closestPointOnLineSegment.set(B(0), B(1));
            }
            else
            {
                closestPointOnLineSegment.set(C(0), C(1));
            }
        }

        double dist = this->distance(closestPointOnLineSegment);

        if (dist < minDist)
        {
            minDist = dist;
        }
    }

    return minDist;
}

Point2D Point2D::normalized() const
{
    Point2D p(*this);

    for (int i = 0; i < 3; i++)
        if (fabs(p[i]) < numeric_limits<double>::epsilon())
            p[i] = 0.0;

    if (p.atInfinity())
    {
        double n = sqrt(p[0] * p[0] + p[1] * p[1]);

        if (n == 0)
            return p;

        if (p[0] < 0)
            n = -n;

        p[0] /= n;
        p[1] /= n;

        return p;
    }

    p[0] /= p[2];
    p[1] /= p[2];
    p[2] = 1;

    return p;
}

Point2D Point2D::projected(const Line2D &l) const
{
    if (!l.atInfinity())
    {
        double xp = (l.b() * (l.b() * x() - l.a() * y()) - l.a() * l.c()) / (l.a() * l.a() + l.b() * l.b());

        double yp = (l.a() * (-l.b() * x() + l.a() * y()) - l.b() * l.c()) / (l.a() * l.a() + l.b() * l.b());

        return Point2D(xp, yp);
    }
    else
    {
        return Point2D(x(), y(), 0.0);
    }
}

Point2D Point2D::projected(const Matrix3x3 &mat) const
{
    Matx31d xp = Matx33d(mat.data()) * Matx31d(_data);

    return Point2D(xp.val);
}

bool Point2D::isWithin(const cv::Size &size) const
{
    return x() >= 0.0 && x() < size.width && y() >= 0.0 && y() < size.height;
}

bool Point2D::atInfinity() const
{
    return _data[2] == 0;
}

Point3D::Point3D()
{
    _data[0] = 0.0;
    _data[1] = 0.0;
    _data[2] = 0.0;
    _data[3] = 1.0;
}

Point3D::Point3D(const double data[4]) : Vector4x1(data)
{
}

Point3D::Point3D(const Point3D &other) : Vector4x1(other)
{
}

Point3D::Point3D(const Vector4x1 &vec)
{
    _data[0] = vec[0];
    _data[1] = vec[1];
    _data[2] = vec[2];
    _data[3] = vec[3];
}

Point3D::Point3D(const Vector3x1 &vec)
{
    _data[0] = vec[0];
    _data[1] = vec[1];
    _data[2] = vec[2];
    _data[3] = 1.0;
}

Point3D::Point3D(double x, double y, double z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
    _data[3] = 1;
}

Point3D Point3D::operator+(const Point3D &other) const
{
    // TODO: Infinity cases
    return Point3D(this->x() + other.x(), this->y() + other.y(), this->z() + other.z());
}

Point3D Point3D::operator-(const Point3D &other) const
{
    // TODO: Infinity cases
    return Point3D(this->x() - other.x(), this->y() - other.y(), this->z() - other.z());
}

Point3D Point3D::operator-() const
{
    // TODO: Infinity cases
    return Point3D(-this->x(), -this->y(), -this->z());
}

Point3D Point3D::operator*(double scale) const
{
    return Point3D(scale * _data[0], scale * _data[1], scale * _data[2], _data[3]);
}

// TODO: infinity case(s)
Point3D Point3D::operator/(double scale) const
{
    return Point3D(1.0 / scale * _data[0], 1.0 / scale * _data[1], 1.0 / scale * _data[2], _data[3]);
}

Point3D::Point3D(double hx, double hy, double hz, double w)
{
    _data[0] = hx;
    _data[1] = hy;
    _data[2] = hz;
    _data[3] = w;
}

Point3D::Point3D(const Point2D &xy, double z)
{
    _data[0] = xy.x();
    _data[1] = xy.y();
    _data[2] = z;
    _data[3] = 1.0;
}

Point3D::Point3D(const std::vector<Line3D> &lines)
{
    Mat_<double> A(3 * lines.size(), 3);
    Mat_<double> b(3 * lines.size(), 1);
    Mat_<double> x(3, 1);

    for (int i = 0; i < lines.size(); i++)
    {
        auto a = lines[i].parametricEquationPoint();
        auto d = lines[i].parametricEquationSlope();
        auto ax = a.x();
        auto ay = a.y();
        auto az = a.z();
        auto dx = d[0];
        auto dy = d[1];
        auto dz = d[2];
        auto dx2 = dx * dx;
        auto dy2 = dy * dy;
        auto dz2 = dz * dz;
        auto n2 = dx2 + dy2 + dz2;

        int i1 = i * 2;
        int i2 = i * 2 + 1;
        int i3 = i * 2 + 2;

        A(i1, 0) = n2 - dx2;
        A(i1, 1) = -dx * dy;
        A(i1, 2) = -dx * dz;
        b(i1) = n2 * ax - ax * dx2 - ay * dx * dy - az * dx * dz;

        A(i2, 0) = -dx * dy;
        A(i2, 1) = n2 - dy2;
        A(i2, 2) = -dy * dz;
        b(i2) = n2 * ay - ax * dx * dy - ay * dy2 - az * dy * dz;

        A(i3, 0) = -dx * dz;
        A(i3, 1) = -dy * dz;
        A(i3, 2) = n2 - dz2;
        b(i3) = n2 * az - ax * dx * dz - ay * dy * dz - az * dz2;
    }

    bool solutionUsable = solve(A, b, x, DECOMP_SVD);
    if (solutionUsable)
    {
        set(x(0), x(1), x(2));
    }
}

double Point3D::x() const
{
    return _data[0] / _data[3];
}

double Point3D::y() const
{
    return _data[1] / _data[3];
}

double Point3D::z() const
{
    return _data[2] / _data[3];
}

double Point3D::hx() const
{
    return _data[0];
}

double Point3D::hy() const
{
    return _data[1];
}

double Point3D::hz() const
{
    return _data[2];
}

double Point3D::w() const
{
    return _data[3];
}

void Point3D::setX(double value)
{
    _data[0] = value * _data[3];
}

void Point3D::setY(double value)
{
    _data[1] = value * _data[3];
}

void Point3D::setZ(double value)
{
    _data[2] = value * _data[3];
}

void Point3D::set(double hx, double hy, double hz, double w)
{
    _data[0] = hx;
    _data[1] = hy;
    _data[2] = hz;
    _data[3] = w;
}

bool Point3D::operator==(const Point3D &other) const
{
    if ((_data[3] == 0.0 && other._data[3] != 0.0) ||
        (_data[3] != 0.0 && other._data[3] == 0.0))
        return false;

    double w1 = (_data[3] == 0.0) ? 1.0 : _data[3];
    double w2 = (other._data[3] == 0.0) ? 1.0 : other._data[3];

    return (_data[0] * w2 == other._data[0] * w1) &&
           (_data[1] * w2 == other._data[1] * w1) &&
           (_data[2] * w2 == other._data[2] * w1);
}

double Point3D::squaredNorm() const
{
    return x() * x() + y() * y() + z() * z();
}

double Point3D::norm() const
{
    return sqrt(squaredNorm());
}

void Point3D::scale(double s)
{
    _data[0] *= s;
    _data[1] *= s;
    _data[2] *= s;
}

double Point3D::squaredDistance(const Point3D &point) const
{
    double dx = x() - point.x();
    double dy = y() - point.y();
    double dz = z() - point.z();
    return dx * dx + dy * dy + dz * dz;
}

double Point3D::distance(const Point3D &point) const
{
    return sqrt(squaredDistance(point));
}

double Point3D::signedDistance(const Plane3D &plane) const
{
    if (plane.atInfinity())
    {
        return this->atInfinity() ? 0.0 : numeric_limits<double>::infinity();
    }

    Plane3D P = plane.normalized();
    Point3D p = this->normalized();

    return P.a() * p.hx() + P.b() * p.hy() + P.c() * p.hz() + P.d() * p.w();
}

double Point3D::distance(const Plane3D &plane) const
{
    return fabs(signedDistance(plane));
}

double Point3D::distance(const Polyline3D &polyline3D) const
{
    double minDist = numeric_limits<double>::infinity();
    for (int i = 0; i < polyline3D.size() - 1; i++)
    {
        Point3D closestPointOnLineSegment;

        Matx31d A(polyline3D[i].x(), polyline3D[i].y(), polyline3D[i].z());
        Matx31d B(polyline3D[i + 1].x(), polyline3D[i + 1].y(), polyline3D[i + 1].z());
        Matx31d AB = B - A;
        double ABdotAB = AB.dot(AB);
        if (ABdotAB < numeric_limits<double>::epsilon())
        {
            closestPointOnLineSegment.set(A(0), A(1), A(2));
        }
        else
        {
            Matx31d P(this->x(), this->y(), this->z());

            Matx31d AP = P - A;
            Matx31d C = A + AP.dot(AB) / ABdotAB * AB;
            Matx31d AC = C - A;

            double t = AC.dot(AB) / ABdotAB;
            if (t < 0.0)
            {
                closestPointOnLineSegment.set(A(0), A(1), A(2));
            }
            else if (t > 1.0)
            {
                closestPointOnLineSegment.set(B(0), B(1), B(2));
            }
            else
            {
                closestPointOnLineSegment.set(C(0), C(1), C(2));
            }
        }

        double dist = this->distance(closestPointOnLineSegment);

        if (dist < minDist)
        {
            minDist = dist;
        }
    }

    return minDist;
}

Point3D Point3D::normalized() const
{
    Point3D p(*this);

    for (int i = 0; i < 4; i++)
        if (fabs(p[i]) < numeric_limits<double>::epsilon())
            p[i] = 0.0;

    if (p.atInfinity())
    {
        double n = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);

        if (n == 0)
            return p;

        if (p[0] < 0)
            n = -n;

        p[0] /= n;
        p[1] /= n;
        p[2] /= n;

        return p;
    }

    p[0] /= p[3];
    p[1] /= p[3];
    p[2] /= p[3];
    p[3] = 1;

    return p;
}

Point2D Point3D::normalizedCentralProjection() const
{
    return Point2D(_data[0], _data[1], _data[2]);
}

Point2D Point3D::xy() const
{
    return Point2D(x(), y(), 1.0);
}

bool Point3D::atInfinity() const
{
    return _data[3] == 0.0;
}

Line2D Line2D::xAxisLine()
{
    return {0.0, 1.0, 0.0};
}

Line2D Line2D::yAxisLine()
{
    return {1.0, 0.0, 0.0};
}

Line2D Line2D::infinityLine()
{
    return {0.0, 0.0, 1.0};
}

Line2D::Line2D() : Vector3x1()
{
}

Line2D::Line2D(const double *data) : Vector3x1(data)
{
}

Line2D::Line2D(const Line2D &other) : Vector3x1(other)
{
}

Line2D::Line2D(const Vector3x1 &vec)
{
    _data[0] = vec[0];
    _data[1] = vec[1];
    _data[2] = vec[2];
}

Line2D::Line2D(double a, double b, double c)
{
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
}

Line2D::Line2D(double a, double b, const Point2D &p)
{
    _data[0] = a;
    _data[1] = b;
    _data[2] = -a * p.x() - b * p.y();
}

Line2D::Line2D(const Point2D &p1, const Point2D &p2)
{
    Vector3x1 cp = p1.normalized().crossProduct(p2.normalized());
    _data[0] = cp[0];
    _data[1] = cp[1];
    _data[2] = cp[2];

    *this = this->normalized();
}

Line2D::Line2D(const vector<Point2D> &points)
{

    vector<Point2f> cvPoints;
    for (int i = 0; i < points.size(); i++)
        cvPoints.emplace_back(Point2f((float)points[i].x(),
                                      (float)points[i].y()));

    Vec4f line;
    fitLine(cvPoints, line, DIST_L2, 0, 0.01, 0.01);
    double vx = line[0];
    double vy = line[1];
    double x0 = line[2];
    double y0 = line[3];

    double a = vy;
    double b = -vx;
    double c = -a * x0 - b * y0;

    _data[0] = a;
    _data[1] = b;
    _data[2] = c;

    *this = this->normalized();
}

Line2D::Line2D(const LineSegment2D &segment)
{
    *this = Line2D(segment.start(), segment.end());
}

double Line2D::a() const
{
    return _data[0];
}

double Line2D::b() const
{
    return _data[1];
}

double Line2D::c() const
{
    return _data[2];
}

void Line2D::set(double a, double b, double c)
{
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
}

bool Line2D::operator==(const Line2D &other) const
{
    if ((atInfinity() && !other.atInfinity()) || (!atInfinity() && other.atInfinity()))
    {
        return false;
    }

    if (atInfinity() && other.atInfinity())
    {
        return _data[2] == other._data[2];
    }

    Point2D intersection(*this, other);
    if (intersection == Point2D(0.0, 0.0, 0.0))
    {
        return true;
    }

    return false;
}

double Line2D::signedDistance(const Point2D &point) const
{
    return point.signedDistance(*this);
}

double Line2D::distance(const Point2D &point) const
{
    return point.distance(*this);
}

Line2D Line2D::normalized() const
{
    Line2D l(*this);

    for (int i = 0; i < 3; i++)
        if (fabs(l[i]) < numeric_limits<double>::epsilon())
            l[i] = 0.0;

    if (l.atInfinity())
    {
        l[2] = 1.0;
        return l;
    }

    double n = sqrt(l[0] * l[0] + l[1] * l[1]);
    if (l[0] < 0.0)
        n = -n;

    l[0] /= n;
    l[1] /= n;
    l[2] /= n;

    return l;
}

Line2D Line2D::projected(const Matrix3x3 &mat) const
{
    Matx31d lp = Matx33d(mat.data()).inv().t() * Matx31d(_data);

    return Line2D(lp.val);
}

Point2D Line2D::intersection(const Line2D &other) const
{
    return Point2D(*this, other);
}

double Line2D::angle(const Line2D &other) const
{
    auto a1 = this->a();
    auto b1 = this->b();
    auto a2 = other.a();
    auto b2 = other.b();
    return acos(fabs((a1 * a2 + b1 * b2) / (sqrt(a1 * a1 + b1 * b1) * sqrt(a2 * a2 + b2 * b2))));
}

Line2D Line2D::perpendicular(const Point2D &passingThroughPoint) const
{
    if (atInfinity() || passingThroughPoint.atInfinity())
    {
        return infinityLine();
    }
    return Line2D(-b(), a(), b() * passingThroughPoint.x() - a() * passingThroughPoint.y()).normalized();
}

bool Line2D::atInfinity() const
{
    return _data[0] == 0 && _data[1] == 0;
}

LineSegment2D::LineSegment2D(const Point2D &s, const Point2D &e) : _start(s), _end(e)
{
}

Point2D LineSegment2D::start() const
{
    return _start;
}

Point2D LineSegment2D::end() const
{
    return _end;
}

double LineSegment2D::squaredLength() const
{
    return _start.squaredDistance(_end);
}

double LineSegment2D::length() const
{
    return _start.distance(_end);
}

LineSegment2D LineSegment2D::projected(const Matrix3x3 &homography) const
{
    return LineSegment2D(_start.projected(homography), _end.projected(homography));
}

pair<Point2D, double> LineSegment2D::lineIntersection(const Line2D &line) const
{
    Point2D intersection(line, Line2D(*this));

    Point2D AB = _end - _start;
    Point2D AC = intersection - _start;

    double lengthNormalizedParameter = (AC.x() * AB.x() + AC.y() * AB.y()) / this->squaredLength();

    return make_pair(intersection, lengthNormalizedParameter);
}

Ellipse2D::Ellipse2D() : Matrix3x3()
{
}

Ellipse2D::Ellipse2D(const double *data) : Matrix3x3(data)
{
}

Ellipse2D::Ellipse2D(const Ellipse2D &other) : Matrix3x3(other)
{
}

Ellipse2D::Ellipse2D(double A, double B, double C, double D, double E, double F)
{
    _data[0] = A;
    _data[1] = B / 2.0;
    _data[2] = D / 2.0;
    _data[3] = B / 2.0;
    _data[4] = C;
    _data[5] = E / 2.0;
    _data[6] = D / 2.0;
    _data[7] = E / 2.0;
    _data[8] = F;
}

Ellipse2D::Ellipse2D(double a, double b, double xc, double yc, double theta)
{
    if (a < b)
        swap(a, b);

    double a2 = a * a;
    double b2 = b * b;
    double sint = sin(theta);
    double cost = cos(theta);
    double st2 = sint * sint;
    double ct2 = cost * cost;

    double A = a2 * st2 + b2 * ct2;
    double B = 2.0 * (b2 - a2) * sint * cost;
    double C = a2 * ct2 + b2 * st2;
    double D = -2.0 * A * xc - B * yc;
    double E = -B * xc - 2.0 * C * yc;
    double F = A * xc * xc + B * xc * yc + C * yc * yc - a2 * b2;

    _data[0] = A;
    _data[1] = B / 2.0;
    _data[2] = D / 2.0;
    _data[3] = B / 2.0;
    _data[4] = C;
    _data[5] = E / 2.0;
    _data[6] = D / 2.0;
    _data[7] = E / 2.0;
    _data[8] = F;
}

Ellipse2D::Ellipse2D(const vector<Point2D> &points)
{
    vector<Point2f> cvPoints;
    for (int i = 0; i < points.size(); i++)
    {
        cvPoints.emplace_back(Point2f((float)points[i].x(), (float)points[i].y()));
    }

    RotatedRect e = fitEllipseDirect(cvPoints);
    double xc = e.center.x;
    double yc = e.center.y;
    double a = e.size.width / 2.0; // getWidth >= height
    double b = e.size.height / 2.0;
    double theta = e.angle * CV_PI / 180.0;
    if (a < b)
    {
        swap(a, b);
        theta += CV_PI / 2.0;
    }

    double a2 = a * a;
    double b2 = b * b;
    double sint = sin(theta);
    double cost = cos(theta);
    double st2 = sint * sint;
    double ct2 = cost * cost;

    double A = a2 * st2 + b2 * ct2;
    double B = 2.0 * (b2 - a2) * sint * cost;
    double C = a2 * ct2 + b2 * st2;
    double D = -2.0 * A * xc - B * yc;
    double E = -B * xc - 2.0 * C * yc;
    double F = A * xc * xc + B * xc * yc + C * yc * yc - a2 * b2;

    _data[0] = A;
    _data[1] = B / 2.0;
    _data[2] = D / 2.0;
    _data[3] = B / 2.0;
    _data[4] = C;
    _data[5] = E / 2.0;
    _data[6] = D / 2.0;
    _data[7] = E / 2.0;
    _data[8] = F;
}

Ellipse2D::Ellipse2D(const Circle2D &circle2D) : Ellipse2D(circle2D.radius(), circle2D.radius(),
                                                           circle2D.center().x(), circle2D.center().y(), 0.0)
{
}

bool Ellipse2D::operator==(const Ellipse2D &other) const
{
    return canonicalForm() == other.canonicalForm();
}

double Ellipse2D::A() const
{
    return _data[0];
}

double Ellipse2D::B() const
{
    return _data[1] + _data[3];
}

double Ellipse2D::C() const
{
    return _data[4];
}

double Ellipse2D::D() const
{
    return _data[2] + _data[6];
}

double Ellipse2D::E() const
{
    return _data[5] + _data[7];
}

double Ellipse2D::F() const
{
    return _data[8];
}

double Ellipse2D::semiMajorAxis() const
{
    double A = this->A();
    double B = this->B();
    double C = this->C();
    double D = this->D();
    double E = this->E();
    double F = this->F();

    return -sqrt(2 * (A * E * E + C * D * D - B * D * E + (B * B - 4 * A * C) * F) * (A + C + sqrt((A - C) * (A - C) + B * B))) / (B * B - 4 * A * C);
}

double Ellipse2D::semiMinorAxis() const
{
    double A = this->A();
    double B = this->B();
    double C = this->C();
    double D = this->D();
    double E = this->E();
    double F = this->F();

    return -sqrt(2 * (A * E * E + C * D * D - B * D * E + (B * B - 4 * A * C) * F) * (A + C - sqrt((A - C) * (A - C) + B * B))) / (B * B - 4 * A * C);
}

double Ellipse2D::centerX() const
{
    double A = this->A();
    double B = this->B();
    double C = this->C();
    double D = this->D();
    double E = this->E();

    return (2 * C * D - B * E) / (B * B - 4 * A * C);
}

double Ellipse2D::centerY() const
{
    double A = this->A();
    double B = this->B();
    double C = this->C();
    double D = this->D();
    double E = this->E();

    return (2 * A * E - B * D) / (B * B - 4 * A * C);
}

Point2D Ellipse2D::center() const
{
    return Point2D(centerX(), centerY());
}

double Ellipse2D::angle() const
{
    double A = this->A();
    double B = this->B();
    double C = this->C();

    if (B == 0 && A <= C)
        return 0.0;
    else if (B == 0 && A > C)
        return CV_PI / 2;

    // B != 0
    return atan2(C - A - sqrt((A - C) * (A - C) + B * B), B);
}

vector<double> Ellipse2D::generalForm() const
{
    vector<double> general(6);
    general[0] = A();
    general[1] = B();
    general[2] = C();
    general[3] = D();
    general[4] = E();
    general[5] = F();

    return general;
}

vector<double> Ellipse2D::canonicalForm() const
{
    vector<double> canonical(5);
    canonical[0] = semiMajorAxis();
    canonical[1] = semiMinorAxis();
    canonical[2] = centerX();
    canonical[3] = centerY();
    canonical[4] = angle();

    return canonical;
}

Ellipse2D Ellipse2D::normalized() const
{
    Ellipse2D e(*this);

    double norm = e.norm();
    if (norm != 0.0)
    {
        double invNorm = 1.0 / norm;
        for (int i = 0; i < 9; i++)
            e[i] *= invNorm;
    }

    return e;
}

Ellipse2D Ellipse2D::projected(const Matrix3x3 &mat) const
{
    Matx33d Minv = Matx33d(mat.data()).inv();
    Matx33d ep = Minv.t() * Matx33d(_data) * Minv;

    return Ellipse2D(ep.val);
}

bool Ellipse2D::contains(Point2D point)
{
    vector<double> canonical = canonicalForm();
    double a = canonical[0];
    double b = canonical[1];
    double xc = canonical[2];
    double yc = canonical[3];
    double theta = canonical[4];

    double cost = cos(theta);
    double sint = sin(theta);
    double xp = point.x() - xc;
    double yp = point.y() - yc;
    double xpp = cost * xp + sint * yp;
    double ypp = sint * xp - cost * yp;
    double dist = (xpp * xpp) / (a * a) + (ypp * ypp) / (b * b);

    if (dist > 1.0 || std::isnan(dist))
        return false;

    return true;
}

Matrix3x3 Ellipse2D::affineTransformationTo(const Ellipse2D &other) const
{
    double a1 = semiMajorAxis();
    double b1 = semiMinorAxis();
    double xc1 = centerX();
    double yc1 = centerY();
    double theta1 = angle();
    Matx33d A1(cos(theta1) * a1, -sin(theta1) * b1, xc1,
               sin(theta1) * a1, cos(theta1) * b1, yc1,
               0.0, 0.0, 1.0);

    double a2 = other.semiMajorAxis();
    double b2 = other.semiMinorAxis();
    double xc2 = other.centerX();
    double yc2 = other.centerY();
    double theta2 = other.angle();
    Matx33d A2(cos(theta2) * a2, -sin(theta2) * b2, xc2,
               sin(theta2) * a2, cos(theta2) * b2, yc2,
               0.0, 0.0, 1.0);

    Matx33d A12 = A2 * A1.inv();

    return Matrix3x3(A12.val);
}

bool Ellipse2D::atInfinity() const
{
    return _data[2] == 0.0 && _data[5] == 0.0 && _data[6] == 0.0 &&
           _data[7] == 0.0 && _data[8] == 0.0;
}

CircleArc2D::CircleArc2D(const Circle2D &c, double s, double e) : _circle(c), _startAngle(s), _endAngle(e)
{
}

CircleArc2D::CircleArc2D(const Point2D &c, double r, double s, double e) : _circle(c, r), _startAngle(s), _endAngle(e)
{
}

Circle2D CircleArc2D::circle() const
{
    return _circle;
}

double CircleArc2D::startAngle() const
{
    return _startAngle;
}

double CircleArc2D::endAngle() const
{
    return _endAngle;
}

Circle2D::Circle2D(const Point2D &c, double r) : _center(c), _radius(r)
{
}

Point2D Circle2D::center() const
{
    return _center;
}

double Circle2D::radius() const
{
    return _radius;
}

Point2D Circle2D::pointAt(double angle) const
{
    return Point2D(_center.x() + _radius * cos(angle), _center.y() + _radius * sin(angle));
}

Ellipse2D Circle2D::projected(const Matrix3x3 &homography) const
{
    return Ellipse2D(*this).projected(homography);
}

Plane3D::Plane3D() : Vector4x1()
{
}

Plane3D::Plane3D(const double *data) : Vector4x1(data)
{
}

Plane3D::Plane3D(const Plane3D &other) : Vector4x1(other)
{
}

Plane3D::Plane3D(const Vector4x1 &vec)
{
    _data[0] = vec[0];
    _data[1] = vec[1];
    _data[2] = vec[2];
    _data[3] = vec[3];
}

Plane3D::Plane3D(double a, double b, double c, double d)
{
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
    _data[3] = d;
}

Plane3D::Plane3D(const Point3D &point, const Vector3x1 &normal)
{
    if (point.atInfinity() || normal.norm() == 0.0)
    {
        _data[0] = 0.0;
        _data[1] = 0.0;
        _data[2] = 0.0;
        _data[3] = 1.0;
    }
    else
    {
        Vector3x1 v(point.x(), point.y(), point.z());
        _data[0] = normal[0];
        _data[1] = normal[1];
        _data[2] = normal[2];
        _data[3] = -v.dotProduct(normal);
    }
}

Plane3D::Plane3D(const Point3D &p1, const Point3D &p2, const Point3D &p3)
{
    if (p1.atInfinity() || p2.atInfinity() || p3.atInfinity())
    {
        _data[0] = 0.0;
        _data[1] = 0.0;
        _data[2] = 0.0;
        _data[3] = 1.0;
    }
    else
    {
        Vector3x1 v1(p1.x(), p1.y(), p1.z());
        Vector3x1 v2(p2.x(), p2.y(), p2.z());
        Vector3x1 v3(p3.x(), p3.y(), p3.z());
        Vector3x1 n = (v2 - v1).crossProduct(v3 - v1);
        _data[0] = n[0];
        _data[1] = n[1];
        _data[2] = n[2];
        _data[3] = -v1.dotProduct(n);
    }
}

Plane3D::Plane3D(const std::vector<Point3D> &points)
{
    Mat_<double> A(points.size(), 4);

    size_t row = 0;
    for (const auto &p : points)
    {
        A(row, 0) = p.x();
        A(row, 1) = p.y();
        A(row, 2) = p.z();
        A(row, 3) = 1.0;
        row++;
    }

    Mat_<double> U;
    Mat_<double> D;
    Mat_<double> Vt;

    SVD::compute(A, D, U, Vt, SVD::MODIFY_A | SVD::FULL_UV);

    // D is the matrix of singular values
    // Find non zero singular values.
    // Use a small threshold to account for numeric errors
    Mat1b nonZeroSingularValues = D > 0.0001;

    // Count the number of non zero
    int rank = countNonZero(nonZeroSingularValues);
    if (rank < 3)
    {
        _data[0] = 0.0;
        _data[1] = 0.0;
        _data[2] = 0.0;
        _data[3] = 1.0;
    }
    else
    {
        int solutionIndex = Vt.rows - 1;
        _data[0] = Vt(solutionIndex, 0);
        _data[1] = Vt(solutionIndex, 1);
        _data[2] = Vt(solutionIndex, 2);
        _data[3] = Vt(solutionIndex, 3);
    }
}

double Plane3D::a() const
{
    return _data[0];
}

double Plane3D::b() const
{
    return _data[1];
}

double Plane3D::c() const
{
    return _data[2];
}

double Plane3D::d() const
{
    return _data[3];
}

void Plane3D::set(double a, double b, double c, double d)
{
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
    _data[3] = d;
}

bool Plane3D::operator==(const Plane3D &other) const
{
    bool thisAtInfinity = this->atInfinity();
    bool otherAtInfinity = other.atInfinity();
    if ((thisAtInfinity && !otherAtInfinity) || (!thisAtInfinity && otherAtInfinity))
    {
        return false;
    }

    if (thisAtInfinity && otherAtInfinity)
    {
        return true;
    }

    Plane3D p1 = this->normalized();
    Plane3D p2 = other.normalized();

    return sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2) + pow(p2[2] - p1[2], 2) + pow(p2[3] - p1[3], 2)) < numeric_limits<double>::epsilon();
}

double Plane3D::signedDistance(const Point3D &point) const
{
    return point.signedDistance(*this);
}

double Plane3D::distance(const Point3D &point) const
{
    return point.distance(*this);
}

Plane3D Plane3D::normalized() const
{
    Plane3D p(*this);

    for (int i = 0; i < 4; i++)
        if (fabs(p[i]) < numeric_limits<double>::epsilon())
            p[i] = 0.0;

    if (p.atInfinity())
    {
        p[3] = 1.0;
        return p;
    }

    double n = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (p[0] < 0.0)
        n = -n;

    p[0] /= n;
    p[1] /= n;
    p[2] /= n;
    p[3] /= n;

    return p;
}

template <typename T>
vector<size_t> sort_indices_ascending_order(const vector<T> &v)
{

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2)
         { return v[i1] < v[i2]; });

    return idx;
}

Point3D Plane3D::intersection(const Line3D &line) const
{
    Point3D intersection;

    bool planeAtInfinity = this->atInfinity();
    bool lineAtInfinity = line.atInfinity();

    if (planeAtInfinity && lineAtInfinity)
    {
        intersection = line.parametricEquationPoint();
    }
    else if (planeAtInfinity && !lineAtInfinity)
    {
        Vector3x1 v = line.parametricEquationSlope();
        intersection.set(v[0], v[1], v[2], 0.0);
    }
    else if (!planeAtInfinity && lineAtInfinity)
    {
        vector<double> n = {_data[0], _data[1], _data[2]};
        auto idx = sort_indices_ascending_order(n);

        vector<double> v1(3);
        v1[idx[0]] = 1.0;
        v1[idx[1]] = 1.0;
        v1[idx[2]] = -(_data[idx[0]] + _data[idx[1]]) / _data[idx[2]];

        vector<double> v2(3);
        if (fabs(_data[idx[1]]) < numeric_limits<double>::epsilon())
        {
            v2[idx[0]] = 2.0;
            v2[idx[1]] = 1.0;
            v2[idx[2]] = -(_data[idx[0]] + _data[idx[1]]) / _data[idx[2]];
        }
        else
        {
            v2[idx[0]] = 1.0;
            v2[idx[1]] = -(_data[idx[0]] + _data[idx[2]]) / _data[idx[1]];
            v2[idx[2]] = 1.0;
        }

        Point3D pi1(v1[0], v1[1], v1[2], 0.0);
        Point3D pi2(v2[0], v2[1], v2[2], 0.0);
        Line3D li(pi1, pi2);
    }
    else if (!planeAtInfinity && !lineAtInfinity)
    {
        Point3D p = line.parametricEquationPoint();
        Vector3x1 v = line.parametricEquationSlope();
        double x1 = p.x();
        double y1 = p.y();
        double z1 = p.z();
        double a = v[0];
        double b = v[1];
        double c = v[2];
        double A = this->a();
        double B = this->b();
        double C = this->c();
        double D = this->d();

        double s = a * A + b * B + c * C;

        if (fabs(s) < numeric_limits<double>::epsilon())
        {
            Vector3x1 v = line.parametricEquationSlope();
            intersection.set(v[0], v[1], v[2], 0.0);
        }
        else
        {
            double t = -(x1 * A + y1 * B + z1 * C + D) / s;
            intersection.set(x1 + t * a, y1 + t * b, z1 + t * c);
        }
    }

    return intersection;
}

Vector3x1 Plane3D::normal() const
{
    Plane3D np = this->normalized();

    return Vector3x1(np.a(), np.b(), np.c());
}

bool Plane3D::atInfinity() const
{
    return _data[0] == 0.0 && _data[1] == 0.0 && _data[2] == 0.0;
}
