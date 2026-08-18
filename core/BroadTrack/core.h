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

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

class Point2D;
class Point3D;
class Line2D;
class LineSegment2D;
class Line3D;
class Vector3x1;
class Vector4x1;
class Matrix3x3;
class CircleArc2D;
class Circle2D;
class Ellipse2D;
class Plane3D;

typedef std::vector<Point2D> Polyline2D;
typedef std::vector<Point3D> Polyline3D;

class Vector4x1
{
public:
    Vector4x1();

    Vector4x1(const double data[4]);

    Vector4x1(const Vector4x1 &other);

    Vector4x1(double v1, double v2, double v3, double v4);

    virtual ~Vector4x1() = default;

    double *data();

    const double *data() const;

    int nbElems() const;

    double &operator[](int index);

    const double &operator[](int index) const;

    Vector4x1 operator+(const Vector4x1 &other) const;

    Vector4x1 operator-(const Vector4x1 &other) const;

    Vector4x1 operator-() const;

    double dotProduct(const Vector4x1 &vec) const;

    virtual double norm() const;

    virtual void scale(double s);

    friend std::ostream &operator<<(std::ostream &out, const Vector4x1 &vec);

protected:
    double _data[4];
};

class Vector3x1
{
public:
    static Vector3x1 zero();

    Vector3x1();

    Vector3x1(const double data[3]);

    Vector3x1(const Vector3x1 &other);

    Vector3x1(double v1, double v2, double v3);

    virtual ~Vector3x1() = default;

    double *data();

    const double *data() const;

    int nbElems() const;

    double &operator[](int index);

    const double &operator[](int index) const;

    Vector3x1 operator+(const Vector3x1 &other) const;

    Vector3x1 operator-(const Vector3x1 &other) const;

    Vector3x1 operator-() const;

    Vector3x1 crossProduct(const Vector3x1 &vec) const;

    double dotProduct(const Vector3x1 &vec) const;

    virtual double norm() const;

    virtual void scale(double s);

    Vector3x1 operator*(double factor) const;

    Vector3x1 operator/(double factor) const;

    Vector3x1 unitLength() const;

    friend std::ostream &operator<<(std::ostream &out, const Vector3x1 &vec);

protected:
    double _data[3];
};

class Matrix3x3
{
public:
    static Matrix3x3 identity();

    Matrix3x3();

    Matrix3x3(const double data[9]);

    Matrix3x3(const Matrix3x3 &other);

    Matrix3x3(double m11, double m12, double m13,
              double m21, double m22, double m23,
              double m31, double m32, double m33);

    Matrix3x3(const Vector3x1 &c1, const Vector3x1 &c2, const Vector3x1 &c3);

    virtual ~Matrix3x3() = default;

    double *data();

    const double *data() const;

    int nbElems() const;

    double &operator[](int index);

    const double &operator[](int index) const;

    Matrix3x3 operator*(const Matrix3x3 &mat) const;

    Vector3x1 operator*(const Vector3x1 &vec) const;

    Point3D operator*(const Point3D &point3D) const;

    Line3D operator*(const Line3D &line3D) const;

    Matrix3x3 operator*(double scalar) const;

    Matrix3x3 operator-() const;

    double &operator()(int row, int col);

    void set(const double data[9]);

    void set(double m11, double m12, double m13,
             double m21, double m22, double m23,
             double m31, double m32, double m33);

    double norm() const;

    Matrix3x3 inverse() const;

    Matrix3x3 transpose() const;

    Matrix3x3 multiply(const Matrix3x3 &mat) const;

    Vector3x1 multiply(const Vector3x1 &vec) const;

    Point3D multiply(const Point3D &point3D) const;

    Line3D multiply(const Line3D &line3D) const;

    Matrix3x3 multiply(double scalar) const;

    Vector3x1 column(int index) const;

    friend std::ostream &operator<<(std::ostream &out, const Matrix3x3 &mat);

protected:
    double _data[9];
};

class Point3D : public Vector4x1
{
public:
    Point3D();

    Point3D(const double data[4]);

    Point3D(const Point3D &other);

    // conversion from Vector4x1 (constructor):
    explicit Point3D(const Vector4x1 &vec);
    //  // conversion from A (assignment):
    //  B& operator= (const A& x) {return *this;}
    //  // conversion to A (type-cast operator)
    //  operator A() {return A();}

    explicit Point3D(const Vector3x1 &vec);

    virtual ~Point3D() = default;

    Point3D(double x, double y, double z);

    Point3D(double hx, double hy, double hz, double w);

    Point3D(const Point2D &xy, double z);

    Point3D(const std::vector<Line3D> &lines);

    Point3D operator+(const Point3D &other) const;

    Point3D operator-(const Point3D &other) const;

    Point3D operator-() const;

    Point3D operator*(double scale) const;

    Point3D operator/(double scale) const;

    double x() const;

    double y() const;

    double z() const;

    double hx() const;

    double hy() const;

    double hz() const;

    double w() const;

    void setX(double value);

    void setY(double value);

    void setZ(double value);

    void set(double hx, double hy, double hz, double w = 1.0);

    bool operator==(const Point3D &other) const;

    double squaredNorm() const;

    double norm() const override;

    void scale(double s) override;

    double squaredDistance(const Point3D &point) const;

    double distance(const Point3D &point) const;

    double signedDistance(const Plane3D &plane) const;

    double distance(const Plane3D &plane) const;

    double distance(const Polyline3D &polyline3D) const;

    Point3D normalized() const;

    Point2D normalizedCentralProjection() const;

    Point2D xy() const;

    bool atInfinity() const;
};

class Point2D : public Vector3x1
{
public:
    Point2D();

    Point2D(const double data[3]);

    Point2D(const Point2D &other);

    explicit Point2D(const Vector3x1 &vec);

    virtual ~Point2D() = default;

    Point2D(double x, double y);

    Point2D(double hx, double hy, double w);

    Point2D(const Line2D &l1, const Line2D &l2);

    double x() const;

    double y() const;

    double hx() const;

    double hy() const;

    double w() const;

    void setX(double value);

    void setY(double value);

    void set(double hx, double hy, double w = 1.0);

    bool operator==(const Point2D &other) const;

    bool operator!=(const Point2D &other) const;

    Point2D operator+(const Point2D &other) const;

    Point2D operator-(const Point2D &other) const;

    Point2D operator-() const;

    Point2D operator*(double scale) const;

    Point2D operator/(double scale) const;

    double squaredNorm() const;

    double norm() const override;

    void scale(double s) override;

    double squaredDistance(const Point2D &point) const;

    double distance(const Point2D &point) const;

    double signedDistance(const Line2D &line) const;

    double distance(const Line2D &line) const;

    double distance(const LineSegment2D &lineSegment);

    double distance(const std::vector<Point2D> pointList);

    double distance(const Polyline2D &polyline2D) const;

    Point2D normalized() const;

    Point2D projected(const Line2D &line) const;

    Point2D projected(const Matrix3x3 &mat) const;

    bool isWithin(const cv::Size &size) const;

    bool atInfinity() const;
};

class Plane3D : public Vector4x1
{
public:
    Plane3D();

    Plane3D(const double data[4]);

    Plane3D(const Plane3D &other);

    explicit Plane3D(const Vector4x1 &vec);

    virtual ~Plane3D() = default;

    Plane3D(double a, double b, double c, double d);

    Plane3D(const Point3D &point, const Vector3x1 &normal);

    Plane3D(const Point3D &p1, const Point3D &p2, const Point3D &p3);

    Plane3D(const std::vector<Point3D> &points);

    double a() const;

    double b() const;

    double c() const;

    double d() const;

    void set(double a, double b, double c, double d);

    bool operator==(const Plane3D &other) const;

    double signedDistance(const Point3D &point) const;

    double distance(const Point3D &point) const;

    Plane3D normalized() const;

    Point3D intersection(const Line3D &line) const;

    Vector3x1 normal() const;

    bool atInfinity() const;
};

class Circle2D
{
public:
    Circle2D() = default;

    Circle2D(const Point2D &c, double r);

    Point2D center() const;

    double radius() const;

    Point2D pointAt(double angle) const;

    Ellipse2D projected(const Matrix3x3 &homography) const;

private:
    Point2D _center;
    double _radius;
};

class CircleArc2D
{
public:
    CircleArc2D() = default;

    CircleArc2D(const Circle2D &c, double s, double e);

    CircleArc2D(const Point2D &c, double r, double s, double e);

    Circle2D circle() const;

    double startAngle() const;

    double endAngle() const;

private:
    Circle2D _circle;
    double _startAngle;
    double _endAngle;
};

class Ellipse2D : public Matrix3x3
{
public:
    Ellipse2D();

    Ellipse2D(const double data[9]);

    Ellipse2D(const Ellipse2D &other);

    Ellipse2D(double A, double B, double C, double D, double E, double F);

    Ellipse2D(double a, double b, double xc, double yc, double theta);

    Ellipse2D(const std::vector<Point2D> &points);

    Ellipse2D(const Circle2D &circle2D);

    bool operator==(const Ellipse2D &other) const;

    double A() const;

    double B() const;

    double C() const;

    double D() const;

    double E() const;

    double F() const;

    double semiMajorAxis() const;

    double semiMinorAxis() const;

    double centerX() const;

    double centerY() const;

    Point2D center() const;

    double angle() const;

    std::vector<double> generalForm() const;

    std::vector<double> canonicalForm() const;

    Ellipse2D normalized() const;

    Ellipse2D projected(const Matrix3x3 &mat) const;

    bool contains(Point2D point);

    Matrix3x3 affineTransformationTo(const Ellipse2D &other) const;

    bool atInfinity() const;
};

class Line2D : public Vector3x1
{
public:
    static Line2D xAxisLine();

    static Line2D yAxisLine();

    static Line2D infinityLine();

    Line2D();

    Line2D(const double data[3]);

    Line2D(const Line2D &other);

    // conversion from Vector3x1 (constructor):
    explicit Line2D(const Vector3x1 &vec);
    //  // conversion from A (assignment):
    //  B& operator= (const A& x) {return *this;}
    //  // conversion to A (type-cast operator)
    //  operator A() {return A();}

    Line2D(double a, double b, double c);

    Line2D(double a, double b, const Point2D &p);

    Line2D(const Point2D &p1, const Point2D &p2);

    Line2D(const std::vector<Point2D> &points);

    explicit Line2D(const LineSegment2D &segment);

    virtual ~Line2D() = default;

    double a() const;

    double b() const;

    double c() const;

    void set(double a, double b, double c);

    bool operator==(const Line2D &other) const;

    double signedDistance(const Point2D &point) const;

    double distance(const Point2D &point) const;

    Line2D normalized() const;

    Line2D projected(const Matrix3x3 &mat) const;

    Point2D pole(const Ellipse2D &ellipse) const;

    Point2D intersection(const Line2D &other) const;

    std::vector<Point2D> intersection(const Ellipse2D &ellipse) const;

    double angle(const Line2D &other) const;

    Line2D perpendicular(const Point2D &passingThroughPoint) const;

    bool atInfinity() const;
};

class Line3D : private std::pair<Point3D, Vector3x1>
{
public:
    Line3D();

    Line3D(const Point3D &firstPoint, const Point3D &secondPoint);

    Line3D(const Point3D &parametricEquationPoint, const Vector3x1 &parametricEquationSlope);

    Line3D(const Line3D &other);

    virtual ~Line3D() = default;

    Point3D parametricEquationPoint(double t = 0.0) const;

    Vector3x1 parametricEquationSlope() const;

    Point3D intersection(const Plane3D &plane) const;

    bool atInfinity() const;
};

class LineSegment2D
{
public:
    LineSegment2D() = default;

    LineSegment2D(const Point2D &s, const Point2D &e);

    Point2D start() const;

    Point2D end() const;

    double squaredLength() const;

    double length() const;

    LineSegment2D projected(const Matrix3x3 &homography) const;

    std::pair<Point2D, double> lineIntersection(const Line2D &line) const;

private:
    Point2D _start;
    Point2D _end;
};
