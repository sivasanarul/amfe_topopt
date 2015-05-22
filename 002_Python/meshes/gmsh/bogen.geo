// Gmsh project created on Thu May 21 19:04:40 2015
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 0.5, 0, 1.0};
Point(3) = {0, 3, 0, 1.0};
Point(4) = {0, 6, 0, 1.0};
Point(5) = {0, 5.5, 0, 1.0};
Point(6) = {3, 5.5, 0, 0.5};
Point(7) = {3, 6, 0, 0.5};
Point(8) = {3, 0, 0, 0.5};
Point(9) = {3, 0.5, 0, 0.5};
Point(10) = {3, 3, 0, 1.0};
Point(11) = {6, 3, 0, 0.5};
Point(12) = {5.5, 3, 0, 0.5};
Line(1) = {4, 7};
Line(2) = {4, 5};
Line(3) = {6, 5};
Line(4) = {9, 2};
Line(5) = {2, 1};
Line(6) = {1, 8};
Circle(7) = {6, 10, 12};
Circle(8) = {12, 10, 9};
Circle(9) = {7, 10, 11};
Circle(10) = {11, 10, 8};
Line Loop(11) = { 2,  -3,   7,   8,   4,   5,   6, -10,  -9,  -1};
Plane Surface(12) = {11};
