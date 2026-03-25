#include "mesh.h"
#include <map>
#include <utility>

Mesh icosphere(double radius, int refinements) {
    double phi = (1.0 + sqrt(5.0)) / 2.0;

    // 12 initial vertices of icosahedron (on unit sphere)
    std::vector<Vec3> verts = {
        {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
        {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
        {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1},
    };
    // Normalize to unit sphere
    double norm0 = verts[0].norm();
    for (auto& v : verts) { v = v * (1.0 / norm0); }

    // 20 initial triangles
    std::vector<int> tris = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };

    for (int ref = 0; ref < refinements; ref++) {
        std::map<std::pair<int,int>, int> edge_mid;
        std::vector<int> new_tris;

        auto get_mid = [&](int a, int b) -> int {
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            auto it = edge_mid.find(key);
            if (it != edge_mid.end()) return it->second;
            Vec3 mid = (verts[a] + verts[b]) * 0.5;
            mid = mid.normalized();
            int idx = (int)verts.size();
            verts.push_back(mid);
            edge_mid[key] = idx;
            return idx;
        };

        int ntri = (int)tris.size() / 3;
        for (int i = 0; i < ntri; i++) {
            int a = tris[3*i], b = tris[3*i+1], c = tris[3*i+2];
            int ab = get_mid(a, b);
            int bc = get_mid(b, c);
            int ca = get_mid(c, a);
            // 4 new triangles
            int t[] = {a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca};
            new_tris.insert(new_tris.end(), t, t+12);
        }
        tris = new_tris;
    }

    // Scale to desired radius
    for (auto& v : verts) { v = v * radius; }

    Mesh m;
    m.verts = verts;
    m.tris = tris;
    return m;
}
