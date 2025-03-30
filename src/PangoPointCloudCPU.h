#ifndef PANGO_POINT_CLOUD_H_
#define PANGO_POINT_CLOUD_H_

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>
#include <iostream>

class PangoPointCloud {
public:
    PangoPointCloud(std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3f>& colors)
    : numvertices(vertices.size()), vertices(std::move(vertices)), colors(std::move(colors)) {
        if (this->vertices.size() != this->colors.size()) {
            throw std::runtime_error("Vertices and colors size mismatch!");
        }
    }

    ~PangoPointCloud() = default;

    void draw() const {
        if (numvertices == 0) {
            return; // Avoid drawing if no vertices exist
        }

        // Enable client states
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        // Pass vertices and colors directly
        glVertexPointer(3, GL_FLOAT, 0, vertices.data());
        glColorPointer(3, GL_FLOAT, 0, colors.data());

        // Draw the point cloud
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numvertices));

        // Disable client states
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    void append(std::vector<Eigen::Vector3f>& newVertices, std::vector<Eigen::Vector3f>& newColors) {
        if (newVertices.size() != newColors.size()) {
            std::cerr << "Error: new vertices and colors size mismatch!" << std::endl;
            return;
        }

        // Append new data to the existing vectors
        vertices.insert(vertices.end(), newVertices.begin(), newVertices.end());
        colors.insert(colors.end(), newColors.begin(), newColors.end());

        // Update the number of vertices
        numvertices = vertices.size();
    }

    void clear() {
        vertices.clear();
        colors.clear();
        numvertices = 0;
    }

    void setVertices(const std::vector<Eigen::Vector3f>& newVertices) {
        vertices = newVertices;
        numvertices = vertices.size();
    }

    void setColors(const std::vector<Eigen::Vector3f>& newColors) {
        if (newColors.size() != numvertices) {
            std::cerr << "Error: New colors size must match the number of vertices!" << std::endl;
            return;
        }
        colors = newColors;
    }

    std::vector<Eigen::Vector3f>& getVertices() {
        return vertices;
    }

    std::vector<Eigen::Vector3f>& getColors() {
        return colors;
    }

private:
    size_t numvertices; // Number of vertices in the cloud
    std::vector<Eigen::Vector3f> vertices; // Vector of vertices (CPU side)
    std::vector<Eigen::Vector3f> colors;   // Vector of colors (CPU side)
};

#endif // PANGO_POINT_CLOUD_H_
