#ifndef PANGO_POINT_CLOUD_H_
#define PANGO_POINT_CLOUD_H_

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <Eigen/Core>
#include <vector>
#include <iostream>

class PangoPointCloud {
public:
    PangoPointCloud(std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3f>& colors)
    : numvertices(vertices.size()), vertices(std::move(vertices)), colors(std::move(colors)) {

    // Initialize VBOs if the data is non-empty
    if (!this->vertices.empty() && !this->colors.empty()) {
        initVBOs();
        
        // Clear the local CPU-side storage to free memory (optional, since vectors are moved)
        // this->vertices.clear();
        // this->vertices.shrink_to_fit();
        // this->colors.clear();
        // this->colors.shrink_to_fit();
    }
}

    ~PangoPointCloud() {
        if (vboVertices != 0) {
            glDeleteBuffers(1, &vboVertices);
            vboVertices = 0; // Avoid double deletion
        }
        if (vboColors != 0) {
            glDeleteBuffers(1, &vboColors);
            vboColors = 0; // Avoid double deletion
        }

    }

    // Initialize VBOs for vertices and colors
    void initVBOs() {
    // Generate buffers
    glGenBuffers(1, &vboVertices);
    glGenBuffers(1, &vboColors);

    // Bind and upload vertices data to VBO
    glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Eigen::Vector3f), vertices.data(), GL_STATIC_DRAW);

    // Bind and upload colors data to VBO
    glBindBuffer(GL_ARRAY_BUFFER, vboColors);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), colors.data(), GL_STATIC_DRAW);

    // Unbind the VBOs
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Clear CPU-side buffers to save memory
    // vertices.clear();
    // vertices.shrink_to_fit();
    // colors.clear();
    // colors.shrink_to_fit();
}


    void draw() const {
        if (numvertices == 0) {
            return; // Avoid drawing if no vertices exist
        }
        // Enable and bind the vertex VBO
        glEnableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
        glVertexPointer(3, GL_FLOAT, 0, nullptr); // No need to pass vertices.data() as data is already in GPU

        // Enable and bind the color VBO
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, vboColors);
        glColorPointer(3, GL_FLOAT, 0, nullptr); // No need to pass colors.data() as data is already in GPU

        // Draw the point cloud
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numvertices));

        // Unbind the VBOs and disable client states
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    // Append new vertices and colors
    void append(std::vector<Eigen::Vector3f>& newVertices, std::vector<Eigen::Vector3f>& newColors) {
        if (newVertices.size() != newColors.size()) {
            std::cerr << "Error: new vertices and colors size mismatch!" << std::endl;
            return;
        }

        // If VBOs haven't been initialized yet, do so now
        if (vboVertices == 0 || vboColors == 0) {
            initVBOs(); // Initialize VBOs if not done already
        }

        // Append new data to the existing vectors
        vertices.insert(vertices.end(), newVertices.begin(), newVertices.end());
        colors.insert(colors.end(), newColors.begin(), newColors.end());

        // Update the number of vertices
        numvertices = vertices.size();

        // Update the VBOs with the new data
        updateVBOs();
    }

    // Getter for vertices
    std::vector<Eigen::Vector3f>& getVertices() {
        return vertices;
    }

    // Getter for colors
    std::vector<Eigen::Vector3f>& getColors() {
        return colors;
    }

    void clear() {
        // Clear the CPU-side vertex and color buffers
        vertices.clear();
        colors.clear();
        numvertices = 0;

        // Reset the VBOs
        glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, vboColors);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

        // Unbind the VBOs
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void setVertices(const std::vector<Eigen::Vector3f>& newVertices) {
        vertices = newVertices;
        updateVBOs(); // Ensure VBOs are updated
    }

    // Update VBOs with the appended data
    void updateVBOs() {
        // Bind and upload updated vertices data to VBO
        glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Eigen::Vector3f), vertices.data(), GL_STATIC_DRAW);

        // Bind and upload updated colors data to VBO
        glBindBuffer(GL_ARRAY_BUFFER, vboColors);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), colors.data(), GL_STATIC_DRAW);

        // Unbind the VBOs
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

private:
    size_t numvertices; // Number of vertices in the cloud
    std::vector<Eigen::Vector3f> vertices; // Vector of vertices (CPU side, for future updates)
    std::vector<Eigen::Vector3f> colors; // Vector of colors (CPU side, for future updates)

    // VBO handles
    GLuint vboVertices = 0;
    GLuint vboColors = 0;

    
};

#endif // PANGO_POINT_CLOUD_H_
