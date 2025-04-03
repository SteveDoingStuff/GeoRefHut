// #include <PangoPointCloud.h>
#include <PangoPointCloudCPU.h>


#include <pangolin/display/default_font.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <E57Format.h>
#include <E57SimpleReader.h>
#include <E57SimpleWriter.h>

#include <memory>
#include <vector>
#include <map>
#include <filesystem>
#include <tinyfiledialogs.h>
#include <GL/glu.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <pangolin/gl/gldraw.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/registration/registration_result.hpp>

#include <nanoflann.hpp>

#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"

#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"

#include <nlohmann/json.hpp>



struct ScanData {
    std::shared_ptr<PangoPointCloud> fullCloud = nullptr;   // Full point cloud stored in VBOs
    std::shared_ptr<PangoPointCloud> subsampledCloud = nullptr; // Precomputed subsampled cloud
    bool isRendered = false;                                // Rendered state
    bool useSubsampled = true;                              // Flag to toggle between full and subsampled
};



// Initiate a global shift
Eigen::Matrix4d globalShift = Eigen::Matrix4d::Identity();

std::filesystem::path curr_path = std::filesystem::current_path().parent_path();

std::unordered_map<std::string, ScanData> loadedScans;
std::unordered_map<std::string, Eigen::Matrix4d > ICPTrafo;

bool draw_labels = false;
bool draw_links = false;
bool custom_stoch = false;
bool use_hessian =false;
bool use_hessian_diag =false;
bool com_rel_cov = false;
std::map<std::string,g2o::VertexSE3*> rel_cov_poses;

bool posesOptimized = false;

// Declare globally accessible shared pointer
std::vector<Eigen::Vector3f> emptyVertices;
std::vector<Eigen::Vector3f> emptyColors;

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> lines;


struct ScanPosition {
    Eigen::Vector3f positionLocal;
    std::string name;
    std::string key;
    Eigen::Matrix4d LocalPose;
    Eigen::Matrix4d GlobalPose;
};

std::vector<ScanPosition> scanPositions;


// Initialize the shared pointer with empty vectors
std::shared_ptr<PangoPointCloud> pangoCloudCube = std::make_shared<PangoPointCloud>(emptyVertices, emptyColors);

// Custom handler to handle mouse clicks for point picking
class CustomHandler3D : public pangolin::Handler3D {
public:
    CustomHandler3D(pangolin::OpenGlRenderState& cam_state, pangolin::View& view, int panelWidth)
        : pangolin::Handler3D(cam_state), view_(view), panelWidth_(panelWidth), picked_point_(nullptr) {}

    ~CustomHandler3D() override = default;

    void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) override {
        pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);

        if (pressed && (button == pangolin::MouseButtonLeft) && (button_state & pangolin::KeyModifierShift)) {
            
            // Apply a correction if necessary
            int corrected_x = x; // Correction by panel width

            // std::cout << "Click position: (" << corrected_x << ", " << y << ")\n";
            PickPoint3D(corrected_x, y); // Get the 3D point at the corrected position
        }
    }

    void PickPoint3D(int x, int y) {
        // Get the depth at the clicked position
        GLfloat depth = view_.GetClosestDepth(x, y, 1);

        if (depth == 1.0f) {
            std::cout << "No point detected at the clicked position (depth=1.0)\n";
            return;
        }

        // Now get the world (object) coordinates based on the depth
        GLdouble world_x, world_y, world_z;
        view_.GetObjectCoordinates(*cam_state, x, y, depth, world_x, world_y, world_z);
        Eigen::Vector4d point_homogeneous(world_x, world_y, world_z, 1.0);
        Eigen::Vector4d corrected_point = globalShift * point_homogeneous;
        double corrected_x = corrected_point[0];
        double corrected_y = corrected_point[1];
        double corrected_z = corrected_point[2];    
        // Print the 3D coordinates
        std::cout << "3D coordinates: (" << corrected_x << ", " << corrected_y << ", " << corrected_z << ")\n";

        // Store the picked point
        picked_point_ = std::make_unique<Eigen::Vector3d>(world_x, world_y, world_z);
        picked_point_global = std::make_unique<Eigen::Vector3d>(corrected_x, corrected_y, corrected_z);
    }


    void DrawPickedPoint() const {
        if (picked_point_) {
            glColor3f(1.0, 0.0, 0.0); // Red color

            float length = 0.1f;

            pangolin::glDrawCross(picked_point_->x(), picked_point_->y(), picked_point_->z(), length);
        }
    }

    // Getter method for accessing the picked point
    Eigen::Vector3d* GetPickedPoint() {
        return picked_point_ ? picked_point_.get() : nullptr;
    }

    const Eigen::Vector3d* GetPickedPointGlobal() const {
        return picked_point_global ? picked_point_global.get() : nullptr;
    }

    private:
        pangolin::View& view_;
        std::unique_ptr<Eigen::Vector3d> picked_point_;
        std::unique_ptr<Eigen::Vector3d> picked_point_global;
        int panelWidth_;
};

// Function to find points in the 1m³ cube around a picked point
void GetPointsInSphere(const Eigen::Vector3f& pickedPoint) {
    const float sphereRadius = 1.0f; 
    float sphereRadiusSquared = sphereRadius * sphereRadius; // Squared radius for distance comparison

    std::vector<Eigen::Vector3f> accumulatedVertices; // Store points that will be appended at the end
    std::vector<Eigen::Vector3f> accumulatedColors;   // Store corresponding colors

    // Iterate through all loaded scans
    for (const auto& scanEntry : loadedScans) {
        const auto& scanData = scanEntry.second;
        if (scanData.isRendered) {
            if (!scanData.fullCloud) {
                std::cerr << "Full cloud not available for scan: " << scanEntry.first << std::endl;
                continue;
            }

            const std::vector<Eigen::Vector3f>& points = scanData.fullCloud->getVertices();
            const std::vector<Eigen::Vector3f>& colors = scanData.fullCloud->getColors();

            // Use thread-local storage to avoid data races
            std::vector<Eigen::Vector3f> localVertices;
            std::vector<Eigen::Vector3f> localColors;

            // Parallel for loop to check points
            #pragma omp parallel for
            for (size_t i = 0; i < points.size(); ++i) {
                float distanceSquared = (points[i] - pickedPoint).squaredNorm(); // Squared distance for efficiency

                // Check if the point is inside the 1-meter radius sphere
                if (distanceSquared <= sphereRadiusSquared) {
                    // Append the points and colors to the local (thread-specific) vectors
                    #pragma omp critical
                    {
                        localVertices.push_back(points[i]);
                        localColors.push_back(colors[i]);
                    }
                }
            }

            // Append local results after the parallel loop is done
            #pragma omp critical
            {
                accumulatedVertices.insert(accumulatedVertices.end(), localVertices.begin(), localVertices.end());
                accumulatedColors.insert(accumulatedColors.end(), localColors.begin(), localColors.end());
            }
        }
    }

    // Now append the accumulated points and colors to pangoCloudCube in one go
    if (!accumulatedVertices.empty() && !accumulatedColors.empty()) {
        pangoCloudCube->append(accumulatedVertices, accumulatedColors);
    }
}





void UpdateWindowSize(int& width, int& height, float& aspect) {
    // Get the current window dimensions
    width = pangolin::DisplayBase().v.w;
    height = pangolin::DisplayBase().v.h;
    aspect = static_cast<float>(width) / height;
}


void SubsamplePointCloud(const std::vector<Eigen::Vector3f>& points, 
                         const std::vector<Eigen::Vector3f>& colors, 
                         std::vector<Eigen::Vector3f>& subsampledPoints, 
                         std::vector<Eigen::Vector3f>& subsampledColors,
                         size_t maxPoints) {
    size_t n = points.size();
    if (n <= maxPoints) {
        subsampledPoints = points;
        subsampledColors = colors;
        return;
    }

    subsampledPoints.resize(maxPoints);
    subsampledColors.resize(maxPoints);

    std::random_device rd;
    std::default_random_engine rng(rd());

    #pragma omp parallel for
    for (size_t i = 0; i < maxPoints; ++i) {
        size_t randomIndex = rng() % n; // Pick a random point
        subsampledPoints[i] = points[randomIndex];
        subsampledColors[i] = colors[randomIndex];
    }
}



std::vector<double> recentScanPose(3, 0.0);

// Function to extract pose matrix
Eigen::Matrix4d GetPoseMatrix(const e57::Data3D& scanHeader) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();

    // Quaternion to Rotation Matrix
    Eigen::Quaterniond quat(scanHeader.pose.rotation.w, scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z);
    mat.block<3,3>(0,0) = quat.toRotationMatrix();

    // Translation
    mat(0,3) = scanHeader.pose.translation.x;
    mat(1,3) = scanHeader.pose.translation.y;
    mat(2,3) = scanHeader.pose.translation.z;

    // Update global pose variable
    recentScanPose[0] = mat(0, 3); // X
    recentScanPose[1] = mat(1, 3); // Y
    recentScanPose[2] = mat(2, 3); // Z

    // Print just the translation part
    std::cout << std::fixed << std::setprecision(8)
              << "Coordinates of scan: [" 
              << scanHeader.pose.translation.x << ", "
              << scanHeader.pose.translation.y << ", "
              << scanHeader.pose.translation.z << "]" << std::endl;

    return mat;
}




void readE57(const std::string& filename, const std::string& scanName, size_t maxPoints = 500000) {
    std::string extension = filename.substr(filename.find_last_of(".") + 1);

    if (extension != "e57") {
        std::cout << "Unsupported file format: " << extension << std::endl;
        return;
    }
    

    e57::Reader e57Reader(filename, e57::ReaderOptions());
    std::cout << "Reading file: " << filename << std::endl;
    std::cout << "Scans in file:" << e57Reader.GetData3DCount() << std::endl;

    for (size_t scanIndex = 0; scanIndex < e57Reader.GetData3DCount(); ++scanIndex) {
        e57::Data3D scanHeader;
        e57Reader.ReadData3D(scanIndex, scanHeader);
        

        if (scanHeader.name != scanName) {
            if (!(e57Reader.GetData3DCount() == 1)) {
                continue;
            }
        }
        std::cout << "Reading Scan: " << scanHeader.name << std::endl;
        std::string scanKey = filename + "_" + scanHeader.name;

        size_t nPoints = scanHeader.pointCount;
        std::cout << "Number of points: " << nPoints << std::endl;
        std::vector<float> xData(nPoints), yData(nPoints), zData(nPoints);
        std::vector<uint16_t> rData(nPoints), gData(nPoints), bData(nPoints);
        std::vector<double> intensityData(nPoints);
        std::vector<Eigen::Vector3f> points, colors, subsampledPoints, subsampledColors;

        // Set up the buffer for reading the data
        e57::Data3DPointsFloat buffers;
        buffers.cartesianX = xData.data();
        buffers.cartesianY = yData.data();
        buffers.cartesianZ = zData.data();
        buffers.colorRed = rData.data();
        buffers.colorGreen = gData.data();
        buffers.colorBlue = bData.data();
        buffers.intensity = intensityData.data();


        // Read the data from the scan
        e57::CompressedVectorReader dataReader = e57Reader.SetUpData3DPointsData(scanIndex, nPoints, buffers);
        dataReader.read();
        dataReader.close();
        std::cout << "Reading finished " << std::endl;

        // Eigen::Matrix4d poseMatrix = Eigen::Matrix4d::Identity();
        // poseMatrix = GetPoseMatrix(scanHeader);

        // Eigen::Matrix4d localPoseMatrix = globalShift.inverse() * poseMatrix;

        // read poses from scanPositions:
        Eigen::Matrix4d localPoseMatrix;
        for (size_t i = 0; i < scanPositions.size(); i++) {
            if (scanPositions[i].key == scanKey) {
                localPoseMatrix = scanPositions[i].LocalPose;
            }
        }
        

        points.reserve(nPoints);  // Pre-allocate memory for points
        colors.reserve(nPoints);  // Pre-allocate memory for colors

        // Parallelized loop for transforming points
        #pragma omp parallel for
        for (size_t i = 0; i < nPoints; ++i) {
            // Apply the transformation to the point
            Eigen::Vector4f point(xData[i], yData[i], zData[i], 1.0f);
            Eigen::Vector4f transformedPoint = localPoseMatrix.cast<float>() * point;
            Eigen::Vector3f transformedPoint3f = transformedPoint.head<3>();

            // If color is invalid, use intensity for grayscale rendering
            bool isColorInvalid = (rData[i] == 255 && gData[i] == 255 && bData[i] == 255);
            Eigen::Vector3f color;
            if (isColorInvalid) {
                float grayscale = static_cast<float>(intensityData[i]);
                color = Eigen::Vector3f(grayscale, grayscale, grayscale);
            } else {
                // Use RGB values as they are
                color = Eigen::Vector3f(rData[i] / 255.0f, gData[i] / 255.0f, bData[i] / 255.0f);
            }

            // Store the transformed point and color in the main vectors directly
            #pragma omp critical
            {
                points.push_back(transformedPoint3f);
                colors.push_back(color);
            }
        }

        std::cout << "Transforming finished " << std::endl;

        // points = std::move(tempPoints);
        // colors = std::move(tempColors);

        // Subsample points and colors (no intensities buffer needed since it's incorporated into colors)
        SubsamplePointCloud(points, colors, subsampledPoints, subsampledColors, maxPoints);
        size_t subsize = subsampledPoints.size();

        std::cout << "Subsampling finished " << std::endl;
        // Create PangoPointCloud using only points and colors
        auto pangoCloudSub = std::make_unique<PangoPointCloud>(subsampledPoints, subsampledColors);
        auto pangoCloudFull = std::make_unique<PangoPointCloud>(points, colors);

        // Store scan data in the loadedScans map
        ScanData scanData;
        scanData.subsampledCloud = std::move(pangoCloudSub);
        scanData.fullCloud = std::move(pangoCloudFull);
        scanData.isRendered = true;

        loadedScans[scanKey] = std::move(scanData);

        std::cout << "Scan " << scanHeader.name << " loaded with " << subsize << " subsampled points.\n" << std::endl;
    }
}


struct FileEntry {
    std::string filepath;
    std::vector<std::string> scanNames;
    bool hasMultipleScans;
    bool checkbox;
    std::vector<bool> scanCheckboxes;

    // Constructor to initialize members
    FileEntry(const std::string& file, const std::vector<std::string>& scans, bool hasScans)
        : filepath(file),
          scanNames(scans),
          hasMultipleScans(hasScans),
          checkbox(false), // Default value for parent checkbox
          scanCheckboxes(scans.size(), false) // Initialize scanCheckboxes with the same size as scanNames, all set to false
    {}
};

void UpdateView(std::vector<double> recentScanPose, pangolin::OpenGlRenderState& s_cam) {
    Eigen::Vector4d poseHomogeneous(recentScanPose[0], recentScanPose[1], recentScanPose[2], 1.0);
    Eigen::Vector4d correctedPose = globalShift.inverse() * poseHomogeneous;
    double cameraX = correctedPose[0] + 20.0; 
    double cameraY = correctedPose[1] + 20.0; 
    double cameraZ = correctedPose[2] + 2.0;   
    
    s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(cameraX, cameraY, cameraZ, correctedPose[0], correctedPose[1], correctedPose[2], pangolin::AxisZ));
}

void ProcessSelectedScans(const std::vector<FileEntry>& fileEntries) {
    for (const auto& entry : fileEntries) {
        if (entry.hasMultipleScans) {
            for (size_t i = 0; i < entry.scanNames.size(); ++i) {
                std::string cacheKey = entry.filepath + "_" + entry.scanNames[i];
                
                // Check if the scan is in the loaded map and whether its point cloud is cleared
                auto scanIt = loadedScans.find(cacheKey);
                if (scanIt == loadedScans.end() || scanIt->second.subsampledCloud->getVertices().empty()) {
                    // Load the scan if not loaded or cleared, and the checkbox is checked
                    if (entry.scanCheckboxes[i]) {
                        readE57(entry.filepath, entry.scanNames[i]);
                    }
                    continue;
                } else {
                    // Update the render status based on the checkbox
                    scanIt->second.isRendered = entry.scanCheckboxes[i];
                }
            }
        } else {
            std::string cacheKey = entry.filepath + "_" + entry.scanNames[0];
            
            // Check if the scan is in the loaded map and whether its point cloud is cleared
            auto scanIt = loadedScans.find(cacheKey);
            if (scanIt == loadedScans.end() || scanIt->second.subsampledCloud->getVertices().empty()) {
                // Load the scan if not loaded or cleared, and the checkbox is checked
                if (entry.checkbox) {
                    readE57(entry.filepath, entry.scanNames[0]);
                }
                continue;
            } else {
                // Update the render status based on the checkbox
                scanIt->second.isRendered = entry.checkbox;
            }
        }
    }
}



void DrawAll() {
    for (const auto& scan : loadedScans) {
        const auto& scanData = scan.second;
        if (scanData.isRendered) {
            scanData.subsampledCloud->draw();
        }
    }
    if (!pangoCloudCube->getVertices().empty()) {
        pangoCloudCube->draw();
    }
}

// Function to create checkboxes for each E57 file and store them in the FileEntry structure
void CreateParentCheckboxes(std::vector<FileEntry>& fileEntries) {
    for (auto& entry : fileEntries) {
        // Create a parent checkbox for each file (using the base name of the file)
        std::string base_name = std::filesystem::path(entry.filepath).stem().string();
        // std::string checkbox_name = "ui." + base_name + (entry.hasMultipleScans ? " - Bundle" : " ");

        // Initialize the checkbox within the FileEntry and render it
        pangolin::Var<bool> parent_checkbox ("scan_list." + base_name, false, true);
        entry.checkbox = false;
        
    }
}


// Helper function to check if a variable exists in the Pangolin VarState
bool VarExists(const std::string& var_name) {
    return pangolin::VarState::I().Exists(var_name);
}

void RenderCheckboxesAndSublist(std::vector<FileEntry>& fileEntries) {
    for (auto& entry : fileEntries) {
        std::string base_name = std::filesystem::path(entry.filepath).stem().string();
        std::string parent_checkbox_name = base_name;
        
        // Create the parent checkbox
        pangolin::Var<bool> parent_checkbox("scan_list." + parent_checkbox_name, entry.checkbox, true);
        
        // Update the entry's checkbox state based on the parent checkbox value
        entry.checkbox = parent_checkbox;
        
        // If the parent checkbox is checked, create or update the child checkboxes
        if (entry.checkbox) {
            if (entry.hasMultipleScans) {
                for (size_t i = 0; i < entry.scanNames.size(); ++i) {
                    std::string child_checkbox_name = entry.scanNames[i];
                    
                    // Only create child checkbox if it doesn't already exist
                    if (!pangolin::VarState::I().Exists(child_checkbox_name)) {
                        pangolin::Var<bool> child_checkbox("scan_list." + child_checkbox_name, entry.scanCheckboxes[i], true);
                        // Store the initial state in the entry's scanCheckboxes vector
                        entry.scanCheckboxes[i] = child_checkbox;
                    }
                }
            } 
        } 
        // If parent checkbox is unchecked, remove all child checkboxes if they exist
        else {
            if (entry.hasMultipleScans) {
                for (size_t i = 0; i < entry.scanNames.size(); ++i) {
                    std::string child_checkbox_name = entry.scanNames[i];

                    // Retrieve the value before removing the checkbox
                    if (pangolin::VarState::I().Exists("scan_list." + child_checkbox_name)) {
                        pangolin::Var<bool> child_checkbox("scan_list." + child_checkbox_name);
                        // Store the current value of the checkbox before removing it
                        entry.scanCheckboxes[i] = child_checkbox;

                        // Now remove the checkbox from the UI
                        pangolin::VarState::I().Remove("scan_list." + child_checkbox_name);
                    }
                }
            }
        }
    }
}



// Function to read the E57 file and return scan names and a flag for multiple scans
FileEntry readE57Header(const std::string& filename) {
    bool hasMultipleScans = false;
    std::vector<std::string> scanNames;

    try {
        e57::Reader e57Reader(filename, e57::ReaderOptions());

        size_t scanCount = e57Reader.GetData3DCount();
        hasMultipleScans = (scanCount > 1);

        // Collect scan names
        for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
            e57::Data3D scanHeader;
            e57Reader.ReadData3D(scanIndex, scanHeader);

            // Add the scan name (or default to "Scan i" if name is empty)
            scanNames.push_back(scanHeader.name.empty() ? "Scan " + std::to_string(scanIndex + 1) : scanHeader.name);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading E57 file: " << e.what() << std::endl;
    }

    // Create and return FileEntry
    return FileEntry(filename, scanNames, hasMultipleScans);
}

struct Point {
    std::string name;
    Eigen::Vector3d coordinates;
};

std::vector<Point> readPointsFromFile(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error reading file: " << filename << std::endl;
        return points;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string name;
        double x, y, z;
        if (!(iss >> name >> x >> y >> z)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
        points.push_back({name, Eigen::Vector3d(x, y, z)});
    }
    return points;
}

struct Target {
    std::string name;
    Eigen::Vector3d local;
    Eigen::Vector3d global;
    bool selected;
};

Eigen::Matrix4d computeTransformationMatrix(const std::vector<Target> matchedTargets) {
    // Temporary vectors to store coordinates of selected targets
    std::vector<Eigen::Vector3d> localPoints;
    std::vector<Eigen::Vector3d> globalPoints;

    // Filter for selected points
    for (const auto& target : matchedTargets) {
        if (target.selected) {
            localPoints.push_back(target.local);
            globalPoints.push_back(target.global);
        }
    }

    // Check for sufficient selected points
    if (localPoints.size() < 3) {
        std::cerr << "Insufficient selected points for transformation." << std::endl;
        std::cout << "Transformation Matrix:\n" << Eigen::Matrix4d::Identity() << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    // Compute centroids of selected points
    Eigen::Vector3d centroidLocal = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroidGlobal = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < localPoints.size(); ++i) {
        centroidLocal += localPoints[i];
        centroidGlobal += globalPoints[i];
    }
    centroidLocal /= localPoints.size();
    centroidGlobal /= globalPoints.size();

    std::cout << "local centroid: " << centroidLocal << "\n" << std::endl;
    std::cout << "global centroid: " << centroidGlobal << "\n" << std::endl;

    // Compute centered vectors
    std::vector<Eigen::Vector3d> localCentered(localPoints.size());
    std::vector<Eigen::Vector3d> globalCentered(globalPoints.size());
    for (size_t i = 0; i < localPoints.size(); ++i) {
        localCentered[i] = localPoints[i] - centroidLocal;
        globalCentered[i] = globalPoints[i] - centroidGlobal;
    }

    // Compute cross-covariance matrix
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < localPoints.size(); ++i) {
        H += localCentered[i] * globalCentered[i].transpose();
    }

    // Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();

    // Ensure a proper rotation (determinant = 1)
    if (rotation.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        rotation = V * svd.matrixU().transpose();
    }

    // // Compute scale
    // double numerator = 0;
    // double denominator = 0;
    // for (size_t i = 0; i < localPoints.size(); ++i) {
    //     numerator += globalCentered[i].dot(rotation * localCentered[i]);
    //     denominator += localCentered[i].squaredNorm();
    // }
    double scale = 1.0; // fix scale to 1

    // Compute translation
    // Eigen::Vector3d translation = centroidGlobal - scale * rotation * centroidLocal;
    Eigen::Vector3d localTranslation = -rotation * centroidLocal;
    Eigen::Vector3d translation = centroidGlobal + localTranslation;    

    // Form the 4x4 transformation matrix
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = scale * rotation;
    transformation.block<3, 1>(0, 3) = translation;

    std::cout << "Transformation Matrix:\n" << transformation << std::endl;

    return transformation;
}



// Function to generate a UUID (unique for each scan)
std::string generateUniqueGUID(size_t scanIndex) {
    std::random_device rd;  // Seed for the random number generator
    std::mt19937 generator(rd());

    // Uniform distribution for hexadecimal digits
    std::uniform_int_distribution<int> distribution(0, 15);

    // Generate a 32-character hex string based on scanIndex as a seed
    std::stringstream ss;
    ss << std::hex;
    ss << std::setw(8) << std::setfill('0') << scanIndex;
    for (int i = 0; i < 24; ++i) {
        ss << std::hex << distribution(generator);
    }

    // Return the GUID string
    return ss.str();
}

void CheckSubBuffers(const e57::Data3DPointsDouble& buffers) {
    // Check for required sub-buffers
    if (!buffers.cartesianX) {
        std::cerr << "Error: 'cartesianX' buffer is not set." << std::endl;
    }
    if (!buffers.cartesianY) {
        std::cerr << "Error: 'cartesianY' buffer is not set." << std::endl;
    }
    if (!buffers.cartesianZ) {
        std::cerr << "Error: 'cartesianZ' buffer is not set." << std::endl;
    }

    // Check optional sub-buffers
    if (!buffers.intensity) {
        std::cout << "Optional: 'intensity' buffer is not set." << std::endl;
    }
    if (!buffers.colorRed || !buffers.colorGreen || !buffers.colorBlue) {
        std::cout << "Optional: One or more color buffers (red, green, blue) are not set." << std::endl;
    }
    if (!buffers.cartesianInvalidState) {
        std::cout << "Optional: 'cartesianInvalidState' buffer is not set." << std::endl;
    }
    if (!buffers.isColorInvalid) {
        std::cout << "Optional: 'isColorInvalid' buffer is not set." << std::endl;
    }
    if (!buffers.rowIndex) {
        std::cout << "Optional: 'rowIndex' buffer is not set." << std::endl;
    }
    // if (!buffers.)
    // Add more checks for other optional fields if needed
}

void InitializeBuffersFromScanHeader(e57::Data3DPointsDouble& buffers, const e57::Data3D& scanHeader) {
    // Required Cartesian coordinates
    buffers.cartesianX = new double[scanHeader.pointCount];
    buffers.cartesianY = new double[scanHeader.pointCount];
    buffers.cartesianZ = new double[scanHeader.pointCount];

    // Optional: Cartesian invalid state
    if (scanHeader.pointFields.cartesianInvalidStateField) {
        buffers.cartesianInvalidState = new int8_t[scanHeader.pointCount];
    }

    // Optional: Intensity
    if (scanHeader.pointFields.intensityField) {
        buffers.intensity = new double[scanHeader.pointCount];
        if (scanHeader.pointFields.isIntensityInvalidField) {
            buffers.isIntensityInvalid = new int8_t[scanHeader.pointCount];
        }
    }

    // Optional: Color
    if (scanHeader.pointFields.colorRedField && scanHeader.pointFields.colorGreenField && scanHeader.pointFields.colorBlueField) {
        buffers.colorRed = new uint16_t[scanHeader.pointCount];
        buffers.colorGreen = new uint16_t[scanHeader.pointCount];
        buffers.colorBlue = new uint16_t[scanHeader.pointCount];
        if (scanHeader.pointFields.isColorInvalidField) {
            buffers.isColorInvalid = new int8_t[scanHeader.pointCount];
        }
    }

    // Optional: Spherical coordinates
    if (scanHeader.pointFields.sphericalRangeField) {
        buffers.sphericalRange = new double[scanHeader.pointCount];
    }
    if (scanHeader.pointFields.sphericalAzimuthField) {
        buffers.sphericalAzimuth = new double[scanHeader.pointCount];
    }
    if (scanHeader.pointFields.sphericalElevationField) {
        buffers.sphericalElevation = new double[scanHeader.pointCount];
    }
    if (scanHeader.pointFields.sphericalInvalidStateField) {
        buffers.sphericalInvalidState = new int8_t[scanHeader.pointCount];
    }

    // Optional: Row/Column indices (useful for gridded data)
    if (scanHeader.pointFields.rowIndexField) {
        buffers.rowIndex = new int32_t[scanHeader.pointCount];
    }
    if (scanHeader.pointFields.columnIndexField) {
        buffers.columnIndex = new int32_t[scanHeader.pointCount];
    }

    // Optional: Return index and count (for multi-return sensors)
    if (scanHeader.pointFields.returnIndexField) {
        buffers.returnIndex = new int8_t[scanHeader.pointCount];
    }
    if (scanHeader.pointFields.returnCountField) {
        buffers.returnCount = new int8_t[scanHeader.pointCount];
    }

    // Optional: Timestamp
    if (scanHeader.pointFields.timeStampField) {
        buffers.timeStamp = new double[scanHeader.pointCount];
        if (scanHeader.pointFields.isTimeStampInvalidField) {
            buffers.isTimeStampInvalid = new int8_t[scanHeader.pointCount];
        }
    }

    // Optional: Surface normals (E57_EXT_surface_normals extension)
    if (scanHeader.pointFields.normalXField && scanHeader.pointFields.normalYField && scanHeader.pointFields.normalZField) {
        buffers.normalX = new float[scanHeader.pointCount];
        buffers.normalY = new float[scanHeader.pointCount];
        buffers.normalZ = new float[scanHeader.pointCount];
    }
}

void DeleteBuffers(e57::Data3DPointsDouble& buffers) {
    // Required Cartesian coordinates
    if (buffers.cartesianX) {
        delete[] buffers.cartesianX;
        buffers.cartesianX = nullptr; // Set pointer to nullptr to avoid dangling pointers
    }
    if (buffers.cartesianY) {
        delete[] buffers.cartesianY;
        buffers.cartesianY = nullptr;
    }
    if (buffers.cartesianZ) {
        delete[] buffers.cartesianZ;
        buffers.cartesianZ = nullptr;
    }

    // Optional: Cartesian invalid state
    if (buffers.cartesianInvalidState) {
        delete[] buffers.cartesianInvalidState;
        buffers.cartesianInvalidState = nullptr;
    }

    // Optional: Intensity
    if (buffers.intensity) {
        delete[] buffers.intensity;
        buffers.intensity = nullptr;
    }
    if (buffers.isIntensityInvalid) {
        delete[] buffers.isIntensityInvalid;
        buffers.isIntensityInvalid = nullptr;
    }

    // Optional: Color
    if (buffers.colorRed) {
        delete[] buffers.colorRed;
        buffers.colorRed = nullptr;
    }
    if (buffers.colorGreen) {
        delete[] buffers.colorGreen;
        buffers.colorGreen = nullptr;
    }
    if (buffers.colorBlue) {
        delete[] buffers.colorBlue;
        buffers.colorBlue = nullptr;
    }
    if (buffers.isColorInvalid) {
        delete[] buffers.isColorInvalid;
        buffers.isColorInvalid = nullptr;
    }

    // Optional: Spherical coordinates
    if (buffers.sphericalRange) {
        delete[] buffers.sphericalRange;
        buffers.sphericalRange = nullptr;
    }
    if (buffers.sphericalAzimuth) {
        delete[] buffers.sphericalAzimuth;
        buffers.sphericalAzimuth = nullptr;
    }
    if (buffers.sphericalElevation) {
        delete[] buffers.sphericalElevation;
        buffers.sphericalElevation = nullptr;
    }
    if (buffers.sphericalInvalidState) {
        delete[] buffers.sphericalInvalidState;
        buffers.sphericalInvalidState = nullptr;
    }

    // Optional: Row/Column indices
    if (buffers.rowIndex) {
        delete[] buffers.rowIndex;
        buffers.rowIndex = nullptr;
    }
    if (buffers.columnIndex) {
        delete[] buffers.columnIndex;
        buffers.columnIndex = nullptr;
    }

    // Optional: Return index and count
    if (buffers.returnIndex) {
        delete[] buffers.returnIndex;
        buffers.returnIndex = nullptr;
    }
    if (buffers.returnCount) {
        delete[] buffers.returnCount;
        buffers.returnCount = nullptr;
    }

    // Optional: Timestamp
    if (buffers.timeStamp) {
        delete[] buffers.timeStamp;
        buffers.timeStamp = nullptr;
    }
    if (buffers.isTimeStampInvalid) {
        delete[] buffers.isTimeStampInvalid;
        buffers.isTimeStampInvalid = nullptr;
    }

    // Optional: Surface normals
    if (buffers.normalX) {
        delete[] buffers.normalX;
        buffers.normalX = nullptr;
    }
    if (buffers.normalY) {
        delete[] buffers.normalY;
        buffers.normalY = nullptr;
    }
    if (buffers.normalZ) {
        delete[] buffers.normalZ;
        buffers.normalZ = nullptr;
    }
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}


std::vector<Eigen::Vector3d> transformedPoints;

void drawResult(const std::vector<Target>& matchedTargets, int scale, int err_lim, Eigen::Vector3d centroid, bool draw_err) {
    const float sphereRadius = (static_cast<float>(err_lim) / 100.0f) * static_cast<float>(scale); // Scaled radius for spheres
    float error_lim = static_cast<float>(err_lim);
    // First, draw all opaque objects (dots and lines) to maintain depth
    glEnable(GL_DEPTH_TEST);

    std::vector<Eigen::Vector3d> dir;

    for (size_t i = 0; i < transformedPoints.size(); ++i) {
        const auto& transformed = transformedPoints[i];
        const auto& target = matchedTargets[i];
        const auto& global = target.global;

        // Draw a black dot at the transformed point
        glColor3f(0.0, 0.0, 0.0);
        glPointSize(5.0f);
        glBegin(GL_POINTS); // Begin drawing points
        glVertex3f(transformed(0), transformed(1), transformed(2)); // Point at transformed position
        glEnd();

        // Draw an arrow from the transformed point to the target's global position
        Eigen::Vector3d diff = global - transformed;
        dir.push_back(diff.normalized());
        double oLength = diff.norm();
        double eLength = oLength * static_cast<double>(scale);
        Eigen::Vector3d extended = transformed + dir.back() * eLength;

        // Set color and line width based on selection
        if (target.selected) {
            glLineWidth(3.0f); // Thicker line for selected target arrow
            glColor4f(0.949, 0.549, 0.051, 0.8); // Selected color
        } else {
            glLineWidth(1.5f); // Default line width
            glColor4f(0.949, 0.549, 0.051, 0.4); // Unselected color
        }

        glBegin(GL_LINES);
        glVertex3d(transformed(0), transformed(1), transformed(2));
        glVertex3d(extended(0), extended(1), extended(2));
        glEnd();
       
    }

    // Now draw transparent objects (spheres)
    glDisable(GL_DEPTH_TEST); // Disable depth testing for transparency
    glEnable(GL_BLEND); // Enable blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set blending function

    for (size_t i = 0; i < transformedPoints.size(); ++i) {
        const auto& transformed = transformedPoints[i];
        const auto& target = matchedTargets[i];

        if (draw_err) {
            // Set color based on whether the target is selected and whether the error is greater than the set limit
            Eigen::Vector3d diff = target.global - transformed;
            double oLength = diff.norm();

            if (target.selected) {
                if (oLength <= (error_lim / 100.0)) {
                    glColor4f(0.0, 1.0, 0.0, 0.2); // Green, 40% transparent for selected targets
                } else {
                    glColor4f(1.0, 0.0, 0.0, 0.1); // Red, 40% transparent for out of limit
                }
            } else {
                if (oLength <= (error_lim / 100.0)) {
                    glColor4f(0.0, 0.0, 1.0, 0.1); // Blue, 20% transparent for unselected targets
                } else {
                    glColor4f(1.0, 0.0, 0.0, 0.05); // Red, 20% transparent for out of limit
                }
            }

            // Draw transparent sphere around the transformed point
            GLUquadric* quad = gluNewQuadric();
            glPushMatrix();
            glTranslated(transformed(0), transformed(1), transformed(2));
            gluSphere(quad, sphereRadius, 20, 20); // Sphere at transformed point
            glPopMatrix();
            gluDeleteQuadric(quad); // Clean up quadric after each sphere

            
        }

        // Calculate the label position on the mirrored side of the arrow
        double yOffset = draw_err ? sphereRadius + 0.1 : 0.4; // Determine the Y offset based on the draw_err flag

        Eigen::Vector3d labelPosition = transformed - dir[i]* yOffset; // Position the label slightly behind the arrow

        // Draw the label
        glColor3f(0.0, 0.0, 0.0); // Black color for labels
        pangolin::GlText label = pangolin::default_font().Text(target.name.c_str());
        label.Draw(labelPosition(0), labelPosition(1), labelPosition(2)); // Draw the label

        
    }

    // Restore state
    glEnable(GL_DEPTH_TEST); // Re-enable depth testing for other objects
    glDisable(GL_BLEND); // Disable blending when done with transparent objects
}

// GUI and Pangolin window setup
Eigen::Matrix4d trafoViz(const Eigen::Matrix4d& initialTransformationMatrix, std::vector<Target> matchedTargets) {
    Eigen::Matrix4d transformationMatrix = initialTransformationMatrix;
    // Define window size
    int window_width = 1000;
    int window_height = 800;

    float aspect = static_cast<float>(window_width) / window_height;

    // Create window and setup camera
    pangolin::CreateWindowAndBind("Transformation Visualization", window_width, window_height);


    glEnable(GL_DEPTH_TEST);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Compute centroid and radius of points to position camera
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double maxDistance = 0.0;
    for (const auto& point : transformedPoints) {
        centroid += point;
    }
    centroid /= transformedPoints.size();

    for (const auto& point : transformedPoints) {
        double distance = (point - centroid).norm();
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }

    // Set up camera based on centroid and bounding radius
    pangolin::OpenGlRenderState si_cam(
        pangolin::ProjectionMatrix(window_width, window_height, 700, 650, window_width / 2.0, window_height / 2.0, 0.2, 500), // Adjust far plane
        pangolin::ModelViewLookAt(centroid(0), centroid(1), centroid(2) + maxDistance * 3, centroid(0), centroid(1), centroid(2), pangolin::AxisY)
    );

    pangolin::View& di_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 200.0f/window_width, 1.0, -aspect) // Adjust left bound to account for panel width
        .SetHandler(new pangolin::Handler3D(si_cam)
    );



    pangolin::CreatePanel("ui_trafo").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(250));
    pangolin::Var<int> scale_var("ui_trafo.Error scale", 100, 1, 1000, true);
    pangolin::Var<int> err_int("ui_trafo.Error limit [cm]",5,0,60);
    pangolin::Var<bool> draw_err("ui_trafo.Draw error limits", true, true);
    pangolin::Var<bool> calc_button("ui_trafo.Recalculate", false, false);
    for (size_t i = 0; i < matchedTargets.size(); ++i) {
        pangolin::Var<bool> pointCheckbox("ui_trafo." + matchedTargets[i].name, true, true);
    }
    pangolin::Var<bool> trafo_button("ui_trafo.Transform", false, false);

    bool transformationCompleted = false;
    
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        int scale = scale_var.Get();
        int err_lim = err_int.Get();

        UpdateWindowSize(window_width,window_height,aspect);
        pangolin::DisplayBase().ActivateScissorAndClear(si_cam);

        if (pangolin::Pushed(trafo_button)) {
            transformationCompleted = true;
            return transformationMatrix;
        }

        if (pangolin::Pushed(calc_button)) {
            transformationMatrix = computeTransformationMatrix(matchedTargets);
            transformedPoints.clear();

            // Set fixed-point notation and precision
            std::cout << std::fixed << std::setprecision(3);

            for (const auto& point : matchedTargets) {
                // Prepare data for output
                Eigen::Vector4d localHomo(point.local(0), point.local(1), point.local(2), 1.0);
                Eigen::Vector3d transformed = (transformationMatrix * localHomo).head<3>();
                transformedPoints.push_back(transformed);

                Eigen::Vector3d difference = point.global - transformed;
                double absDifference = difference.norm();

                // Format and align output
                std::cout << std::left << std::setw(8) << point.name << "|"  // Left-align point name
                        << "diff:"
                        << std::right << std::setw(6) << difference(0) << " "
                        << std::setw(6) << difference(1) << " "
                        << std::setw(6) << difference(2) << " [m] | "
                        << "abs. diff: " << std::setw(6) << absDifference << " [m]"
                        << (point.selected ? "" : " (not selected)")  // Add "(not selected)" if point is not selected
                        << std::endl;
            }
        }



        for (size_t i = 0; i < matchedTargets.size(); ++i) {
            pangolin::Var<bool> pointCheckbox("ui_trafo." + matchedTargets[i].name);
            matchedTargets[i].selected = pointCheckbox;
        }

        di_cam.Activate(si_cam);

        drawResult(matchedTargets, scale, err_lim, centroid, draw_err);

        pangolin::FinishFrame();
    }

    // If the loop exits without pressing Transform, print message and abort
    if (!transformationCompleted) {
        std::cerr << "Transformation aborted." << std::endl;
        exit(EXIT_FAILURE); // Stop the entire program
    }

    return transformationMatrix;
}

// Function to convert JSON to Eigen::Matrix
Eigen::MatrixXd JsonToMatrix(const nlohmann::json& jsonMatrix) {
    int rows = jsonMatrix.size();
    int cols = jsonMatrix[0].size();

    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = jsonMatrix[i][j];
        }
    }

    return matrix;
}

// Function to read JSON and return a map of ID to Eigen::Matrix4d
std::unordered_map<std::string, Eigen::Matrix4d> LoadGlobalPoses(const std::string& poseFile) {
    // Open the JSON file
    std::ifstream inFile(poseFile);
    if (!inFile) {
        throw std::runtime_error("Failed to open file: " + poseFile);
    }

    // Parse JSON
    nlohmann::json jsonData;
    inFile >> jsonData;

    // Create the map to hold the results
    std::unordered_map<std::string, Eigen::Matrix4d> globalPoses;

    // Iterate through the JSON object
    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        const std::string& id = it.key();                      // Extract the ID
        Eigen::Matrix4d globalPose = JsonToMatrix(it.value()["GlobalPose"]); // Convert GlobalPose to Eigen::Matrix4d
        globalPoses[id] = globalPose;                          // Store in the map
    }

    return globalPoses;
}



void transformE57(std::vector<FileEntry> fileEntries) {
    try {
        pangolin::DestroyWindow("GeoRefHut");
        // std::unordered_map<std::string, ScanData> emptyMap;
        // loadedScans.swap(emptyMap);

        // Read local points from the local.txt file
        std::vector<Point> localPointsMap = readPointsFromFile("local.txt");
        if (localPointsMap.empty()) {
            std::cerr << "No local points found." << std::endl;
            return;
        }

        // Select the target coordinates file using tinyfd
        const char* targetCoordinatesFile = tinyfd_openFileDialog(
            "Select target coordinates file",
            curr_path.c_str(), // Default path (empty for user to choose)
            1, // Number of filter patterns
            nullptr, // No specific filter
            "Text files (*.txt)", // Single filter description
            0 // Don't allow multiple selects
        );
        

        if (!targetCoordinatesFile) {
            std::cerr << "No target file selected." << std::endl;
            return;
        }
        std::filesystem::path TfilePath(targetCoordinatesFile);
        curr_path = TfilePath.parent_path();

        // Read global points from the selected target file
        std::vector<Point> globalPointsMap = readPointsFromFile(targetCoordinatesFile);
        if (globalPointsMap.empty()) {
            std::cerr << "No global points found." << std::endl;
            return;
        }

        std::vector<Target> matchedTargets;

        for (const auto& localPoint : localPointsMap) {
            for (const auto& globalPoint : globalPointsMap) {
                if (localPoint.name == globalPoint.name) {
                    // Create a Target instance with the matched local and global points
                    Target target;
                    target.name = localPoint.name;
                    target.local = localPoint.coordinates;
                    target.global = globalPoint.coordinates;
                    target.selected = true;

                    // Add the target to the matched targets list
                    matchedTargets.push_back(target);
                }
            }
        }

        if (matchedTargets.size() < 3) {
            std::cerr << "Insufficient corresponding points for transformation." << std::endl;
            return;
        }

        
        // Compute transformation matrix
        Eigen::Matrix4d initialTransformationMatrix = computeTransformationMatrix(matchedTargets);

        for (const auto& point : matchedTargets) {
            Eigen::Vector4d localHomog(point.local(0), point.local(1), point.local(2), 1.0);
            Eigen::Vector3d transformed = (initialTransformationMatrix* localHomog).head<3>();
            transformedPoints.push_back(transformed);
            Eigen::Vector3d difference = point.global - transformed;
            double absDifference = difference.norm();
            std::cout << std::fixed << std::setprecision(3);
            // std::cout << point.name 
            //         << "; diff: " << difference.transpose() << " [m] ; abs. diff: " << absDifference << " [m]" << std::endl;

            // Format and align output
            std::cout << std::left << std::setw(8) << point.name << "|"  // Left-align point name
                    << "diff:"
                    << std::right << std::setw(6) << difference(0) << " "
                    << std::setw(6) << difference(1) << " "
                    << std::setw(6) << difference(2) << " [m] | "
                    << "abs. diff: " << std::setw(6) << absDifference << " [m]"
                    << std::endl;
        }
        
        Eigen::Matrix4d transformationMatrix = trafoViz(initialTransformationMatrix,matchedTargets);

        // Create a single output E57 file
        // std::filesystem::path firstInputPath(fileEntries[0].filepath);
        // std::string outputFilename;
        // if (fileEntries.size() < 2) {
        //     outputFilename = firstInputPath.stem().string() + "_georef.e57";  // Append _georef.e57
        // } else {
        //     outputFilename = firstInputPath.stem().string() + "_merged_georef.e57";  // Append _merged_georef.e57
        // }
        
        // std::filesystem::path outputPath = firstInputPath.parent_path() / outputFilename;
        // std::cout << "Writing merged E57 to: " << outputPath << std::endl;

        // // Create the E57 Writer for the single output file
        // e57::WriterOptions writerOptions;
        // e57::Writer e57Writer(outputPath.string().c_str(), writerOptions);
        // if (!e57Writer.IsOpen()) {
        //     std::cerr << "Failed to open E57 writer for " << outputFilename << std::endl;
        //     return;
        // }
        
        size_t guidCounter = 1;

        std::unordered_map<std::string, Eigen::Matrix4d> optPoses;
        std::ifstream fin("optimized_poses.json");
        if (!posesOptimized && fin) {
            optPoses = LoadGlobalPoses("optimized_poses.json");
        } else {
            for (auto& pose : scanPositions) {
                optPoses[pose.key] = pose.GlobalPose;
            }
        }
        

        // Process each input E57 file
        for (const auto& file : fileEntries) {

            std::filesystem::path firstInputPath(file.filepath);
            std::string outputFilename;
            if (fileEntries.size() < 2) {
                outputFilename = firstInputPath.stem().string() + "_georef.e57";  // Append _georef.e57
            } else {
                outputFilename = firstInputPath.stem().string() + "_georef.e57";  // Append _merged_georef.e57
            }

            std::filesystem::path outputPath = firstInputPath.parent_path() / outputFilename;
            std::cout << "Writing E57 to: " << outputPath << std::endl;

            // Create the E57 Writer for the single output file
            e57::WriterOptions writerOptions;
            e57::Writer e57Writer(outputPath.string().c_str(), writerOptions);
            if (!e57Writer.IsOpen()) {
                std::cerr << "Failed to open E57 writer for " << outputFilename << std::endl;
                return;
            }


            // Open the current E57 file for reading
            e57::Reader e57Reader(file.filepath, e57::ReaderOptions());
            std::cout << "Reading E57 file: " << file.filepath << std::endl;

            size_t scanCount = e57Reader.GetData3DCount();
            std::cout << "Number of scans in the file: " << scanCount << std::endl;

            // Process each scan in the current file
            for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
                e57::Data3D scanHeader;
                e57Reader.ReadData3D(scanIndex, scanHeader);

                std::string scanKey = file.filepath + "_" + scanHeader.name;

                // Apply the transformation to the scan pose
                Eigen::Matrix4d poseMatrix = optPoses[scanKey];
                Eigen::Matrix4d transformedPoseMatrix = transformationMatrix * poseMatrix;

                // Update the scan header with the new pose
                Eigen::Quaterniond eigenQuat(transformedPoseMatrix.block<3, 3>(0, 0));
                e57::Quaternion e57Quat;
                e57Quat.w = eigenQuat.w();
                e57Quat.x = eigenQuat.x();
                e57Quat.y = eigenQuat.y();
                e57Quat.z = eigenQuat.z();
                scanHeader.pose.rotation = e57Quat;
                scanHeader.pose.translation.x = transformedPoseMatrix(0, 3);
                scanHeader.pose.translation.y = transformedPoseMatrix(1, 3);
                scanHeader.pose.translation.z = transformedPoseMatrix(2, 3);


                // Assign a unique GUID for each scan
                scanHeader.guid = generateUniqueGUID(guidCounter);
                std::cout << "scan GUID: " << scanHeader.guid << std::endl;
                guidCounter++;

                std::cout << "Point count in scan header: " << scanHeader.pointCount << std::endl;


                // Read the point data
                e57::Data3DPointsData_t<double> buffers(scanHeader);

                // e57::Data3DPointsDouble buffers;
                // InitializeBuffersFromScanHeader(buffers, scanHeader); // TODO: 
                e57::CompressedVectorReader dataReader = e57Reader.SetUpData3DPointsData(scanIndex, scanHeader.pointCount, buffers);
                dataReader.read();
                

                // Write the transformed scan data to the output file
                int64_t dataIndex = e57Writer.NewData3D(scanHeader);  // Create a new Data3D block in the output file
                e57::CompressedVectorWriter dataWriter = e57Writer.SetUpData3DPointsData(dataIndex, scanHeader.pointCount, buffers);
                dataWriter.write(scanHeader.pointCount);
                if (dataIndex != scanHeader.pointCount) {
                    std::cerr << "Error: Not all points were written!" << std::endl;
                }
                dataWriter.close();

                std::cout << "Processed scan index " << scanIndex << " from file: " << file.filepath << std::endl;

                e57Writer.Close();
                if (fileEntries.size() < 2) {            
                    std::cout << "Transformed E57 point cloud written to " << outputPath.string() << std::endl;
                } else {
                    std::cout << "Merged and transformed E57 point cloud written to " << outputPath.string() << std::endl;
                }
            }
        }

        // Close the writer after all scans are written
        // e57Writer.Close();
        // if (fileEntries.size() < 2) {            
        //     std::cout << "Transformed E57 point cloud written to " << outputPath.string() << std::endl;
        // } else {
        //     std::cout << "Merged and transformed E57 point cloud written to " << outputPath.string() << std::endl;
        // }
        

    } catch (const e57::E57Exception& e) {
        std::cerr << e.what() << ": " << e.errorStr() << std::endl;
        return;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }
}


using namespace e57;
void printTranslation(const Translation& t) {
    std::cout << "Translation:\n";
    std::cout << "  x: " << t.x << "\n";
    std::cout << "  y: " << t.y << "\n";
    std::cout << "  z: " << t.z << "\n";
}

void printQuaternion(const Quaternion& q) {
    std::cout << "Quaternion:\n";
    std::cout << "  w: " << q.w << "\n";
    std::cout << "  x: " << q.x << "\n";
    std::cout << "  y: " << q.y << "\n";
    std::cout << "  z: " << q.z << "\n";
}

void printRigidBodyTransform(const RigidBodyTransform& rbt) {
    std::cout << "RigidBodyTransform:\n";
    printQuaternion(rbt.rotation);
    printTranslation(rbt.translation);
}

void printCartesianBounds(const CartesianBounds& cb) {
    std::cout << "CartesianBounds:\n";
    std::cout << "  xMinimum: " << cb.xMinimum << "\n";
    std::cout << "  xMaximum: " << cb.xMaximum << "\n";
    std::cout << "  yMinimum: " << cb.yMinimum << "\n";
    std::cout << "  yMaximum: " << cb.yMaximum << "\n";
    std::cout << "  zMinimum: " << cb.zMinimum << "\n";
    std::cout << "  zMaximum: " << cb.zMaximum << "\n";
}

void printSphericalBounds(const SphericalBounds& sb) {
    std::cout << "SphericalBounds:\n";
    std::cout << "  rangeMinimum: " << sb.rangeMinimum << "\n";
    std::cout << "  rangeMaximum: " << sb.rangeMaximum << "\n";
    std::cout << "  elevationMinimum: " << sb.elevationMinimum << "\n";
    std::cout << "  elevationMaximum: " << sb.elevationMaximum << "\n";
    std::cout << "  azimuthStart: " << sb.azimuthStart << "\n";
    std::cout << "  azimuthEnd: " << sb.azimuthEnd << "\n";
}

void printLineGroupRecord(const LineGroupRecord& lgr) {
    std::cout << "LineGroupRecord:\n";
    std::cout << "  idElementValue: " << lgr.idElementValue << "\n";
    std::cout << "  startPointIndex: " << lgr.startPointIndex << "\n";
    std::cout << "  pointCount: " << lgr.pointCount << "\n";
    printCartesianBounds(lgr.cartesianBounds);
    printSphericalBounds(lgr.sphericalBounds);
}

void printDateTime(const DateTime& dt) {
    std::cout << "DateTime:\n";
    std::cout << "  dateTimeValue: " << dt.dateTimeValue << "\n";
    std::cout << "  isAtomicClockReferenced: " << dt.isAtomicClockReferenced << "\n";
}

void printE57Root(const Data3D& root) {
    std::cout << "E57Root:\n";
    // std::cout << "  formatName: " << root.guid << "\n";
    std::cout << "  guid: " << root.guid << "\n";
    std::cout << "  sensorFirmware: " << root.sensorFirmwareVersion << "\n";
    std::cout << "  sensorHardware: " << root.sensorHardwareVersion << "\n";
    std::cout << "  sensorModel: " << root.sensorModel << "\n";
    // printDateTime(root.);
    std::cout << "  sensorSerailnumber: " << root.sensorSerialNumber << "\n";
    std::cout << "  sensorSoftwareVersion: " << root.sensorSoftwareVersion << "\n";
    std::cout << "  sensorVendor: " << root.sensorVendor << "\n";
    // std::cout << "  sensorVendor: " << root. << "\n";
}



void exportE57(std::vector<FileEntry> fileEntries) {
    // Create a single output E57 file
    std::filesystem::path firstInputPath(fileEntries[0].filepath);
    std::string outputFilename;
    if (fileEntries.size() < 2) {
        outputFilename = firstInputPath.stem().string() + "_subset.e57";  
    } else {
        outputFilename = firstInputPath.stem().string() + "_merged_subset.e57";  
    }
    
    std::filesystem::path outputPath = firstInputPath.parent_path() / outputFilename;
    std::cout << "Writing merged E57 to: " << outputPath << std::endl;

    // Create the E57 Writer for the single output file
    e57::WriterOptions writerOptions;
    e57::Writer e57Writer(outputPath.string().c_str(), writerOptions);
    if (!e57Writer.IsOpen()) {
        std::cerr << "Failed to open E57 writer for " << outputFilename << std::endl;
        return;
    }
    
    size_t guidCounter = 1;
    

    // Process each input E57 file
    for (const auto& file : fileEntries) {
        // Open the current E57 file for reading
        // skip if not selected
        if (!file.checkbox) {
            continue;
        }
        e57::Reader e57Reader(file.filepath, e57::ReaderOptions());
        std::cout << "Reading E57 file: " << file.filepath << std::endl;

        size_t scanCount = e57Reader.GetData3DCount();
        std::cout << "Number of scans in the file: " << scanCount << std::endl;

        // Process each scan in the current file
        for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
            e57::Data3D scanHeader;
            e57Reader.ReadData3D(scanIndex, scanHeader);

            std::string scanKey = file.filepath + "_" + scanHeader.name;
            bool skipscan = false;
            // // skip if not rendered
            // if (!loadedScans[scanKey].isRendered) {
            //     continue;
            // }

            // skip if not selected
            if (file.hasMultipleScans) {
                for (size_t i = 0; i < file.scanCheckboxes.size(); i++) {
                    if ((file.filepath + "_" + file.scanNames[i]) == scanKey) {
                        if (!file.scanCheckboxes[i]) {
                            skipscan = true;
                            break;
                        }
                    }
                }
            }
            
            if (skipscan) {
                continue;
            }

            std::unordered_map<std::string, Eigen::Matrix4d> optPoses;
            std::ifstream fin("optimized_poses.json");
            if (!posesOptimized && fin) {
                optPoses = LoadGlobalPoses("optimized_poses.json");
            } else {
                for (auto& pose : scanPositions) {
                    optPoses[pose.key] = pose.GlobalPose;
                }
            }


            scanHeader.sphericalBounds.rangeMaximum = 5000.0;
            scanHeader.sensorSoftwareVersion = "";

            Eigen::Matrix4d poseMatrix = optPoses[scanKey];
            // Eigen::Matrix4d transformedPoseMatrix = transformationMatrix * poseMatrix;

            // Update the scan header with the new pose
            Eigen::Quaterniond eigenQuat(poseMatrix.block<3, 3>(0, 0));
            e57::Quaternion e57Quat;
            e57Quat.w = eigenQuat.w();
            e57Quat.x = eigenQuat.x();
            e57Quat.y = eigenQuat.y();
            e57Quat.z = eigenQuat.z();
            scanHeader.pose.rotation = e57Quat;
            scanHeader.pose.translation.x = poseMatrix(0, 3);
            scanHeader.pose.translation.y = poseMatrix(1, 3);
            scanHeader.pose.translation.z = poseMatrix(2, 3);
            

            // Assign a unique GUID for each scan
            scanHeader.guid = generateUniqueGUID(guidCounter);
            // std::cout << "scan GUID: " << scanHeader.guid << std::endl;
            guidCounter++;

            std::cout << "Point count in scan header: " << scanHeader.pointCount << std::endl;
            
            // Read the point data
            e57::Data3DPointsData_t<double> buffers(scanHeader);
            // e57::Data3DPointsDouble buffers;
            // InitializeBuffersFromScanHeader(buffers, scanHeader); 
            e57::CompressedVectorReader dataReader = e57Reader.SetUpData3DPointsData(scanIndex, scanHeader.pointCount, buffers);
            dataReader.read();
            

            // Write the transformed scan data to the output file
            int64_t dataIndex = e57Writer.NewData3D(scanHeader);  // Create a new Data3D block in the output file
            e57::CompressedVectorWriter dataWriter = e57Writer.SetUpData3DPointsData(dataIndex, scanHeader.pointCount, buffers);
            dataWriter.write(scanHeader.pointCount);
            dataWriter.close();

            std::cout << "Processed scan index " << scanIndex << " from file: " << file.filepath << std::endl;
        }
    }

    // Close the writer after all scans are written
    e57Writer.Close();
    if (fileEntries.size() < 2) {            
        std::cout << "E57 point cloud written to " << outputPath.string() << std::endl;
    } else {
        std::cout << "Merged E57 point cloud written to " << outputPath.string() << std::endl;
    }
}

void mergeE57() {
    try {
        std::unordered_map<std::string, ScanData> emptyMap;
        loadedScans.swap(emptyMap);
        const char* e57FileFilterPatterns[] = {"*.e57"};
        const char* inputE57Filenames = tinyfd_openFileDialog(
            "Select E57 files to merge",
            curr_path.c_str(), // Default path (empty for user to choose)
            1, // Number of filter patterns
            e57FileFilterPatterns, // Filter patterns for E57 files
            "E57 files (*.e57)", // Single filter description
            1 // Allow multiple selects
        );

        if (!inputE57Filenames) {
            std::cerr << "No E57 files selected." << std::endl;
            return;
        }

        // Split the returned file paths by the pipe character '|'
        std::vector<std::string> e57Files = splitString(inputE57Filenames, '|');
        if (e57Files.empty()) {
            std::cerr << "No valid E57 files found after split." << std::endl;
            return;
        }

        std::filesystem::path EfilePath(e57Files[0]);
        curr_path = EfilePath.parent_path();

        // Create a single output E57 file
        std::filesystem::path firstInputPath(e57Files[0]);
        std::string outputFilename = firstInputPath.stem().string() + "_merged.e57"; 
        std::filesystem::path outputPath = firstInputPath.parent_path() / outputFilename;
        std::cout << "Writing merged E57 to: " << outputPath << std::endl;

        // Create the E57 Writer for the single output file
        e57::WriterOptions writerOptions;
        e57::Writer e57Writer(outputPath.string().c_str(), writerOptions);
        if (!e57Writer.IsOpen()) {
            std::cerr << "Failed to open E57 writer for " << outputFilename << std::endl;
            return;
        }

        size_t guidCounter = 1;

        // Process each input E57 file
        for (const auto& inputE57Filename : e57Files) {
            // Open the current E57 file for reading
            e57::Reader e57Reader(inputE57Filename.c_str(), e57::ReaderOptions());
            std::cout << "Reading E57 file: " << inputE57Filename << std::endl;

            size_t scanCount = e57Reader.GetData3DCount();
            std::cout << "Number of scans in the file: " << scanCount << std::endl;

            

            // Process each scan in the current file
            for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
                e57::Data3D scanHeader;
                e57Reader.ReadData3D(scanIndex, scanHeader);

                // Assign a unique GUID for each scan
                // scanHeader.guid = generateUniqueGUID(scanIndex);
                scanHeader.guid = generateUniqueGUID(guidCounter);
                std::cout << "scan GUID: " << scanHeader.guid << std::endl;
                guidCounter++;

                // Read the point data
                e57::Data3DPointsDouble buffers;
                // InitializeBuffersFromScanHeader(buffers, scanHeader);
                e57::CompressedVectorReader dataReader = e57Reader.SetUpData3DPointsData(scanIndex, scanHeader.pointCount, buffers);
                dataReader.read();

                // Write the transformed scan data to the output file
                int64_t dataIndex = e57Writer.NewData3D(scanHeader);  // Create a new Data3D block in the output file
                e57::CompressedVectorWriter dataWriter = e57Writer.SetUpData3DPointsData(dataIndex, scanHeader.pointCount, buffers);
                dataWriter.write(scanHeader.pointCount);
                dataWriter.close();

                // Clean up the memory used for buffers
                // DeleteBuffers(buffers);

                std::cout << "Merged scan index " << scanIndex << " from file: " << inputE57Filename << std::endl;
            }
        }

        // Close the writer after all scans are written
        e57Writer.Close();
        std::cout << "Merged E57 point cloud written to " << outputPath.string() << std::endl;

    } catch (const e57::E57Exception& e) {
        std::cerr << e.what() << ": " << e.errorStr() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }
}



std::vector<ScanPosition> ReadAllScanPositions(const std::vector<FileEntry>& fileEntries) {

    // Iterate through all files and read scan positions
    for (const auto& fileEntry : fileEntries) {
        e57::Reader e57Reader(fileEntry.filepath.c_str(), e57::ReaderOptions());
        size_t scanCount = e57Reader.GetData3DCount();

        // Iterate through each scan in the file
        for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
            e57::Data3D scanHeader;
            e57Reader.ReadData3D(scanIndex, scanHeader);

            // Get the pose matrix (extract translation part)
            Eigen::Matrix4d globalPose = GetPoseMatrix(scanHeader);
            Eigen::Matrix4d poseMatrix = globalShift.inverse() * globalPose;
            // poseMatrix = globalShift.inverse() * poseMatrix;
            Eigen::Vector3f scanPosition = poseMatrix.block<3, 1>(0, 3).cast<float>();

            // Store the scan position, pose matrix, and its name in ScanPosition structure
            scanPositions.push_back({
                scanPosition,                   // Position
                fileEntry.scanNames[scanIndex], // Name
                fileEntry.filepath + "_" + fileEntry.scanNames[scanIndex], // full key
                poseMatrix,
                globalPose                      // Pose
            });
        }
    }

    return scanPositions;
}


void DrawScanPositions(const std::vector<ScanPosition>& scanPositions) {
    for (const auto& scan : scanPositions) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // Draw the scan position as a small sphere using GLU quadric
        glColor4f(1.0, 0.0, 0.0, 0.6);  // Red color for the points
        
        // Create a quadric object for drawing spheres
        GLUquadric* quad = gluNewQuadric();
        
        // Translate to the scan position before drawing the sphere
        glPushMatrix();  // Save current transformation matrix
        glTranslatef(scan.positionLocal[0], scan.positionLocal[1], scan.positionLocal[2]);

        // Draw the sphere at the translated position with a radius of 0.2
        gluSphere(quad, 0.2, 20, 20);

        glPopMatrix();  // Restore the original transformation matrix

        gluDeleteQuadric(quad);  // Delete the quadric object after use

        // Draw the scan name as a label
        if (draw_labels) {
            glColor4f(0.0f, 0.0f, 0.0f, 1.0f); 
            pangolin::GlText label = pangolin::default_font().Text(scan.name.c_str());
            label.Draw(scan.positionLocal[0], scan.positionLocal[1], scan.positionLocal[2]);
        }
    } 


    if (draw_links) {
        // Draw lines between scan positions based on `lines` pairs
        glLineWidth(2.0f);  // Set the line width
        glColor4f(0.0, 0.0, 1.0, 0.8);  // Blue color for the lines with some transparency

        glBegin(GL_LINES);
        for (const auto& line : lines) {
            const Eigen::Vector3f& start = line.first;  // Start point of the line
            const Eigen::Vector3f& end = line.second;   // End point of the line

            glVertex3f(start[0], start[1], start[2]);  // Specify the start vertex
            glVertex3f(end[0], end[1], end[2]);        // Specify the end vertex
        }
        glEnd();
    }
}




void setglobalShift(const std::vector<FileEntry>& fileEntries) {
    e57::Reader e57Reader(fileEntries[0].filepath.c_str(), e57::ReaderOptions());
    e57::Data3D scanHeader;
    e57Reader.ReadData3D(0, scanHeader);
    globalShift = GetPoseMatrix(scanHeader);
}

std::pair<std::string, std::string> ExtractFilepathAndSuffix(const std::string& filepath) {
    size_t lastDot = filepath.find_last_of('.');
    if (lastDot == std::string::npos) {
        return {filepath, ""}; // No dot found, return the full filepath and an empty suffix
    }

    // Find ".e57_" in the filepath
    size_t e57Pos = filepath.find(".e57_", lastDot);
    if (e57Pos == std::string::npos) {
        return {filepath, ""}; // No ".e57_" found, return the full filepath and an empty suffix
    }

    // Separate the filepath and suffix
    std::string baseFilepath = filepath.substr(0, e57Pos + 4); // Include ".e57"
    std::string suffix = filepath.substr(e57Pos + 5); // Extract everything after ".e57_"

    return {baseFilepath, suffix};
}


struct ICPResult {
    std::string sourceName;
    std::string targetName;
    small_gicp::RegistrationResult result; // Includes T, num_inliers, etc.
    Eigen::Matrix4d relPose;
};

// global map for storing ICP results:
std::unordered_map<std::string, ICPResult> icpResults;

nlohmann::json MatrixToJson(const Eigen::MatrixXd& matrix) {
    nlohmann::json jsonMatrix = nlohmann::json::array();
    for (int i = 0; i < matrix.rows(); ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(matrix(i, j));
        }
        jsonMatrix.push_back(row);
    }
    return jsonMatrix;
}




// Function to read ICP results from JSON file
std::unordered_map<std::string, ICPResult> ReadICPFromFile(const std::string& filename) {
    std::unordered_map<std::string, ICPResult> icpMap;

    // Open and parse the JSON file
    std::ifstream inFile(filename);
    if (!inFile) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    nlohmann::json jsonData;
    try {
        inFile >> jsonData;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse JSON file: " + std::string(e.what()));
    }
    inFile.close();

    // Convert JSON data into the map
    for (const auto& [key, value] : jsonData.items()) {
        ICPResult result;
        result.sourceName = value.at("sourceName").get<std::string>();
        result.targetName = value.at("targetName").get<std::string>();
        Eigen::Matrix4d T = JsonToMatrix(value.at("T"));
        result.result.T_target_source = T;
        result.result.num_inliers = value.at("num_inliers").get<size_t>();
        result.result.converged = value.at("converged").get<bool>();
        result.result.error = value.at("error").get<double>();
        result.result.iterations = value.at("iterations").get<int>();
        Eigen::Matrix<double, 6, 6> H = JsonToMatrix(value.at("hessian"));
        result.result.H = H;
        Eigen::Matrix<double, 6, 1> b = JsonToMatrix(value.at("b"));
        result.result.b = b;
        Eigen::Matrix4d relPose = JsonToMatrix(value.at("relPose"));
        result.relPose = relPose;
    

        icpMap[key] = result;
    }

    return icpMap;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> updateLinks(const std::vector<ScanPosition>& scanPositions) {
    std::unordered_map<std::string, ICPResult> icpr;
    try {
        icpr = ReadICPFromFile("icp_results.json");
    } catch (const std::exception& e) {
        std::cerr << "No file containing ICP results: " << e.what() << "\n";
        return {}; // Return an empty vector if the file doesn't exist
    }

    // Extract `.first` values into a vector of strings
    std::vector<std::string> links;
    for (const auto& res : icpr) {
        links.push_back(res.first);
    }

    // Split strings by "<->" and create a vector of pairs
    std::vector<std::pair<std::string, std::string>> scanLinks; // Vector of scan ID pairs
    for (const auto& link : links) {
        size_t delimiterPos = link.find("<->");
        if (delimiterPos != std::string::npos) {
            std::string scan1 = link.substr(0, delimiterPos);                // Part before "<->"
            std::string scan2 = link.substr(delimiterPos + 3);              // Part after "<->"
            scanLinks.emplace_back(scan1, scan2); // Save as a pair
        } else {
            std::cerr << "Invalid link format: " << link << "\n";
        }
    }

    // Look up scan positions and store pairs of XYZ coordinates
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> lineSegments; // Vector of 3D line segments
    for (const auto& [scan1, scan2] : scanLinks) {
        Eigen::Vector3f pose1, pose2;
        bool found1 = false, found2 = false;

        // Search for the positions corresponding to the scan IDs
        for (const auto& scan : scanPositions) {
            if (scan.key == scan1) {
                pose1 = scan.positionLocal;
                found1 = true;
            }
            if (scan.key == scan2) {
                pose2 = scan.positionLocal;
                found2 = true;
            }
            if (found1 && found2) break; // Stop searching once both are found
        }

        if (found1 && found2) {
            // Save the line segment as a pair of poses
            lineSegments.emplace_back(pose1, pose2);
        } else {
            std::cerr << "Missing scan positions for: " << scan1 << " or " << scan2 << "\n";
        }
    }

    // Return the vector of line segments
    return lineSegments;
}



void AppendICPResult(const std::string& icpKey, const ICPResult& icpResult, const std::string& filename) {
    // JSON object to store all ICP results
    nlohmann::json jsonData;

    // Try to read the existing file
    std::ifstream inFile(filename);
    if (inFile) {
        try {
            inFile >> jsonData; // Load existing JSON data
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Warning: Failed to parse JSON file, starting with empty data. Error: " << e.what() << "\n";
        }
    }
    inFile.close();

    // Check for existing results with the same source-target or switched target-source pair
    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        const auto& existingResult = it.value();
        std::string existingSource = existingResult["sourceName"];
        std::string existingTarget = existingResult["targetName"];

        // Check if the source-target pair matches or is switched
        if ((existingSource == icpResult.sourceName && existingTarget == icpResult.targetName) ||
            (existingSource == icpResult.targetName && existingTarget == icpResult.sourceName)) {
            std::cout << "Found existing result for source-target pair: "
                      << existingSource << " - " << existingTarget
                      << ". Overwriting with new result.\n";

            // Remove the existing entry
            jsonData.erase(it);
            break; // Exit the loop after finding the match
        }
    }

    // Convert the new ICPResult to JSON
    nlohmann::json icpResultJson;
    icpResultJson["sourceName"] = icpResult.sourceName;
    icpResultJson["targetName"] = icpResult.targetName;
    icpResultJson["T"] = MatrixToJson(icpResult.result.T_target_source.matrix()); // Transformation matrix
    icpResultJson["num_inliers"] = icpResult.result.num_inliers;    // Number of inliers
    icpResultJson["converged"] = icpResult.result.converged;        // Convergence status
    icpResultJson["error"] = icpResult.result.error;               // Error value
    icpResultJson["iterations"] = icpResult.result.iterations;     // Iteration count
    icpResultJson["hessian"] = MatrixToJson(icpResult.result.H.matrix());                 // Hessian matrix
    icpResultJson["b"] = MatrixToJson(icpResult.result.b);
    icpResultJson["relPose"] = MatrixToJson(icpResult.relPose.matrix());

    // Add or update the ICP result in the JSON object
    jsonData[icpKey] = icpResultJson;

    // Write updated JSON data back to the file
    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    outFile << jsonData.dump(4); // Pretty print with 4 spaces
}

void DeleteResultsByRenderedStatus(const std::string& filename) {
    // JSON object to store all ICP results
    nlohmann::json jsonData;

    // Try to read the existing file
    std::ifstream inFile(filename);
    if (inFile) {
        try {
            inFile >> jsonData; // Load existing JSON data
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Warning: Failed to parse JSON file. Error: " << e.what() << "\n";
            return; // Exit the function as we cannot proceed with invalid data
        }
    }
    inFile.close();

    // Iterate over the JSON data and remove results corresponding to rendered scans
    for (auto it = jsonData.begin(); it != jsonData.end();) {
        const auto& result = it.value();
        std::string sourceName = result["sourceName"];
        std::string targetName = result["targetName"];
        std::string exsource = ExtractFilepathAndSuffix(sourceName).second;
        std::string extarget = ExtractFilepathAndSuffix(targetName).second;

        // Check if either source or target scan is rendered
        bool sourceRendered = loadedScans.count(sourceName) && loadedScans.at(sourceName).isRendered;
        bool targetRendered = loadedScans.count(targetName) && loadedScans.at(targetName).isRendered;

        if (sourceRendered && targetRendered) {
            std::cout << "Deleting ICP result for source: " << exsource 
                      << " and target: " << extarget << " because it is rendered.\n";

            // Erase the entry and move to the next
            it = jsonData.erase(it);
        } else {
            ++it; // Move to the next entry
        }
    }

    // Write updated JSON data back to the file
    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    outFile << jsonData.dump(4); // Pretty print with 4 spaces
    std::cout << "Updated JSON file saved successfully.\n";
    lines = updateLinks(scanPositions);
    pangolin::FinishFrame();
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
}




std::pair<std::string, ICPResult> currICP;

void ICP(const std::vector<FileEntry>& fileEntries, const int downsample) {

    ProcessSelectedScans(fileEntries);

    // Ensure exactly 2 scans are rendered
    int renderedCount = 0;
    std::vector<std::string> renderedScans; // To store the keys of the rendered scans
    for (const auto& [key, scanData] : loadedScans) {
        if (scanData.isRendered) {
            renderedScans.push_back(key);
            if (++renderedCount > 2) {
                std::cerr << "Error: Selection of only 2 scans allowed for ICP." << std::endl;
            }
        }
    }

    if (renderedCount != 2) {
        std::cerr << "Error: Exactly 2 scans must be rendered for ICP." << std::endl;
    }
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
    

    // Randomly assign source and target initially
    std::srand(std::time(nullptr)); // Seed random number generator
    int sourceIndex = std::rand() % 2;
    int targetIndex = 1 - sourceIndex;
    std::string extractedSource = ExtractFilepathAndSuffix(renderedScans[sourceIndex]).second;
    std::string extractedTarget = ExtractFilepathAndSuffix(renderedScans[targetIndex]).second;

    std::cout << "Source cloud (transform): " << renderedScans[sourceIndex] << "\n";
    std::cout << "Target cloud (fixed): " << renderedScans[targetIndex] << "\n";

    if (sourceIndex < 0 || sourceIndex >= renderedScans.size() ||
        targetIndex < 0 || targetIndex >= renderedScans.size()) {
        std::cerr << "Error: Invalid source/target index.\n";
    }

    // Access the selected scans
    auto& sourceCloud = loadedScans[renderedScans[sourceIndex]];
    auto& targetCloud = loadedScans[renderedScans[targetIndex]];

    std::vector<Eigen::Vector3f> source_points = sourceCloud.fullCloud->getVertices();
    std::vector<Eigen::Vector3f> target_points = targetCloud.fullCloud->getVertices();

    if (source_points.empty() || target_points.empty()) {
        std::cerr << "Error: No points in source or target cloud.\n";
    }

    // small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
    small_gicp::RegistrationSetting setting;
    setting.type = small_gicp::RegistrationSetting::GICP;
    unsigned int nThreads = std::thread::hardware_concurrency();
    // std::cout << "Number of threads available: " << nThreads << std::endl;
    setting.num_threads = nThreads;                    // Number of threads to be used
    float down_float = downsample / 100.0f;
    setting.downsampling_resolution = down_float;     // Downsampling resolution
    setting.max_correspondence_distance = 0.4;  // Maximum correspondence distance between points (e.g., trimming threshold)
    setting.max_iterations = 10;
    setting.verbose = true;

    // std::cout << "Source points size: " << source_points.size() << "\n";
    // std::cout << "Target points size: " << target_points.size() << "\n";

    Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
    // auto resululu = registration.align();
    small_gicp::RegistrationResult result = small_gicp::align(target_points, source_points, init_T_target_source, setting);

    Eigen::Isometry3d T = result.T_target_source;  // Estimated transformation
    size_t num_inliers = result.num_inliers;       // Number of inlier source points
    Eigen::Matrix<double, 6, 6> H = result.H;      // Final Hessian matrix (6x6)
    bool conv = result.converged;
    double err = result.error;
    Eigen::Matrix<double, 6, 1> b = result.b;

    // std::cout << "ICP complete. Transformation matrix:\n" << T.matrix() << "\n";
    // std::cout << "Number of inliers: " << num_inliers << "\n";
    // std::cout << "Converged: " << conv << "\n";
    // std::cout << "Error: " << err << "\n";
    // std::cout << "b: " << b << "\n";
    // std::cout << "Iterations: " << result.iterations << "\n";

    std::vector<Eigen::Vector3f> subsampled_points = sourceCloud.subsampledCloud->getVertices();
    for (auto& point : subsampled_points) {
        Eigen::Vector4f homogenousPoint(point.x(), point.y(), point.z(), 1.0f); // Convert to homogeneous coordinates
        Eigen::Vector4f transformedPoint = T.matrix().cast<float>() * homogenousPoint; // Apply the transformation
        point = transformedPoint.head<3>(); // Convert back to 3D
    }

    for (auto& point : source_points) {
        Eigen::Vector4f homogenousPoint(point.x(), point.y(), point.z(), 1.0f); // Convert to homogeneous coordinates
        Eigen::Vector4f transformedPoint = T.matrix().cast<float>() * homogenousPoint; // Apply the transformation
        point = transformedPoint.head<3>(); // Convert back to 3D
    }


    // Update the source cloud vertices with transformed points
    sourceCloud.subsampledCloud->setVertices(subsampled_points);
    sourceCloud.fullCloud->setVertices(source_points);

    Eigen::Matrix4d relPose;

    for (auto& scanPosit : scanPositions) {
        if (scanPosit.name == extractedSource) {
            Eigen::Matrix4d updatedLocalPoseSource = T.matrix() * scanPosit.LocalPose;
            Eigen::Matrix4d updatedGlobalPoseSource = globalShift * updatedLocalPoseSource;

            scanPosit.LocalPose = updatedLocalPoseSource;
            scanPosit.GlobalPose = updatedGlobalPoseSource;
            scanPosit.positionLocal = updatedLocalPoseSource.block<3, 1>(0, 3).cast<float>();

            // std::cout << "Updated scan position for source (" << scanPosit.name << "):\n";
            // std::cout << "New global pose:\n" << scanPosit.GlobalPose << "\n";
            // std::cout << "New local position: " << scanPosit.positionLocal.transpose() << "\n";

            for (auto& scanPositT : scanPositions) {
                if (scanPositT.name == extractedTarget) {
                    relPose = scanPositT.LocalPose.inverse() * scanPosit.LocalPose; // relPose in respect to the targets local coordinate frame!!
                }
            }

        }
    }

    // Saving results to map and updating json file:
    std::string icpKey = renderedScans[sourceIndex] + "<->" + renderedScans[targetIndex];
    ICPResult icpR;
    icpR = {renderedScans[sourceIndex],renderedScans[targetIndex],result,relPose};
    icpResults[icpKey] = icpR;
    currICP = {icpKey, icpR};

    // Redraw the scene to show the transformed source cloud
    lines = updateLinks(scanPositions);
    pangolin::FinishFrame();
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
}

void fixPoses(const std::vector<FileEntry>& fileEntries) {

    ProcessSelectedScans(fileEntries);

    // Ensure exactly 2 scans are rendered
    int renderedCount = 0;
    std::vector<std::string> renderedScans; // To store the keys of the rendered scans
    for (const auto& [key, scanData] : loadedScans) {
        if (scanData.isRendered) {
            renderedScans.push_back(key);
            if (++renderedCount > 2) {
                std::cerr << "Error: Selection of only 2 scans allowed for fixing relative poses." << std::endl;
            }
        }
    }

    if (renderedCount != 2) {
        std::cerr << "Error: Exactly 2 scans must be rendered for fixing relative poses." << std::endl;
    }
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
    

    // Randomly assign source and target initially
    std::srand(std::time(nullptr)); // Seed random number generator
    int sourceIndex = std::rand() % 2;
    int targetIndex = 1 - sourceIndex;
    std::string extractedSource = ExtractFilepathAndSuffix(renderedScans[sourceIndex]).second;
    std::string extractedTarget = ExtractFilepathAndSuffix(renderedScans[targetIndex]).second;

    std::cout << "Relative poses fixed for:\n" << extractedSource << " and " << extractedTarget << "\n";


    Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> H_manual = Eigen::Matrix<double, 6, 6>::Identity() * 1e9;

    small_gicp::RegistrationResult manualResult;    
    manualResult.T_target_source = T_target_source;
    manualResult.H = H_manual;
    manualResult.num_inliers = 0;  // Optional, you could set this to a default
    manualResult.converged = true;  // Mark as "converged" since this is manual
    manualResult.error = 0.0;  // USED TO DETECT FIXED POSES
    manualResult.iterations = 0;  // No iterations for manual input
    manualResult.b.setZero();

    Eigen::Matrix4d relPose;
    for (auto& scanPosit : scanPositions) {
        if (scanPosit.name == extractedSource) {
            for (auto& scanPositT : scanPositions) {
                if (scanPositT.name == extractedTarget) {
                    relPose = scanPositT.LocalPose.inverse() * scanPosit.LocalPose; // relPose in respect to the targets local coordinate frame!!
                }
            }
        }
    }

    // Saving results to map and updating json file:
    std::string icpKey = renderedScans[sourceIndex] + "<->" + renderedScans[targetIndex];
    ICPResult icpR;
    icpR = {renderedScans[sourceIndex],renderedScans[targetIndex],manualResult,relPose};
    icpResults[icpKey] = icpR;
    currICP = {icpKey, icpR};

    // Redraw the scene to show the transformed source cloud
    pangolin::FinishFrame();
    lines = updateLinks(scanPositions);
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
}


void saveICPtoFile(std::pair<std::string, ICPResult> currICP,const std::string& newFile) {
    // Append to JSON file
    try {
        AppendICPResult(currICP.first, currICP.second, newFile);
        std::cout << "Saved ICP result to JSON file under key: " << currICP.first << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error saving ICP result to file: " << e.what() << "\n";
    }
}








void ConfigureOptimizer(g2o::SparseOptimizer& optimizer) {
    auto linearSolver = std::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(true);
}

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        std::hash<T1> hash1;
        std::hash<T2> hash2;
        return hash1(pair.first) ^ (hash2(pair.second) << 1);
    }
};

Eigen::Matrix4d ComputeMeanPose(const std::vector<Eigen::Matrix4d>& poses) {
    if (poses.empty()) {
        throw std::runtime_error("No poses to average!");
    }

    // Initialize accumulators
    Eigen::Vector3d meanTranslation = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotationSum = Eigen::Matrix3d::Zero();

    // Accumulate translation and rotation
    for (const auto& pose : poses) {
        Eigen::Vector3d translation = pose.block<3, 1>(0, 3); // Extract translation
        Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);    // Extract rotation

        meanTranslation += translation;  // Sum translations
        rotationSum += rotation;         // Sum rotations
    }

    // Compute mean translation
    meanTranslation /= poses.size();

    // Compute mean rotation: Re-orthogonalize the summed rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotationSum, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d meanRotation = svd.matrixU() * svd.matrixV().transpose();

    // Construct the mean transformation matrix
    Eigen::Matrix4d meanPose = Eigen::Matrix4d::Identity();
    meanPose.block<3, 3>(0, 0) = meanRotation; // Set mean rotation
    meanPose.block<3, 1>(0, 3) = meanTranslation; // Set mean translation

    return meanPose;
}


Eigen::Matrix<double, 6, 6> medianHessian(const std::vector<Eigen::Matrix<double, 6, 6>>& hessians) {
    // Check if the input is not empty
    if (hessians.empty()) {
        throw std::invalid_argument("The hessians vector is empty!");
    }

    // Check if all matrices have the correct size (6x6)
    for (const auto& hessian : hessians) {
        if (hessian.rows() != 6 || hessian.cols() != 6) {
            throw std::invalid_argument("All matrices in the hessians vector must be 6x6!");
        }
    }

    // Initialize the median matrix
    Eigen::Matrix<double, 6, 6> medianHessian;

    // Iterate over each element in the 6x6 matrix
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            // Extract the (i, j) elements into a vector
            std::vector<double> values;
            for (const auto& hessian : hessians) {
                values.push_back(hessian(i, j));
            }

            // Compute the median of the extracted values
            std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
            medianHessian(i, j) = values[values.size() / 2];
        }
    }

    return medianHessian;
}


void saveCovarianceToJson(const g2o::SparseBlockMatrix<Eigen::MatrixXd>& spinv,
                          const std::map<std::string, int>& nodeMap,
                          const std::string& filename) {
    nlohmann::json jsonOutput;

    // Create vector to store scan keys in order
    std::vector<std::string> scanKeys;

    // Fill scan keys from nodeMap
    for (const auto& pair : nodeMap) {
        scanKeys.push_back(pair.first);  // Assuming nodeMap is already ordered as needed
    }

    // Convert SparseBlockMatrix to Dense Eigen Matrix
    int rows = spinv.rows();
    int cols = spinv.cols();
    Eigen::MatrixXd denseCovarianceMatrix(rows, cols);
    
    // Fill the dense matrix with values from the sparse matrix
    denseCovarianceMatrix.setZero();  // Initialize with zeros
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const auto block = spinv.block(i, j);
            if (block) {
                denseCovarianceMatrix.block(i, j, block->rows(), block->cols()) = *block;
            }
        }
    }

    // Convert the dense covariance matrix to a nested vector for JSON output
    std::vector<std::vector<double>> denseMatrixVec;
    for (int i = 0; i < denseCovarianceMatrix.rows(); ++i) {
        denseMatrixVec.emplace_back(denseCovarianceMatrix.row(i).data(),
                                    denseCovarianceMatrix.row(i).data() + denseCovarianceMatrix.cols());
    }

    // Store the dense covariance matrix and scan keys in the JSON
    jsonOutput["covariance_matrix"] = denseMatrixVec;
    jsonOutput["scan_keys"] = scanKeys;

    // Save to file
    std::ofstream file(filename);
    if (file.is_open()) {
        file << jsonOutput.dump(4);  // Pretty-print JSON
        file.close();
        std::cout << "Covariance matrix and scan keys saved to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
    }
}



void OptimizePoseGraph(const std::unordered_map<std::string, ICPResult>& icpR, const std::string& outputFilename, std::pair<double,double> stoch_a_priori) {
    // Build a graph of poses
    g2o::SparseOptimizer optimizer;
    ConfigureOptimizer(optimizer); // Setup solver, linear system, etc.

    // Map scan names to node IDs
    std::map<std::string, int> nodeMap; // Maps scan names to graph node IDs

    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();

    nlohmann::json optimizationResults = nlohmann::json::object(); // JSON object to store results

    // Add nodes (scans) to the graph
    for (size_t i = 0; i < scanPositions.size(); ++i) {
        const auto& scan = scanPositions[i];
        g2o::VertexSE3* vertex = new g2o::VertexSE3();
        vertex->setId(i);
        
        
        std::vector<Eigen::Matrix4d> poses;
        // for (const auto& res : icpR) {
        //     if (res.second.sourceName == scan.key) {

        //         poses.push_back(res.second.result.T_target_source.matrix() * scan.LocalPose);
        //     }
        // }
        // Check if poses are available, otherwise use the default LocalPose
        if (!poses.empty()) {
            vertex->setEstimate(Eigen::Isometry3d(ComputeMeanPose(poses)));
        } else {
            vertex->setEstimate(Eigen::Isometry3d(scan.LocalPose));
        }

        // vertex->setEstimate(Eigen::Isometry3d(scan.LocalPose)); 
        if (i == 0) vertex->setFixed(true);  // Fix the first node as the reference
        optimizer.SparseOptimizer::addVertex(vertex);
        nodeMap[scan.key] = i; // Populate node map

        // Save initial vertex data
        optimizationResults["vertices"][scan.key] = {
            {"initialPose", MatrixToJson(scan.LocalPose)},
            {"optimizationTransform", {}},
            {"optimizedPose", {}}, // Will be updated later
            {"globalPose", {}},
            {"covariance", {}}
        };
    }

    // //compute median hessian:
    // Eigen::Matrix<double, 6, 6> medianHess;
    // std::vector<Eigen::Matrix<double, 6, 6>> hessians;
    // for (const auto& result : icpR) {
    //     hessians.push_back(result.second.result.H);
    // }

    // Compute median weight:
    std::vector<double> weights;
    for (const auto& result : icpR) {
        double weight = result.second.result.error/static_cast<double>(result.second.result.num_inliers);
        weights.push_back(weight);
        std::cout << weight << std::endl;
    }
    size_t n = weights.size();
    std::nth_element(weights.begin(), weights.begin() + n / 2, weights.end());
    double med = weights[n / 2];
    
    if (n % 2 == 0) {
        // For an even number of elements, find the largest element in the lower half.
        std::nth_element(weights.begin(), weights.begin() + n / 2 - 1, weights.end());
        med = (med + weights[n / 2 - 1]) / 2.0;
    }
    std::cout << "Median error: " << med << std::endl;
    

    // Compute the median Hessian
    try {
        // medianHess = medianHessian(hessians);
        // std::cout << "Median Hessian:\n" << medianHess << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error computing median Hessian: " << e.what() << std::endl;
    }


    for (const auto& result : icpR) {
        const std::string& sourceName = result.second.sourceName;
        const std::string& targetName = result.second.targetName;

        int sourceId = nodeMap[sourceName];
        int targetId = nodeMap[targetName];

        // Create a new edge
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        
        edge->setVertex(0, optimizer.vertex(sourceId));
        edge->setVertex(1, optimizer.vertex(targetId));
        edge->setMeasurement(Eigen::Isometry3d(result.second.relPose.inverse()));
        // edge->

        // Set the edge information matrix
        // Eigen::Matrix<double, 6, 6> hessian = result.second.result.H;
        // Eigen::Matrix<double, 6, 1> b = result.second.result.b;
        // double sigma_trans = std::max(1e-6, result.second.result.error / 
        //             (static_cast<double>(result.second.result.num_inliers) + 1e-6));
        double sigma_trans = 0.008;
        double d_avg = result.second.relPose.block<3,1>(0,3).norm() + 1e-6;  // Prevent division by zero
        // double sigma_trans = 1.85*std::sqrt(d_avg);
        double sigma_squared_trans = sigma_trans * sigma_trans;

        // double d_avg = result.second.relPose.block<3,1>(0,3).norm() + 1e-6;  // Prevent division by zero
        double sigma_gamma = 2*std::atan(sigma_trans/(2*d_avg));
        // double sigma_squared_rot = (sigma_trans / d_avg) * (sigma_trans / d_avg);
        double sigma_squared_rot = sigma_gamma * sigma_gamma;

        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigSolver(result.second.result.H);
        // Eigen::Matrix<double, 6, 6> D = eigSolver.eigenvalues().asDiagonal();
        // Eigen::Matrix<double, 6, 6> V = eigSolver.eigenvectors();

        // // Prevent near-zero eigenvalues from making the matrix singular
        // for (int i = 0; i < 6; ++i) {
        //     if (D(i, i) < 1e-6) D(i, i) = 1e-6;  // Regularization
        // }

        // // Normalize rotational and translational components separately
        // double mean_trans = (D.block<3, 3>(0, 0).trace()) / 3.0;
        // double mean_rot = (D.block<3, 3>(3, 3).trace()) / 3.0;

        // // Avoid division by zero
        // if (mean_rot < 1e-6) mean_rot = 1e-6;
        // if (mean_trans < 1e-6) mean_trans = 1e-6;

        // // Compute the scaling ratio
        // double scale_ratio = mean_trans / mean_rot;

        // // Scale the eigenvalues to balance translation/rotation terms
        // D.block<3, 3>(3, 3) *= scale_ratio;  // Scale rotational confidence to match translation

        // // Reconstruct the Hessian
        // Eigen::Matrix<double, 6, 6> H_scaled = V * D * V.transpose();

        // Eigen::Matrix3d H_t = result.second.result.H.block<3,3>(0,0);
        // Eigen::Matrix3d H_r = result.second.result.H.block<3,3>(3,3);
        // Eigen::Matrix3d H_tr = result.second.result.H.block<3,3>(0,3);

        // Eigen::Matrix<double,6,6> H = result.second.result.H;
        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eig(H);
        // Eigen::Matrix<double,6,6> D = eig.eigenvalues().asDiagonal();
        // Eigen::Matrix<double,6,6> V = eig.eigenvectors();

        // // Ensure positive eigenvalues
        // for (int i = 0; i < 6; ++i) {
        //     if (D(i,i) < 1e-6) D(i,i) = 1e-6;
        // }

        // H = V * D * V.transpose();  // Regularized Hessian

        


        Eigen::Matrix<double, 6, 6> info_ = result.second.result.H;
        // Eigen::Quaterniond eigenQuat(info_.block<3, 3>(0, 0));
        
        double e = 1e-6;
        double num_inliers = static_cast<double>(result.second.result.num_inliers) + e;
        double err_per_inlier = result.second.result.error / num_inliers;
        // if (!(result.second.result.error == 0.0)) {
        //     info_ /= num_inliers;
        // }
        // // info_ /= num_inliers;
        // // double scale_factor = 1.0 / ((num_inliers + e) * (err_per_inlier +e));
        // double scale_factor;
        // if (result.second.result.error == 0.0) {
        //     scale_factor = 1e9;
        // } else {
        //     scale_factor = lambda/(err_per_inlier*err_per_inlier);
        // }
        // // double scale_factor = 1/(err_per_inlier*err_per_inlier);

        // info_ *= scale_factor;

        // Extract the rotational part
        // Eigen::Matrix3d H_rot = info_.block<3,3>(3,3); // Extract 3x3 rotational Hessian

        // Convert rotation Hessian to quaternion form
        // Eigen::Quaterniond quat(H_rot);

        // Ensure quaternion normalization
        // quat.normalize();

        // Convert translation part (remains unchanged)
        // Eigen::Matrix3d H_trans = info_.block<3,3>(0,0) * scale_factor;

        // Extract translation and rotation Hessians
        Eigen::Matrix3d H_trans = info_.block<3,3>(0,0); // Translation Hessian
        Eigen::Matrix3d H_rot = info_.block<3,3>(3,3);   // Rotation Hessian 

        // // Compute right Jacobian of SO(3) (approximation)
        // Eigen::Matrix3d Jr_inv = Eigen::Matrix3d::Identity() - 0.5 * H_rot;

        // // Map rotation Hessian to quaternion tangent space
        // Eigen::Matrix3d H_rot_quat = Jr_inv * H_rot * Jr_inv.transpose();

        // // Assign modified blocks back to information matrix
        // info_.block<3,3>(0,0) = H_trans;        // Translation part remains unchanged
        // info_.block<3,3>(3,3) = H_rot_quat;     // Corrected rotation information


        // --- Initialize Covariance Matrix ---
        Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6,6>::Identity();
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6,6>::Identity();
        if (custom_stoch) {
            if (result.second.result.error == 0.0) {
                info = cov.inverse() * 1e9;  // Fixed pose → high weight
            } else {
                cov.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * stoch_a_priori.first*stoch_a_priori.first;  // Translation
                double sigma_squared_rot_custom = (stoch_a_priori.first / d_avg) * (stoch_a_priori.first / d_avg);
                // cov.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * sigma_squared_rot_custom;   // Rotation
                // std::cout << sigma_squared_rot_custom << std::endl;
                cov.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * stoch_a_priori.second*stoch_a_priori.second;   // Rotation
                info = cov.inverse();
                // H_rot = info.block<3,3>(3,3);
                // Eigen::Quaterniond quat(H_rot);
                // quat.normalize();
                // H_trans = info.block<3,3>(0,0);
                // info.block<3,3>(0,0) = H_trans;
                // info.block<3,3>(3,3) = H_rot;
            }
        } else if (use_hessian) {
            if (result.second.result.error == 0.0) {
                info = info_*1e6;  // Fixed pose → high weight
            } else {
                // info = H_scaled.inverse();
                info = info_;
                // info.block<3,3>(0,0) = H_trans;
                // info.block<3,3>(3,3) = H_rot;
                // info = H.inverse();
                // info.block<3,3>(0,0) *= 1.0;   // Translation confidence (tune as needed)
                // info.block<3,3>(3,3) *= 100.0; // Rotation confidence (higher weight)

                // info = result.second.result.H;
            }
        } else if (use_hessian_diag) {
            if (result.second.result.error == 0.0) {
                info = info_.diagonal().asDiagonal() * 1e9;  // Fixed pose → high weight
            } else {
                // info = info_.diagonal().asDiagonal();
                info.block<3,3>(0,0) = H_trans;
                info.block<3,3>(3,3) = H_rot;
            }
        } else {
            if (result.second.result.error == 0.0) {
                info = cov.inverse() * 1e9;  // Fixed pose → high weight
            } else {
                cov.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * sigma_squared_trans;  // Translation
                cov.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * sigma_squared_rot;   // Rotation
                std::cout << sigma_squared_rot << std::endl;
                info = cov.inverse();
            }
        }
        
        std::cout << "Information: \n" << info << std::endl;
        edge->setInformation(info);

        // hessian = hessian / hessian.maxCoeff(); // Normalize to the range [0, 1]
        // edge->setInformation(hessian.inverse()); 

        // Add the edge to the optimizer
        optimizer.SparseOptimizer::addEdge(edge);

        // // Debug output
        // std::cout << "Adding edge: Source " << ExtractFilepathAndSuffix(sourceName).second << " -> Target " << ExtractFilepathAndSuffix(targetName).second << std::endl;
        // std::cout << "Measurement: \n" << result.second.relPose.matrix().inverse() << std::endl;
        // std::cout << "Information: \n" << medianHessian << std::endl;

        // Save edge data
        optimizationResults["edges"].push_back({
            {"source", sourceName},
            {"target", targetName},
            {"relativePose", MatrixToJson(result.second.relPose.inverse().matrix())},
            {"Hessian", MatrixToJson(info)},
            {"residualNorm", {}}, // Will be updated later
        });
    }


    // Perform optimization
    optimizer.setVerbose(true);
    optimizer.computeInitialGuess();
    std::cout << "Initial chi2: " << optimizer.chi2() << std::endl;

    optimizer.initializeOptimization();
    optimizer.optimize(1000);
    // Compute Degrees of Freedom (DoF)
    int numEdges = icpR.size();  // Number of relative pose constraints
    int numPoses = scanPositions.size(); // Number of poses
    int numParams = numPoses * 6; // Each pose has 6 DoF
    int numFixedParams = 6; // First pose is fixed (6 DoF removed)

    int dof = (numEdges * 6) - (numParams - numFixedParams);
    // std::cout << "Degrees of Freedom (DoF): " << dof << std::endl;

    // std::cout << "Final chi2: " << optimizer.chi2() << std::endl;

    for (const auto* edge : optimizer.edges()) {
        const auto* e = dynamic_cast<const g2o::EdgeSE3*>(edge);
        if (!e) continue;

        // Get the two vertices associated with the edge
        const auto* v1 = dynamic_cast<const g2o::VertexSE3*>(e->vertex(0));
        const auto* v2 = dynamic_cast<const g2o::VertexSE3*>(e->vertex(1));

        // Get their optimized estimates
        Eigen::Isometry3d T1 = v1->estimate();
        Eigen::Isometry3d T2 = v2->estimate();
        std::string source;
        std::string target;
        std::string sourceP;
        std::string targetP;
        for (auto& node : nodeMap) {
            if (node.second == v1->id()) {
                source = ExtractFilepathAndSuffix(node.first).second;
                sourceP = node.first;
            } else if (node.second == v2->id())
            {
                target = ExtractFilepathAndSuffix(node.first).second;
                targetP = node.first;
            }
            
        }

        // Compute the measured transformation from ICP
        Eigen::Isometry3d T_measured = e->measurement();

        // Compute the relative transformation from optimized poses
        Eigen::Isometry3d T_optimized = T1.inverse() * T2;

        // Compute the residual
        Eigen::Isometry3d errorTransform = T_measured.inverse() * T_optimized;
        Eigen::Matrix< double, 6, 1 > residual = g2o::internal::toVectorMQT(errorTransform);

        // Compute the norm of the residual (optional)
        double residualNorm = residual.norm();

        // Debug output
        std::cout << "Edge between vertices " << source << " and " << target << " has residual norm: " << residualNorm << std::endl;

        // Threshold to identify outliers
        if (residualNorm > 0.05) {
            std::cout << "High residual detected for edge: " << source << " -> " << target << std::endl;
        }

        // Update JSON with residuals
        for (auto& edgeJson : optimizationResults["edges"]) {
            if (edgeJson["source"] == sourceP &&
                edgeJson["target"] == targetP) {
                edgeJson["residualNorm"] = residualNorm;
                break;
            }
        }
    }

    std::unordered_map<std::string, Eigen::Matrix4d> trafos;
    // optimizer.computeMarginals();
    // Update scan positions with optimized poses
    nlohmann::json optimizedData = nlohmann::json::object();

    // std::vector<std::pair<int, int>> blockIndices;
    // for (size_t i = 0; i < scanPositions.size(); ++i) {
    //     if (i % 10 == 0) {
    //         blockIndices.emplace_back(i, i); // Diagonal blocks for each vertex
    //     }
        
    // }


    // g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv; // Inverse of Hessian
    // optimizer.computeMarginals(spinv, blockIndices); 
    for (size_t i = 0; i < scanPositions.size(); ++i) {
        g2o::VertexSE3* vertex = static_cast<g2o::VertexSE3*>(optimizer.vertex(nodeMap[scanPositions[i].key]));
        // prevScanPoses.emplace_back(scanPositions[i].LocalPose);
        trafos[scanPositions[i].key] = vertex->estimate().matrix() * scanPositions[i].LocalPose.inverse();
        Eigen::Matrix4d initialPose = scanPositions[i].LocalPose;
        scanPositions[i].LocalPose = vertex->estimate().matrix();
        scanPositions[i].GlobalPose = globalShift * scanPositions[i].LocalPose;
        scanPositions[i].positionLocal = scanPositions[i].LocalPose.block<3, 1>(0, 3).cast<float>();

        // Add to JSON data
        nlohmann::json scanJson;
        scanJson["ID"] = scanPositions[i].key;
        scanJson["GlobalPose"] = MatrixToJson(scanPositions[i].GlobalPose);
        optimizedData[scanPositions[i].key] = scanJson;

        // Update JSON data:
        optimizationResults["vertices"][scanPositions[i].key] = {
            {"initialPose", MatrixToJson(initialPose)},
            {"optimizedPose", MatrixToJson(scanPositions[i].LocalPose)}, // Will be updated later
            {"globalPose", MatrixToJson(scanPositions[i].GlobalPose)}
        };

        // g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
        // optimizer.computeMarginals();
        if (!(i == scanPositions.size())) {
            g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
            // std::vector<std::pair<int, int>> blockIndices = {{i, i}};
            // optimizer.computeMarginals(spinv, blockIndices);
            // Eigen::Matrix<double, 6, 6> hessian = optimizer.vertex(nodeMap[scanPositions[i].key])->hessian;
            // if (optimizer.vertex(nodeMap[scanPositions[i].key])->fixed()) {continue;}
            // std::vector<std::pair<int, int>> blockIndices = {{i, i}};
            // std::cout << nodeMap[scanPositions[i].key] << " or " << scanPositions[i].key << std::endl;
            // std::cout << "Hessian index: " << hessian_index << std::endl;
            // g2o::SparseOptimizer::computeMarginals
            // optimizer.computeMarginals(spinv, optimizer.vertex(nodeMap[scanPositions[i].key]));
            // optimizer.computeMarginals(spinv, blockIndices);
            Eigen::MatrixXd hessianBlock;
            if (optimizer.computeMarginals(spinv, vertex)) {
                auto block = spinv.block(vertex->hessianIndex(), vertex->hessianIndex());
                if (block) {
                    hessianBlock = *block;
                    // std::cout << "Hessian block for vertex:\n" << hessianBlock << std::endl;
                }
            }



        

            // if (!spinv.block(i, i)) {
            //     std::cerr << "Covariance block for vertex " << scanPositions[i].name << " is null!" << std::endl;
            //     continue;
            // }

            if (true) {
                // const Eigen::MatrixXd covariance = *spinv.block(i, i); 
                const Eigen::MatrixXd covariance = hessianBlock; 
                std::cout << "Covariance for vertex " << scanPositions[i].name << ":\n" << covariance << std::endl;
                // std::cout << "Hessian inverse for vertex " << scanPositions[i].name << ":\n" << hessian.inverse() << std::endl;

                optimizationResults["vertices"][scanPositions[i].key]["covariance"] = MatrixToJson(covariance);
            }
            
        }
    }

    if (com_rel_cov) {
        // std::map<std::string, g2o::VertexSE3*> vertex_map;
        std::vector<g2o::VertexSE3*> vertices;
        std::vector<std::string> keys;
        for (size_t i = 0; i < scanPositions.size(); ++i) {
            for (auto& relscan : rel_cov_poses) {
                if (scanPositions[i].key == relscan.first) {
                    // relscan.second = static_cast<g2o::VertexSE3*>(optimizer.vertex(nodeMap[scanPositions[i].key]));
                    // keys.push_back(scanPositions[i].key);
                    vertices.push_back(static_cast<g2o::VertexSE3*>(optimizer.vertex(nodeMap[scanPositions[i].key])));                
                }
            }
        }

        // g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
        // Eigen::MatrixXd covBlock;
        // // const std::vector<std::pair<int, int>>& blockIndices = {{rel_cov_poses[keys[0]]->hessianIndex(), rel_cov_poses[keys[1]]->hessianIndex()}};
        // optimizer.computeMarginals(spinv, vertices);
        // auto block = spinv.block(rel_cov_poses[keys[0]]->hessianIndex(), rel_cov_poses[keys[0]]->hessianIndex());
        // if (block) {
        //     covBlock = *block;
        //     std::cout << "relative covariances for vertices " << keys[0] << " and " << keys[1] << ":\n" << covBlock << std::endl;
        // }
        

    }
    
    std::cout << "Degrees of Freedom (DoF): " << dof << std::endl;
    std::cout << "Final chi2: " << optimizer.chi2() << std::endl;
    double red_chi2 = optimizer.chi2()/dof;
    std::cout << "Reduced chi2: " << red_chi2 << std::endl;

    // Save updated poses to a new JSON file
    std::ofstream outFile(outputFilename);
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + outputFilename);
    }
    outFile << optimizedData.dump(4); // Pretty print with 4 spaces
    outFile.close();

    for (const auto& scan : loadedScans) {
        if (!scan.second.subsampledCloud->getVertices().empty()) {
            std::vector<Eigen::Vector3f> subsampled_points = scan.second.subsampledCloud->getVertices();
            for (auto& point : subsampled_points) {
                Eigen::Vector4f homogenousPoint(point.x(), point.y(), point.z(), 1.0f); // Convert to homogeneous coordinates
                Eigen::Vector4f transformedPoint = trafos[scan.first].cast<float>() * homogenousPoint; // Apply the transformation
                point = transformedPoint.head<3>(); // Convert back to 3D
            }
            std::vector<Eigen::Vector3f> full_points = scan.second.fullCloud->getVertices();
            for (auto& point : full_points) {
                Eigen::Vector4f homogenousPoint(point.x(), point.y(), point.z(), 1.0f); // Convert to homogeneous coordinates
                Eigen::Vector4f transformedPoint = trafos[scan.first].cast<float>() * homogenousPoint; // Apply the transformation
                point = transformedPoint.head<3>(); // Convert back to 3D
            }
            scan.second.subsampledCloud->setVertices(subsampled_points);
            scan.second.fullCloud->setVertices(full_points);

        }
    }    

    // Save results to file
    std::string outFileG2O = "pose_optimization_results.json";
    std::ofstream outFileG2On(outFileG2O);
    if (!outFileG2On) {
        throw std::runtime_error("Failed to open file for writing: " + outFileG2O);
    }
    outFileG2On << optimizationResults.dump(4); // Pretty print with 4 spaces

    std::cout << "Optimization results saved to " << outFileG2O << std::endl;
    outFileG2On.close();

    lines = updateLinks(scanPositions);
    DrawAll();
    DrawScanPositions(scanPositions);
    pangolin::FinishFrame();
}

void computeRelCov(const std::unordered_map<std::string, ICPResult>& icpR, const std::string& outputFilename, std::pair<double,double> stoch_a_priori) {
    // Ensure exactly 2 scans are rendered
    int renderedCount = 0;
    std::vector<std::string> renderedScans; // To store the keys of the rendered scans
    for (const auto& [key, scanData] : loadedScans) {
        if (scanData.isRendered) {
            renderedScans.push_back(key);
            if (++renderedCount > 2) {
                std::cerr << "Error: Selection of only 2 scans allowed for ICP." << std::endl;
            }
        }
    }

    for (auto const rend : renderedScans) {
        rel_cov_poses[rend] = nullptr;
    }


    OptimizePoseGraph(icpR,"optimized_poses.json",stoch_a_priori);
}

// Adapter for std::vector<Eigen::Vector3f> to work with nanoflann
struct PointCloudAdapter {
    const std::vector<Eigen::Vector3f>& points;

    PointCloudAdapter(const std::vector<Eigen::Vector3f>& pts) : points(pts) {}

    // nanoflann requires this: number of points in the dataset
    inline size_t kdtree_get_point_count() const { return points.size(); }

    // nanoflann requires this: access the x, y, z coordinates of a point
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim]; // 0 for x, 1 for y, 2 for z
    }

    // nanoflann requires this: return false if bounding box calculation is not implemented
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// Typedef for KDTree using the adapter
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudAdapter>, PointCloudAdapter, 3>;

void computeNormals(const std::vector<Eigen::Vector3f>& points, 
                    std::vector<Eigen::Vector3f>& normals, 
                    float radius, 
                    float zenithThresholdCos) {
    PointCloudAdapter adapter(points);
    KDTree tree(3, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(50));
    tree.buildIndex();

    normals.resize(points.size());
    #pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i) {
        // Find neighbors
        std::vector<nanoflann::ResultItem<uint32_t, float>> results; // Store index and distance
        nanoflann::SearchParameters params;
        size_t num_neighbors = tree.radiusSearch(&points[i][0], radius * radius, results, params);

        if (num_neighbors < 3) {
            normals[i] = Eigen::Vector3f::Zero(); // Not enough neighbors
            continue;
        }

        // Compute covariance matrix
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();

        for (const auto& result : results) {
            const Eigen::Vector3f& neighbor = points[result.first];
            mean += neighbor;
        }
        mean /= num_neighbors;

        for (const auto& result : results) {
            const Eigen::Vector3f& neighbor = points[result.first];
            Eigen::Vector3f diff = neighbor - mean;
            covariance += diff * diff.transpose();
        }

        covariance /= num_neighbors;

        // Compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Vector3f normal = solver.eigenvectors().col(0); // Smallest eigenvector

        // Ensure the normal points upward
        if (normal.z() < 0) {
            normal = -normal;
        }

        // Check zenith constraint
        Eigen::Vector3f zenith(0, 0, 1); // Upward vertical direction
        float cosTheta = normal.dot(zenith); // Dot product with zenith direction

        if (cosTheta >= zenithThresholdCos) {
            normals[i] = Eigen::Vector3f::Zero(); // Filter out near-vertical normals
        } else {
            normals[i] = normal;
        }
    }
}


// Function to compute azimuth and elevation angles of normals
void computeAzimuthElevation(const std::vector<Eigen::Vector3f>& normals, 
                             std::vector<float>& azimuths, 
                             std::vector<float>& elevations) {
    if (normals.empty()) return; // Check for empty input

    for (const auto& normal : normals) {
        // Check if the normal vector is valid (non-zero)
        if (normal.norm() == 0) continue;  // Skip invalid (zero) normals

        float azimuth = std::atan2(normal.y(), normal.x()) * 180.0f / M_PI;
        float elevation = std::asin(normal.z() / normal.norm()) * 180.0f / M_PI;
        azimuths.push_back(azimuth);
        elevations.push_back(elevation);
    }
}

void computeStatistics(const std::vector<float>& data, 
                       float& mean, 
                       float& median, 
                       float& std_dev, 
                       std::vector<float>& percentiles,
                       const std::vector<float>& percentile_vals) {
    if (data.empty()) {
        mean = 0.0f;
        median = 0.0f;
        std_dev = 0.0f;
        percentiles.clear();
        return;
    }

    // Compute mean
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    mean = sum / data.size();

    // Compute median
    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    size_t mid = sorted_data.size() / 2;
    median = (sorted_data.size() % 2 == 0) 
                ? (sorted_data[mid - 1] + sorted_data[mid]) / 2.0f 
                : sorted_data[mid];

    // Compute standard deviation
    if (data.size() > 1) {
        float variance = 0.0f;
        for (float value : data) {
            variance += (value - mean) * (value - mean);
        }
        std_dev = std::sqrt(variance / (data.size() - 1)); // Sample standard deviation
    } else {
        std_dev = 0.0f;
    }

    // Compute percentiles
    for (float perc : percentile_vals) {
        size_t idx = static_cast<size_t>(perc * sorted_data.size());
        if (idx >= sorted_data.size()) idx = sorted_data.size() - 1;
        percentiles.push_back(sorted_data[idx]);
    }
}


// Function to compute statistical data and save it as JSON
void saveNormalStatistics(const std::vector<Eigen::Vector3f>& normals, const std::string& output_file, const std::string& key) {
    // Compute azimuth and elevation angles
    std::vector<float> azimuths;
    std::vector<float> elevations;
    computeAzimuthElevation(normals, azimuths, elevations);

    // Statistical metrics
    float mean_azimuth, median_azimuth, std_dev_azimuth;
    float mean_elevation, median_elevation, std_dev_elevation;
    std::vector<float> percentiles_azimuth;
    std::vector<float> percentiles_elevation;

    // Percentile values (e.g., 25%, 50%, 75%)
    std::vector<float> percentile_values = {0.25, 0.5, 0.75};

    // Compute statistics for azimuths
    computeStatistics(azimuths, mean_azimuth, median_azimuth, std_dev_azimuth, percentiles_azimuth, percentile_values);

    // Compute statistics for elevations
    computeStatistics(elevations, mean_elevation, median_elevation, std_dev_elevation, percentiles_elevation, percentile_values);

    // Bin histogram information
    size_t num_bins = 20; // Number of bins for the histogram
    std::vector<size_t> histogram_azimuth(num_bins, 0);
    std::vector<size_t> histogram_elevation(num_bins, 0);

    float azimuth_min = *std::min_element(azimuths.begin(), azimuths.end());
    float azimuth_max = *std::max_element(azimuths.begin(), azimuths.end());
    float elevation_min = *std::min_element(elevations.begin(), elevations.end());
    float elevation_max = *std::max_element(elevations.begin(), elevations.end());

    float azimuth_bin_size = (azimuth_max - azimuth_min) / num_bins;
    float elevation_bin_size = (elevation_max - elevation_min) / num_bins;

    for (float azimuth : azimuths) {
        size_t bin = std::min(static_cast<size_t>((azimuth - azimuth_min) / azimuth_bin_size), num_bins - 1);
        histogram_azimuth[bin]++;
    }

    for (float elevation : elevations) {
        size_t bin = std::min(static_cast<size_t>((elevation - elevation_min) / elevation_bin_size), num_bins - 1);
        histogram_elevation[bin]++;
    }

    // Create the new result object
    nlohmann::json new_result;
    new_result["azimuth"] = {
        {"mean", mean_azimuth},
        {"median", median_azimuth},
        {"std_dev", std_dev_azimuth},
        {"percentiles", percentiles_azimuth},
        {"histogram", histogram_azimuth},
        {"bin_size", azimuth_bin_size},
        {"range", {azimuth_min, azimuth_max}}
    };

    new_result["elevation"] = {
        {"mean", mean_elevation},
        {"median", median_elevation},
        {"std_dev", std_dev_elevation},
        {"percentiles", percentiles_elevation},
        {"histogram", histogram_elevation},
        {"bin_size", elevation_bin_size},
        {"range", {elevation_min, elevation_max}}
    };

    // Load existing JSON file if it exists
    nlohmann::json result;
    std::ifstream in(output_file);
    if (in.is_open()) {
        try {
            in >> result; // Load existing JSON
        } catch (...) {
            // If there's an error in reading/parsing, start with an empty JSON object
            result = nlohmann::json::object();
        }
        in.close();
    }

    // Append the new result under the provided key
    result[key] = new_result;

    // Write the updated JSON back to the file
    std::ofstream out(output_file);
    if (out.is_open()) {
        out << result.dump(4); // Pretty print with 4 spaces
        out.close();
    }
}








void applyPoses(const std::string& poseFile) {
    std::unordered_map<std::string, Eigen::Matrix4d> optPoses;
    try {
        optPoses = LoadGlobalPoses(poseFile);

        // Print the loaded poses
        for (const auto& [id, pose] : optPoses) {
            std::cout << "ID: " << id << "\nPose:\n" << pose << "\n\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    std::unordered_map<std::string, ScanData> emptyMap;
    loadedScans.swap(emptyMap);
    const char* e57FileFilterPatterns[] = {"*.e57"};
    const char* inputE57Filenames = tinyfd_openFileDialog(
        "Select E57 files to apply poses to",
        curr_path.c_str(), // Default path (empty for user to choose)
        1, // Number of filter patterns
        e57FileFilterPatterns, // Filter patterns for E57 files
        "E57 files (*.e57)", // Single filter description
        1 // Allow multiple selects
    );

    if (!inputE57Filenames) {
        std::cerr << "No E57 files selected." << std::endl;
        return;
    }

    // Split the returned file paths by the pipe character '|'
    std::vector<std::string> e57Files = splitString(inputE57Filenames, '|');
    if (e57Files.empty()) {
        std::cerr << "No valid E57 files found after split." << std::endl;
        return;
    }

    std::filesystem::path EfilePath(e57Files[0]);
    curr_path = EfilePath.parent_path();
    std::filesystem::path firstInputPath(e57Files[0]);
    std::string outputFilename;
    if (e57Files.size() < 2) {
        outputFilename = firstInputPath.stem().string() + "_optimized.e57";  
    } else {
        outputFilename = firstInputPath.stem().string() + "_merged_optimized.e57";  
    }
    
    std::filesystem::path outputPath = firstInputPath.parent_path() / outputFilename;
    std::cout << "Writing optimized E57 to: " << outputPath << std::endl;

    // Create the E57 Writer for the single output file
    e57::WriterOptions writerOptions;
    e57::Writer e57Writer(outputPath.string().c_str(), writerOptions);
    if (!e57Writer.IsOpen()) {
        std::cerr << "Failed to open E57 writer for " << outputFilename << std::endl;
        return;
    }

    size_t guidCounter = 1;

    // Process each input E57 file
    for (const auto& inputE57Filename : e57Files) {
        // Open the current E57 file for reading
        e57::Reader e57Reader(inputE57Filename.c_str(), e57::ReaderOptions());
        std::cout << "Reading E57 file: " << inputE57Filename << std::endl;

        size_t scanCount = e57Reader.GetData3DCount();
        std::cout << "Number of scans in the file: " << scanCount << std::endl;

        // Process each scan in the current file
        for (size_t scanIndex = 0; scanIndex < scanCount; ++scanIndex) {
            e57::Data3D scanHeader;
            e57Reader.ReadData3D(scanIndex, scanHeader);

            std::string scanKey = inputE57Filename + "_" + scanHeader.name;

            auto poseIt = optPoses.find(scanKey);
            if (poseIt == optPoses.end()) {
                std::cerr << "Error: Pose not found for scanKey: " << scanKey << std::endl;
                continue;
            }
            Eigen::Matrix4d transformedPoseMatrix = poseIt->second;


            // Eigen::Matrix4d transformedPoseMatrix = optPoses[scanKey];

            // Update the scan header with the new pose
            Eigen::Quaterniond eigenQuat(transformedPoseMatrix.block<3, 3>(0, 0));
            e57::Quaternion e57Quat;
            e57Quat.w = eigenQuat.w();
            e57Quat.x = eigenQuat.x();
            e57Quat.y = eigenQuat.y();
            e57Quat.z = eigenQuat.z();
            scanHeader.pose.rotation = e57Quat;
            scanHeader.pose.translation.x = transformedPoseMatrix(0, 3);
            scanHeader.pose.translation.y = transformedPoseMatrix(1, 3);
            scanHeader.pose.translation.z = transformedPoseMatrix(2, 3);

            scanHeader.guid = generateUniqueGUID(guidCounter);
            std::cout << "scan GUID: " << scanHeader.guid << std::endl;
            guidCounter++;

            // Read the point data
            e57::Data3DPointsDouble buffers;
            // InitializeBuffersFromScanHeader(buffers, scanHeader);
            e57::CompressedVectorReader dataReader = e57Reader.SetUpData3DPointsData(scanIndex, scanHeader.pointCount, buffers);
            dataReader.read();

            // Write the transformed scan data to the output file
            int64_t dataIndex = e57Writer.NewData3D(scanHeader);  // Create a new Data3D block in the output file
            e57::CompressedVectorWriter dataWriter = e57Writer.SetUpData3DPointsData(dataIndex, scanHeader.pointCount, buffers);
            dataWriter.write(scanHeader.pointCount);
            dataWriter.close();

            // Clean up the memory used for buffers
            // DeleteBuffers(buffers);

            std::cout << "Processed scan " << scanHeader.name << " from file: " << inputE57Filename << std::endl;
        }
    }

    // Close the writer after all scans are written
    e57Writer.Close();
    if (e57Files.size() < 2) {            
        std::cout << "Transformed E57 point cloud written to " << outputPath.string() << std::endl;
    } else {
        std::cout << "Merged and transformed E57 point cloud written to " << outputPath.string() << std::endl;
    }

}


void unloadScans() {
    for (auto it = loadedScans.begin(); it != loadedScans.end();) {
        auto& scan = it->second;

        if (!scan.isRendered) {
            // Clear the point cloud data
            if (scan.fullCloud) {
                scan.fullCloud->clear();
                scan.fullCloud.reset(); // Release the shared_ptr memory
            }
            if (scan.subsampledCloud) {
                scan.subsampledCloud->clear();
                scan.subsampledCloud.reset(); // Release the shared_ptr memory
            }

            // Log the unload action
            std::cout << "Scan unloaded: " << it->first << std::endl;

            // Erase the scan from the map (this also frees the memory for the key and value)
            it = loadedScans.erase(it);
        } else {
            ++it; // Move to the next scan if this one is still rendered
        }
    }

    // At this point, all unrendered scans are completely removed from RAM
}


void RecomputeICPWithRelPose(const std::string& oldFile, const std::string& newFile, std::vector<FileEntry>& fileEntries, const int downsample) {
    // Read the old ICP results
    std::unordered_map<std::string, ICPResult> oldICPResults;
    try {
        oldICPResults = ReadICPFromFile(oldFile);
    } catch (const std::exception& e) {
        std::cerr << "Error reading old ICP results: " << e.what() << "\n";
        return;
    }

    // Update fileEntries based on oldICPResults
    for (const auto& [key, oldResult] : oldICPResults) {
        // Extract source and target names from the old result
        std::string sourceName = oldResult.sourceName;
        std::string targetName = oldResult.targetName;

        for (auto& entry : fileEntries) {
            if (entry.hasMultipleScans) {
                for (size_t i = 0; i < entry.scanNames.size(); i++) {
                    std::string scanKey = entry.filepath + "_" + entry.scanNames[i];
                    if (scanKey == sourceName || scanKey == targetName) {
                        entry.scanCheckboxes[i] = true;
                    } else {
                        entry.scanCheckboxes[i] = false;
                    }
                }
            } else {
                std::string scanKey = entry.filepath + "_" + entry.scanNames[0];
                if (scanKey == sourceName || scanKey == targetName) {
                    entry.checkbox = true;
                } else {
                    entry.checkbox = false;
                }
            }
        }
        DrawAll();
        DrawScanPositions(scanPositions);
        pangolin::FinishFrame();

        // Process all scans (fileEntries with updated scanCheckbox)
        ProcessSelectedScans(fileEntries);
        

        // Log ICP computation
        std::cout << "Recomputing ICP for pair: " << sourceName << " <-> " << targetName << "\n";

        // Perform ICP
        ICP(fileEntries,downsample); // This function updates the global `currICP` variable
        DrawAll();
        DrawScanPositions(scanPositions);
        pangolin::FinishFrame();

        // Save the result to the new file
        saveICPtoFile(currICP,newFile); // Append to new file
        unloadScans();

        // Notify the user
        std::cout << "Recomputed ICP results with relPose saved to file: " << newFile << "\n";
    }

    
}







int main() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Prompt the user to select one or more E57 files
    const char* filterPatterns[] = {"*.e57"};
    const char* filenames = tinyfd_openFileDialog(
        "Select E57 files",
        "", // Default path
        1, // Number of filter patterns
        filterPatterns, // Filter patterns
        "E57 files", // Single filter description
        1 // Allow multiple selects
    );
    

    if (!filenames) {
        std::cerr << "No files selected." << std::endl;
        return 1;
    }



    // Split the filenames string by '|'
    std::vector<std::string> e57Filepaths;
    std::string filenamesStr(filenames);
    size_t pos = 0;
    while ((pos = filenamesStr.find('|')) != std::string::npos) {
        e57Filepaths.push_back(filenamesStr.substr(0, pos));
        filenamesStr.erase(0, pos + 1);
    }
    e57Filepaths.push_back(filenamesStr); // Add the last file path

    std::filesystem::path EfilePath(e57Filepaths[0]);
    curr_path = EfilePath.parent_path();

    // Define window size
    int window_width = 1500;
    int window_height = 1000;

    float aspect = static_cast<float>(window_width) / window_height;

    // Create window and setup camera
    pangolin::CreateWindowAndBind("GeoRefHut", window_width, window_height);


    glEnable(GL_DEPTH_TEST);

    glClearColor(0.99f, 0.99f, 1.0f, 1.0f);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(window_width, window_height, 700, 650, window_width / 2.0, window_height / 2.0, 0.05, 40.0),
        pangolin::ModelViewLookAt(3, 3, 1.8, 0, 0, 0, pangolin::AxisZ) 
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, static_cast<double>(-window_width / window_height))
        .SetHandler(new pangolin::Handler3D(s_cam));

    // Choose a sensible left UI Panel width based on the width of 20
    // charectors from the default font.
    const int UI_WIDTH = 20* pangolin::default_font().MaxWidth();


    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<bool> f_button("ui.Full resolution",false,false);
    pangolin::Var<float> point_size_var("ui.Point size [px]", 1.0f, 0.1f, 10.0f);
    pangolin::Var<float> render_dist_var("ui.Render distance [m]", 200.0f, 5.0f, 1000.0f);
    pangolin::Var<bool> scan_box ("ui.Render scan positions", false, true);
    pangolin::Var<bool> link_box ("ui.Render links", false, true);
    pangolin::Var<bool> label_box ("ui.Render scan labels", false, true);
    pangolin::Var<std::string> a_string("ui.Target name", "");
    pangolin::Var<bool> s_button("ui.Save target",false,false);
    // pangolin::Var<std::string> spacer1("ui.Space1", " ");
    pangolin::Var<bool> d_button("ui.Redraw selection",false,false);
    // pangolin::Var<std::string> spacer2("ui.Space2", " ");
    pangolin::Var<bool> m_button("ui.Merge",false,false);
    pangolin::Var<bool> t_button("ui.Transform",false,false);
    // pangolin::Var<std::string> spacer3("ui.Space3", " ");
    pangolin::Var<bool> fix_button("ui.Fix relative pose",false, false);
    pangolin::Var<bool> del_link_button("ui.Delete link", false, false);
    pangolin::Var<bool> i_button("ui.ICP registration",false,false);
    pangolin::Var<int> downsample_var("ui.ICP downsample [cm]", 5, 1, 100);
    pangolin::Var<bool> save_button("ui.ICP to file",false,false);
    pangolin::Var<bool> hessian_box ("ui.Use hessians", false, true);
    // pangolin::Var<std::string> lambda_string("ui.lambda []", "0.001");
    // pangolin::Var<bool> hessian_diag_box ("ui.Use diagonal hessians", false, true);
    pangolin::Var<bool> stoch_box ("ui.Use custom stochastics", false, true);
    pangolin::Var<std::string> trans_string("ui.sigma_t [m]", "0.01");
    pangolin::Var<std::string> rot_string("ui.sigma_r [rad]", "0.001");
    // pangolin::Var<float> trans_sig_var("ui.Transl err [m]", 0.01f, 0.001f, 10.0f);
    // pangolin::Var<float> rot_sig_var("ui.Rot err [rad]", 0.001f, 0.0001f, 1.0f);
    pangolin::Var<bool> o_button("ui.ICP Optimization",false,false);
    pangolin::Var<bool> apply_button("ui.Apply poses", false, false);
    // pangolin::Var<std::string> spacer4("ui.Space4", " ");
    pangolin::Var<bool> unload_button("ui.Unload scans",false, false);
    // pangolin::Var<std::string> spacer5("ui.Space5", " ");
    pangolin::Var<bool> recomp_button("ui.Recompute ICPs",false, false);
    // pangolin::Var<std::string> spacer6("ui.Space6", " ");
    
    pangolin::Var<bool> normals_button("ui.Compute normals",false, false);
    pangolin::Var<int> radius_var("ui.Search radius [cm]", 5, 1, 100);
    pangolin::Var<int> angle_var("ui.Max zenith dist [°]", 15, 0, 90);
    // pangolin::Var<bool> subsample_cloud_button("ui.Subsample cloud", false, false);
    // pangolin::Var<int> subsample_size("ui.Cloud size [kpts]", 500, 1, 10000);
    pangolin::Var<bool> expo_button("ui.Export selected",false, false);
    // pangolin::Var<bool> rel_cov_button("ui.Compute rel covariance",false, false);
    

    pangolin::CreatePanel("scan_list")
        .SetBounds(0.0, 1.0, (1.0 - (UI_WIDTH / static_cast<float>(window_width))), 1.0);
  

    // pangolin::Var<std::string> scan_string("ui_scan_list.Scans:", "");


    

    // Store the handler manually
    CustomHandler3D* custom_handler = new CustomHandler3D(s_cam, d_cam, UI_WIDTH);
    d_cam.SetHandler(custom_handler);


    // Create file entries for all E57 files and read scan names
    std::vector<FileEntry> fileEntries;
    for (const auto& filepath : e57Filepaths) {
        FileEntry entry = readE57Header(filepath);
        fileEntries.push_back(entry);
    }
    
   setglobalShift(fileEntries);
   scanPositions = ReadAllScanPositions(fileEntries);
   
    

    // Create checkboxes for the parent E57 files
    CreateParentCheckboxes(fileEntries);

    lines = updateLinks(scanPositions);

    

    bool drawPushed = false;
    while (!pangolin::ShouldQuit()) {

        draw_labels = label_box.Get();
        draw_links = link_box.Get();
        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -aspect);
        
        custom_stoch = stoch_box.Get();
        use_hessian = hessian_box.Get();
        // use_hessian_diag= hessian_diag_box.Get();
        
        float render_dist = render_dist_var.Get();
        UpdateWindowSize(window_width, window_height, aspect);
        s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(window_width, window_height, 700, 650 / aspect, window_width / 2.0, window_height / 2.0, 0.05, render_dist));
        
        pangolin::DisplayBase().ActivateScissorAndClear(s_cam);

        d_cam.Activate(s_cam);

        // Render the checkboxes and sublists
        RenderCheckboxesAndSublist(fileEntries);

        float point_size = point_size_var.Get(); // Get the point size from the UI variable
        glPointSize(point_size);
        Eigen::Vector3d* picked_point = custom_handler->GetPickedPoint();

        // Load and draw point cloud
        if (pangolin::Pushed(d_button)) {
            ProcessSelectedScans(fileEntries);
            // if (!drawPushed) {
            //     UpdateView(recentScanPose,s_cam);
            //     drawPushed = true;
            // }
            if (picked_point) {
                pangoCloudCube->clear();
                GetPointsInSphere(picked_point->cast<float>());
            }
        }

        DrawAll();
        

        if (custom_handler) {
            custom_handler->DrawPickedPoint();
        } else {
            std::cerr << "CustomHandler3D is null" << std::endl;
        }

        
        // Full-resolution rendering logic on button click
        if (pangolin::Pushed(f_button)) {
            // Eigen::Vector3d* picked_point = custom_handler->GetPickedPoint();
            if (picked_point) {
                pangoCloudCube->clear();
                GetPointsInSphere(picked_point->cast<float>());
            } else {
                std::cout << "No point picked yet." << std::endl;
            }
        }
        

        // Save target functionality
        if (pangolin::Pushed(s_button)) {
            std::string targetName = a_string.Get();
            const Eigen::Vector3d* picked_point = custom_handler->GetPickedPointGlobal();
    
            if (targetName.empty()) {
                std::cout << "Target has no name. Please enter a name before saving." << std::endl;
            } else if (picked_point) {
                std::ofstream outfile("local.txt", std::ios::app);
                outfile << std::fixed << std::setprecision(3); 
                outfile << targetName << " " << picked_point->transpose() << std::endl;
                outfile.close();
                std::cout << "Target saved: " << targetName << std::endl;
            } else {
                std::cout << "No point selected to save." << std::endl;
            }
        }

        // Transformation:
        if (pangolin::Pushed(t_button)) {
            for (auto& scan : loadedScans) {
                scan.second.isRendered = false;
            }
            unloadScans();
            // ProcessSelectedScans(fileEntries);
            pangolin::FinishFrame();
            DrawAll();
            pangolin::FinishFrame();
            transformE57(fileEntries);
        }

        // if (pangolin::Pushed(rel_cov_button)) {
        //     auto stoch_a_priori = std::make_pair(
        //         std::stod(trans_string),
        //         std::stod(rot_string)
        //         // static_cast<double>(trans_sig_var.Get()), 
        //         // static_cast<double>(rot_sig_var.Get())
        //     );
        //     // if (loadedScans)
        //     // auto lambda = std::stod(lambda_string.Get());            
        //     std::unordered_map<std::string, ICPResult> icpR = ReadICPFromFile("icp_results.json");
        //     com_rel_cov = true;
        //     computeRelCov(icpR,"optimized_poses.json",stoch_a_priori);
        //     com_rel_cov = false;
        // }

        if (pangolin::Pushed(i_button)) {
            int downsample = downsample_var.Get();
            ICP(fileEntries,downsample);
            if (picked_point) {
                pangoCloudCube->clear();
                GetPointsInSphere(picked_point->cast<float>());
            }
        }

        if (pangolin::Pushed(save_button)) {
            saveICPtoFile(currICP,"icp_results.json");
            lines = updateLinks(scanPositions);
        }

        if (pangolin::Pushed(m_button)) {
            mergeE57();
        }

        if (scan_box.Get()) {
            DrawScanPositions(scanPositions);
        }

        if (pangolin::Pushed(o_button)) {
            auto stoch_a_priori = std::make_pair(
                std::stod(trans_string),
                std::stod(rot_string)
                // static_cast<double>(trans_sig_var.Get()), 
                // static_cast<double>(rot_sig_var.Get())
            );
            // if (loadedScans)
            // auto lambda = std::stod(lambda_string.Get());            
            std::unordered_map<std::string, ICPResult> icpR = ReadICPFromFile("icp_results.json");
            OptimizePoseGraph(icpR,"optimized_poses.json",stoch_a_priori);
            posesOptimized = true;
        }

        if (pangolin::Pushed(apply_button)) {
            applyPoses("optimized_poses.json");
        }
        if (pangolin::Pushed(unload_button)) {
            unloadScans();
        }
        if (pangolin::Pushed(fix_button)) {
            fixPoses(fileEntries);
        }
        if (pangolin::Pushed(recomp_button)) {
            int downsample = downsample_var.Get();
            RecomputeICPWithRelPose("icp_results.json","icp_results_automated.json",fileEntries, downsample);
        }
        if (pangolin::Pushed(del_link_button)) {
            DeleteResultsByRenderedStatus("icp_results.json");
        }
        if (pangolin::Pushed(normals_button)) {
            float radius = static_cast<float>(radius_var.Get())/100.0f;
            float angle = static_cast<float>(angle_var.Get());
            float zenithThresholdCos = std::cos(30.0f * M_PI / 180.0f);
            for (auto& entry : fileEntries) {
                entry.checkbox = true;
                for (size_t i = 0; i<entry.scanCheckboxes.size(); i++) {
                    entry.scanCheckboxes[i] = true;
                    ProcessSelectedScans(fileEntries);
                    for (auto& scan : loadedScans) {
                        if (scan.first == (entry.filepath + "_" + entry.scanNames[i])) {
                            scan.second.isRendered = true;
                            std::cout << "computing normals for " << scan.first << std::endl;
                            std::vector<Eigen::Vector3f> points = scan.second.subsampledCloud->getVertices();
                            std::vector<Eigen::Vector3f> normals;
                            computeNormals(points,normals,radius,zenithThresholdCos);
                            saveNormalStatistics(normals,"normal_statistics.json",scan.first);
                            std::cout << "normals saved: " << scan.first << std::endl;
                            scan.second.isRendered = false;
                            // ProcessSelectedScans(fileEntries);
                        } 
                        
                    }
                    unloadScans();
                    entry.scanCheckboxes[i] = false;
                }
                entry.checkbox = false;
            }
        }
        if (pangolin::Pushed(expo_button)) {
            exportE57(fileEntries);
        }
        pangolin::FinishFrame();
    }
    // Cleanup
    delete custom_handler;
    return 0;
}
