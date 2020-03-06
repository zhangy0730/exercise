#include <iostream>
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <cmath>
#include <chrono>
#include <fstream>

using namespace std;

// camera ：R(3),t(3),f,k1,k2
// 表示旋转，平移，焦距，扰动模型
// 旋转为 罗德里格向量 θa
// 扰动模型为径向畸变
// r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4
// 相机成像模型：
//              P = R * X + t  世界坐标转相机坐标
//              p = -P / P.z 映射到归一化平面
//              p' = f * r(p) * p 转换到像素坐标

struct SynavelyReprojectionError{

    SynavelyReprojectionError(double x, double y):x_(x),y_(y){};

    template<typename T>
    bool operator()(const T* const camera, const T* const point,T* residual)const {
        
        // 计算P = R * X + t 
        // rotate point

        T p[3]; // point
        T R[3];
        R[0] = camera[0];
        R[1] = camera[1];
        R[2] = camera[2];
        ceres::AngleAxisRotatePoint(R,point,p);

        // translate
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        //计算 P = -P / P.z

        p[0] = -p[0] / p[2];
        p[1] = -p[1] / p[2];
        //p[2] = -p[2] / p[2]; //齐次项

        //计算 p' = f * r(p) * p

        T r2 = p[0] * p[0] + p[1] * p[1];
        T rp = T(1) + camera[7] * r2 + camera[8] * r2 * r2;

        p[0] = camera[6] * rp * p[0];
        p[1] = camera[6] * rp * p[1];
        
        // observed 和 predicted相减
        residual[0] = T(x_) - p[0];
        residual[1] = T(x_) - p[1];

        return true;
    }

private:
    double x_;
    double y_;
};

int main(int argc,char** argv){

    ifstream myfile("../data/problem-49-7776-pre.txt");
    if(!myfile.is_open()){
        cout << "file not found" << endl;
        return 0;
    }

    int num_cameras,num_points,num_observations;
    myfile >> num_cameras >> num_points >> num_observations;

    // 待估计变量同时包括所有的point和所有的point九维信息

    int camera_ind[40000],point_ind[40000];
    double data_x[40000],data_y[40000];
    double* camera_para = new double[900];
    double* point_para = new double[30000];

    for(int i = 0 ; i < num_observations ; i++ ){
        myfile >> camera_ind[i] >> point_ind[i] >> data_x[i] >> data_y[i];
    }

    for(int t = 0 ; t < num_cameras ; t++){
        int i = t*3;
        myfile  >> camera_para[i] >> camera_para[i+1] >> camera_para[i+2]
                >> camera_para[i+3] >> camera_para[i+4] >> camera_para[i+5]
                >> camera_para[i+6] >> camera_para[i+7] >> camera_para[i+8] ;
    }

    for(int t = 0 ; t < num_points ; t++){
        int i = t*3;
        myfile  >> point_para[i] >> point_para[i+1] >> point_para[i+2];
    }


    ceres::Problem problem;

    for(int i = 0 ; i < num_observations ; i++ ){
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SynavelyReprojectionError,2,9,3>(new SynavelyReprojectionError(data_x[i],data_y[i]));
        problem.AddResidualBlock(   cost_function,
                                    NULL,
                                    camera_para + camera_ind[i] * 9,
                                    point_para + point_ind[i] * 3);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 400;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    delete [] camera_para;
    delete [] point_para;
    return 0;
}