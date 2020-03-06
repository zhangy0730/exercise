#include <iostream>
#include <ceres/ceres.h>
#include <cmath>
#include <chrono>

using namespace std;
//using namespace ceres;

struct CostFunctor{
    template <typename T>   // 1. 使用模版类的意义是什么 . 好像是 double 和 Jet二选一
    bool operator()(const T* const x , T* residual) const{ // 2. 为什么是bool 3. 传指针是为了传递数组吗
        residual[0] = T(10.0) - x[0]; //residual 也是指针，代表一个数组
        return true;
    }
};

struct NumericDiffCostFunctor{
    bool operator()(const double* const x, double* residual) const{
        residual[0] = 10.0 - x[0];
        return true;
    }
};

class QuadraticCostFunction : public ceres::SizedCostFunction<1,1>{
public :
    virtual ~QuadraticCostFunction(){}
    virtual bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const {
        const double x = parameters[0][0];
        residuals[0] = 10 - x;
        if (jacobians != NULL && jacobians[0]!=NULL){
            jacobians[0][0] = -1; // 是个矩阵
        }
        return true;
    }
};


int main(int argc, char** argv){

    //google::InitGoogleLogging(argv[0]);

    double init_x = 5.0;
    double x = init_x;

    ceres::Problem problem;

    // Problem
    ceres::CostFunction* cost_function = new QuadraticCostFunction;
    //ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<NumericDiffCostFunctor,ceres::CENTRAL,1,1>(new NumericDiffCostFunctor);
    //ceres::CostFunction*  cost_function = new ceres::AutoDiffCostFunction<CostFunctor,1,1>(new CostFunctor);


    //残差项
    problem.AddResidualBlock(cost_function,NULL,&x);

    //slover

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary  summary;
    ceres::Solve(options,&problem,&summary);

    cout << summary.BriefReport() << endl;
    cout << "x : " << init_x << " -> " << x << endl;

    return 0;
}