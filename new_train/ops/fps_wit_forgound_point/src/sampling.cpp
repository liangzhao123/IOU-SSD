
#include "sampling.h"
#include "cmath"
#include "math.h"


void fps_cpu(int b,int n, int m,  float *points, float *temp, int *index ){
    int old = 0;
    index[0] = 0;
    for (int i=1;i<m;i++){
        float x1= points[old * 3 + 0];
        float y1 = points[old * 3 + 1];
        float z1 = points[old * 3+ 2];
        float best =-1;
        int besti = 0;
        for (int j= 0;j<n;j++){
            float x2= points[j * 3+0];
            float y2 = points[j * 3 +1];
            float z2 = points[j * 3+2];
            float d2;
            float d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (i==1){
                d2= d;
            } else{
                d2 = std::min(temp[j],d);}
            temp[j] = d2;
            if (d2>best){
                best = d2;
                besti = j;
            } else{
                best = best;
                besti = besti;
            }

        }
        old = besti;
        index[i] = besti;
    }
}


int gather_points_wrapper_fast(int b, int c, int n, int npoints,
                               at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out, stream);
    return 1;
}



int furthest_point_sampling_wrapper(int b, int n, int m,
                                    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {
    //b: batch size;
    //n :of raw points;
    //m: num of keypoints;
    //c is the num of channel of point tensor;
    //point_tensor: shape(batch size, num of raw points, channels),
    // the channels for the raw points are 4 ,x,y,z,i respectively,
    //temp tensor:shape(batch size, num of raw points) this is used for save the min distance
    //out: shape( batch size, num of keypoints) this is the idx in raw points for indicing the keypoints from raw points;
    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}

int fps_with_features_wrapper(int b, int n, int m,int m1, int c,
                              at::Tensor points_tensor, at::Tensor predix_tensor,at::Tensor temp_tensor, at::Tensor idx_tensor){
    //c is the number of channnel of points_tensor
    //m1 is the number of predix
    // predix (m1), m1<n
    at::Tensor valuse1_tensor = torch::ones({c},at::kFloat).cuda();
    at::Tensor values2_tensor = torch::ones({c},at::kFloat).cuda();
//    std::cout<<valuse1_tensor<<std::endl;
    float *values1 = valuse1_tensor.data<float>();
    float *values2 = values2_tensor.data<float>();
    const float *points = points_tensor.data<float>();
    const int *predix = predix_tensor.data<int>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    further_point_sampling_with_features_kernel_launch(b, n, m,m1,c, points,predix, temp, idx,values1,values2, stream);
    return 1;
}




