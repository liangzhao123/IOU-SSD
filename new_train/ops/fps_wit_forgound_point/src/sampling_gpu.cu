#include "sampling.h"
#include <stdio.h>
#include <stdlib.h>
//#define DEBUG
#include "cmath"
#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#include "math.h"


__global__  void addkernel(int row, int col, const float *aa, const float *bb, float *cc){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.x;

    cc[i*col+j]=aa[i*col+j]+bb[i*col+j];
//    printf("i:%d,j:%d ",i,j,"\n");

};



void AddLaunch(const float *a,const float *b,float *c, int size[],cudaStream_t stream){
//    int *pts_assign = NULL;
//    cudaMalloc(&pts_assign, 1 * 1 * 1 * sizeof(int));
//    int N = DIVUP(3,5);
    int x = size[0],y = size[1];
    dim3 numblock(2,3,1);
    dim3 threadPerBlock(1);
    addkernel<<<numblock,threadPerBlock,0,stream>>>(x,y,a,b,c);
//    cudaDeviceSynchronize();  // for using printf in kernel function
}




inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);

}



__global__ void gather_points_kernel_fast(int b, int c, int n, int m,
                                          const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];

}





__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}



template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m,
                                               const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    //c is the num of channel
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];


//    printf("block size %d", block_size);

    int batch_index = blockIdx.x;//range from 0-batch size
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;


    int tid= threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];

        int for_cout = 1;
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            // if (mag <= 1e-3)
            // continue;
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;//the distance to fps points
            besti = d2 > best ? k : besti;//the further point id
            best = d2 > best ? d2 : best;// the further point distance
            for_cout++;
        }
//        printf("for cout %d", for_cout);

//        printf("bockIdx.x %d,threadIdx.x:%d, d2:%.4f",batch_index,tid,for_cout);
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();

        }

        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();

        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();

        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();

        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}


void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
                                        const float *points, const int *idx, float *out, cudaStream_t stream) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<blocks, threads, 0, stream>>>(b, c, n, npoints, points, idx, out);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}





void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp, int *idxs, cudaStream_t stream) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // ouut:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);
    switch (n_threads) {
        case 1024:
            furthest_point_sampling_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
            furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
            furthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
            furthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
            furthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
            furthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
            furthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
            furthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
            furthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
            furthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
            furthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        default:
            furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }
//    cudaDeviceSynchronize();//using for print in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

}

template <unsigned int block_size>
__global__ void further_sample_points_with_features_kernel(int b, int n, int m,int m1,int c,
                                      const float *__restrict__ dataset,const int *__restrict__ predix, float *__restrict__ temp,float *__restrict__ values1,float *__restrict__ values2, int *__restrict__ idxs){

    if (m <= 0) return;
    __shared__ float dist[block_size];
    __shared__ int disti[block_size];
    int bs = blockIdx.x;
    dataset += bs*n*c;
    temp += bs*n;
    idxs += bs*m;
    predix += bs*m1;


//    predix += bs*m1;0

    int tid = threadIdx.x;
    const int stride = block_size;

    int pred_idx;
    int sums = 0;
    for (int j = tid;j<n; j += stride){//Traverse all raw points
//        printf(" %d ",j);
        sums++;
        float pred_best = 1e38;
        float pred_p1,pred_p2;
        for(int k=0;k<m1;k++){
//            printf(" %d ",k);//traverse all forground points to inital temp
            pred_idx = predix[k];
            float pred_d = 0;
            for(int c_id=0;c_id<c;c_id++){
                pred_p1= dataset[pred_idx*c+c_id];
                pred_p2 = dataset[j*c+c_id];
                pred_d += (pred_p2-pred_p1)* (pred_p2-pred_p1);

            }//calculate (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (y2-y1)*(y2-y1)...
            pred_best = min(pred_best,pred_d);
        }
        temp[j] = pred_best;
    }

    __syncthreads();//initalize the old and pred_best
    int old=0;
    float pred_best = -1;
    for (int i=0; i<n;i++){
        if(pred_best<temp[i]){
            pred_best = temp[i];
            old=i;
        }
    }
    if (threadIdx.x==0){
        idxs[0] = old;//randomly select a thread and randomly give the firt idx of keypoints
    }
    __syncthreads();
    for (int i = 1; i<m;i++){
        float best = -1;
        int besti=0;
        float x1 = dataset[old*c+0];
        float y1 = dataset[old*c+1];
        float z1 = dataset[old*c+2];
//        float f1 = dataset[old*c+3];
//        for (int key_id= 0;key_id<c;key_id++){
//            values1[key_id] = dataset[old*c+key_id];
//        }//get the x1,y1,z1,f11,f21,f31,...of keypoints
//        printf("%.f",x1);
        for (int k= tid;k<n;k+=stride){
            float x2 = dataset[k*c+0];
            float y2 = dataset[k*c+1];
            float z2 = dataset[k*c+2];
//            float f2 = dataset[k*c+3];
//            for (int raw_id=0;raw_id<c;raw_id++){
//                values2[raw_id] = dataset[k*c+raw_id];
//            }//get the x2,y2,z2,f12,f22,f32,f42,.... of raw point cloud

//            float d = 0;
//            for(int c_id = 0 ; c_id<c;c_id++){
//                d += (values2[c_id]-values1[c_id])*(values2[c_id]-values1[c_id]);
//            }//calculate the distance between the sampled and unsampled points
            float d  = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
            float d2 = min(temp[k],d);
            temp[k] = d2;//identify the distance between the sampled points and unsampled points
            if(d2>best){
            best = d2;
            besti = k;
            }//chose the further points
        }
        dist[tid] = best;
        disti[tid] = besti;
        __syncthreads();
        if(block_size>=1024){
            if(tid<512){
                __update(dist,disti,tid,tid+512);
            }
            __syncthreads();
        }
        if (block_size>=512){
            if (tid<256){
                __update(dist,disti,tid,tid+256);
            }
            __syncthreads();
        }
        if(block_size>=256){
            if (tid<128){
                __update(dist,disti,tid,tid+128);
            }
            __syncthreads();
        }
        if(block_size>=128){
            if(tid<64){
                __update(dist,disti,tid,tid+64);
            }
            __syncthreads();
        }
        if(block_size>=64){
            if(tid<32){
                __update(dist,disti,tid,tid+32);
            }
            __syncthreads();
        }
        if(block_size>=32){
            if(tid<16){
                __update(dist,disti,tid,tid+16);
            }
            __syncthreads();
        }
        if(block_size>=16){
            if(tid<8){
                __update(dist,disti,tid,tid+8);
       }
            __syncthreads();
        }
        if(block_size>=8){
            if(tid<4){
                __update(dist,disti,tid,tid+4);
            }
            __syncthreads();
        }
        if(block_size>=4){
            if(tid<2){
                __update(dist,disti,tid,tid+2);
            }
            __syncthreads();
        }
        if(block_size>=2){
            if(tid<1){
                __update(dist,disti,tid,tid+1);
            }
            __syncthreads();
        }
        old = disti[0];
        if(tid==0){
            idxs[i] = old;
        }// for avoid repeatly assginment in all thread , so there is if condition , in fact it can be replaceed by other format like tid==1 or tid ==3
    }
}

void further_point_sampling_with_features_kernel_launch(int b, int n, int m,int m1,int c,
                                  const float *dataset,const int *predix, float *temp, int *idxs, float *values1, float * values2, cudaStream_t stream){
//values1 is a 1D array shape is (c) c is the channel of datatset
    cudaError_t err;
    unsigned int n_tread =opt_n_threads(n);
    switch (n_tread) {
        case 1024:
            further_sample_points_with_features_kernel<1024><<<b,n_tread>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 512:
            further_sample_points_with_features_kernel<512><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 256:
            further_sample_points_with_features_kernel<256><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 128:
            further_sample_points_with_features_kernel<128><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 64:
            further_sample_points_with_features_kernel<64><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 32:
            further_sample_points_with_features_kernel<32><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 16:
            further_sample_points_with_features_kernel<16><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 8:
            further_sample_points_with_features_kernel<8><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 4:
            further_sample_points_with_features_kernel<4><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 2:
            further_sample_points_with_features_kernel<2><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        case 1:
            further_sample_points_with_features_kernel<1><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
        default:
            further_sample_points_with_features_kernel<512><<<b,n_tread,0,stream>>>(b,n,m,m1,c,dataset,predix,temp,values1,values2,idxs);
            break;
    }
//    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}





