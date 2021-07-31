//
// Created by liang on 2020/8/28.
//
#include "first_pkg/first_pkg.h"
#include "sampling.h"
#include <fstream>
using namespace std;
#include <torch/serialize/tensor.h>

void lidar_callback(const sensor_msgs::PointCloud2 &input) {
    pcl::PointCloud<pcl::PointXYZI> outpoint;
    pcl::fromROSMsg(input, outpoint);
    outpoint.points[0].getVector4fMap();
    vector<vector<float>> points = {{0,1.,2.,3.}};
    cout<<"Done"<<endl;
}




int inital_ros(int argc, char **argv) {
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/kitti/velo/pointcloud", 1, lidar_callback);
    std::cout << "hja" << std::endl;
    ros::Rate rate(20);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();

    }
    return 0;
}





vector<point_3d> read_bin()
{
    ifstream ifs("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training/velodyne/000000.bin",std::ios::binary);
    float xyzi[4]={};
    vector<point_3d> pts;
    if(!ifs){
        cout<<"read error"<<endl;
        abort();
    }
    while (!ifs.eof()){
        ifs.read((char *)xyzi, sizeof(xyzi));
        point_3d op(xyzi[0],xyzi[1],xyzi[2],xyzi[3]);
        pts.push_back(op);
    }
    ifs.clear();
    ifs.close();
    return pts;
}



vector<float> read_bin_to_vector(){
    ifstream ifs("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training/velodyne/000211.bin",std::ios::binary);
    float data[4] = {};
    vector<float> points;
    if(!ifs){
        cout<<"error open file"<<endl;
        abort();
    }
    while (!ifs.eof()){
        ifs.read((char *) data,sizeof(data));
        points.push_back(float (data[0]));
        points.push_back(float (data[1]));
        points.push_back(float (data[2]));
        points.push_back(float (data[3]));
    }
    ifs.clear();
    ifs.close();
    return points;
}






at::Tensor fps() {
    //b: batch size;
    //n :of raw points;
    //m: num of keypoints;
    //point_tensor: shape(batch size, num of raw points, channels),
    // the channels for the raw points are 4 ,x,y,z,i respectively,
    //temp tensor:shape(batch size, num of raw points) this is used for save the min distance
    //out: shape( batch size, num of keypoints) this is the idx in raw points for indicing the keypoints from raw points;

//    clock_t start = clock();
//    vector<point_3d> pts= read_bin();
//    cout<<double(clock()-start)/CLOCKS_PER_SEC<<endl;
    int b = 1;
    int n,m=2048;
    vector<float> pts = read_bin_to_vector();
    int sizes = pts.size();
    n = sizes/4;

    float * points = pts.data();
    torch::Tensor point_tensor = torch::from_blob(points, sizes,torch::kFloat32).to("cuda");
    point_tensor = at::reshape(point_tensor,{1,-1,4}).cuda();

    torch::Tensor indices = torch::tensor({ 0, 1, 2 },at::kLong).cuda();
    at::Tensor point_tensor_for_fps = torch::index_select(point_tensor,2,indices).cuda();

    torch::Tensor temp = at::ones({b,n},torch::kFloat32).cuda();

    at::Tensor out = at::ones({b,m},torch::kInt32).cuda();

    string cpu_or_cuda = "cuda";

    if(cpu_or_cuda=="cuda"){
        clock_t start = clock();
        furthest_point_sampling_wrapper(b,n,m,point_tensor_for_fps,temp,out);
//        cout<<(clock()-start)/CLOCKS_PER_SEC<<endl;
        cout<<double(clock()-start)/CLOCKS_PER_SEC<<endl;
    } else{
        assert(cpu_or_cuda=="cpu");
        point_tensor_for_fps = point_tensor_for_fps.cpu();
        temp =temp.cpu();
        out = out.cpu();
        clock_t start = clock();
        float *pts = point_tensor_for_fps.data<float>();
        int *idx = out.data<int>();
        float *temp_ptr = temp.data<float>();
        fps_cpu(b,n,m,pts,temp_ptr,idx);
        cout<<(clock()-start)/CLOCKS_PER_SEC<<endl;
//        cout<<out<<endl;
    }

    out = out.type_as(indices);
    out = torch::squeeze(out,0).cuda();
//    cout<<out<<endl;
    point_tensor = at::index_select(point_tensor,1,out).cuda();
//    cout<<point_tensor<<endl;
    return point_tensor;
//    test_the_cuda_add_fun();
}


void pub_raw_cloud(int argc, char **argv){
    vector<point_3d> pts= read_bin();
    ros::init(argc,argv,"raw_point_cloud_publisher");
    ros::NodeHandle n;
    ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud>("raw_coud",1);
    unsigned int num_points = pts.size();
    ros::Rate r(10);
    while (n.ok()){
        sensor_msgs::PointCloud cloud;
        cloud.header.stamp = ros::Time::now();
        cloud.header.frame_id = "lidar";
        cloud.points.resize(num_points);
        cloud.channels.resize(1);
        cloud.channels[0].name = "rgb";
        cloud.channels[0].values.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            cloud.points[i].x = pts[i].mx;
            cloud.points[i].y = pts[i].my;
            cloud.points[i].z = pts[i].mz;
            cloud.channels[0].values[i] = pts[i].mi;
        }
        pc_pub.publish(cloud);
        r.sleep();
    }
}

void pub_fps_cloud(int argc, char **argv){
    torch::Tensor keypoints = fps();
    ros::init(argc,argv,"fps_point_cloud_publisher");
    ros::NodeHandle n;
    ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud>("fps_coud",1);
    unsigned int num_points = keypoints.size(1);
    ros::Rate r(10);
    while (n.ok()){
        sensor_msgs::PointCloud cloud;
        cloud.header.stamp = ros::Time::now();
        cloud.header.frame_id = "lidar";
        cloud.points.resize(num_points);
        cloud.channels.resize(1);
        cloud.channels[0].name = "rgb";
        cloud.channels[0].values.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            cloud.points[i].x = keypoints[0][i][0].item().toFloat();
            cloud.points[i].y = keypoints[0][i][1].item().toFloat();
            cloud.points[i].z = keypoints[0][i][2].item().toFloat();
            cloud.channels[0].values[i] = keypoints[0][i][3].item().toFloat();
        }
        pc_pub.publish(cloud);
        r.sleep();
    }

}


void point_group(){
    //b : batch size
    //c: channels for features
    //n: num of raw points per batch size
    //npoints : num of keypoints
    //features: (bach size, channels, num of raw points)
    int b = 1;
    int c = 32;
    int n = 10;
    int npoints = 5;
    torch::Tensor features = torch::rand({b,c,n},torch::kFloat32).to("cuda");
//    std::cout<<features<<std::endl;
    torch::Tensor idx_tensor = torch::randint(0,n,{b,npoints},torch::kInt32).to("cuda");
//    std::cout<<idx_tensor<<std::endl;
    torch::Tensor out = torch::zeros({b,c,npoints},torch::kFloat32).to("cuda");
    std::cout<<out.tensor_data()<<std::endl;

    gather_points_wrapper_fast(b,  c,  n, npoints,
                               features,  idx_tensor, out);
    std::cout<<out.tensor_data()<<std::endl;
}


//void test_the_cuda_add_fun(){
//
//    float arr_1[2][3] = {{3.0, 4.0, 5.0},{1.2, 3.6, 7.8}};
//    float arr_2[2][3] = {{1.0, 1.0, 1.0},{2.,1.0,1.}};
//
////    nc::NdArray<float> array = {{1,2,3},{34,1,2}};
//    torch::Tensor tensor_1 = torch::from_blob(arr_1,{2,3}).to("cuda");//rray to tensor
//    std::cout<<tensor_1<<std::endl;
//    torch::Tensor tensor_2 = torch::from_blob(arr_2,{2,3}).to("cuda");
//    std::cout<<tensor_2<<std::endl;
//    torch::Tensor  out = torch::zeros({2,3},torch::TensorOptions().dtype(torch::kFloat32)).cuda();
//
//    //get the shape x
//    std::cout<<"inital value"<<out<<std::endl;
//    add_cuda(tensor_1,tensor_2,out);
//    std::cout<<"1+2"<<out<<std::endl;

//}
//int fps_with_features_wrapper(int b, int n, int m, int c,
//                              at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor)
at::Tensor read_pred_idx(){
    ifstream ifs("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/seg_label/training/000211.bin",std::ios::binary);
    long data={} ;
    vector<long> points;
    if(!ifs){
        cout<<"error open file"<<endl;
        abort();
    }
    while (!ifs.eof()){
        ifs.read((char *) &data,sizeof(data));
        points.push_back(long (data));
    }
    ifs.clear();
    ifs.close();
//    points.pop_back();
    torch::Tensor f_points_idx = torch::tensor(points,at::kInt).cuda();
    f_points_idx = f_points_idx.reshape({1,-1,1});
    return f_points_idx;
}

void test_fps_f(){
    int b = 1;
    int n,m=2048;
    vector<float> pts = read_bin_to_vector();
    int sizes = pts.size();
    n = sizes/4;
    int c = 3;
    float * points = pts.data();
    torch::Tensor point_tensor = torch::from_blob(points, sizes,torch::kFloat32).to("cuda");
    point_tensor = at::reshape(point_tensor,{1,-1,4}).cuda();
    torch::Tensor indices = torch::tensor({0,1,2},at::kLong).cuda();
    at::Tensor point_tensor_for_fps = torch::index_select(point_tensor,2,indices).cuda();
    torch::Tensor temp = at::ones({b,n},torch::kFloat32).cuda().fill_(1e10);
    at::Tensor out = at::ones({b,m},torch::kInt32).cuda();
    at::Tensor pred_idx = read_pred_idx();
    int m1 = pred_idx.size(1);
    clock_t start = clock();
    fps_with_features_wrapper(b,n,m,m1,c,point_tensor_for_fps,pred_idx,temp,out);
    cout<<double(clock() - start)/CLOCKS_PER_SEC<<endl;
//    cout<<out<<endl;
}

void test_points_idx(){
    at::Tensor pred_idx = read_pred_idx().cuda();
    pred_idx = pred_idx.squeeze(0);
    pred_idx = pred_idx.squeeze(1).to(torch::kLong);
    cout<<pred_idx<<endl;
    vector<float> pts = read_bin_to_vector();
    int sizes = pts.size();
    int n = sizes/4;
    int c = 3;
    float * points = pts.data();
    torch::Tensor point_tensor = torch::from_blob(points, sizes,torch::kFloat32).to("cuda");
    point_tensor = point_tensor.reshape({1,-1,4});
    cout<<point_tensor<<endl;
    point_tensor = torch::index_select(point_tensor,1,pred_idx);
    cout<<point_tensor<<endl;
}

int main(int argc, char **argv){

    test_fps_f();

//     pub_raw_cloud(argc,argv);
//    pub_fps_cloud(argc, argv);
//
//    thread pub_raw_pt(pub_raw_cloud,argc,argv);
//    thread pub_pfs_pt(pub_fps_cloud,argc,argv);
//    pub_raw_pt.join();
//    pub_pfs_pt.join();
//    cout<<"main thread finished!"<<endl;
//    torch::Tensor keypoints = fps();
    return 0;
}


