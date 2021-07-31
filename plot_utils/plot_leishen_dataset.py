from single_stage_model.dataset.leishen_dataset.utils import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen_e70_32/dataset_image_pcd"
OUTPUT_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1"

subplot_fig_size=(20,30)
subplot_xticks_size=20
subplot_yticks_size=20
subplot_xlabel_size=20
subplot_ylabel_size=20
subplot_title_size=20
subplot_legend_size=20

fig_size = (8, 6)
xticks_size = 20
yticks_size = 20
xlabel_size = 20
ylabel_size = 20
title_size = 20
legend_size = 20
bins_number = 20

def read_class_number_data(path):
    with open(path,"rb") as f:
        data = pickle.load(f)
    return data
def plot_class_num_bar(path):
    data = read_class_number_data(path)
    class_number = data["number_of_box"]
    classes,number = [],[]
    class_number = class_number.items()
    classes = [i[0] for i in class_number]
    number = [i[1] for i in class_number]
    plt.figure(figsize=fig_size)
    plt.bar(classes,number)
    plt.title("Number of instances per class", fontsize=title_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.ylabel('Number of instances ', fontsize=ylabel_size)
    for a, b in zip(classes, number):
        plt.text(a, b + 50, str(b), ha="center", va="center", fontsize=xticks_size)
    save_path = os.path.join(OUTPUT_DIR,"output","figures","instances_per_class.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    print("done")

def plot_position_distribution(path):

    anno_path_list = glob.glob(os.path.join(path, "*.txt"))

    dataset_info_pth = os.path.join(os.path.abspath(path + "/.."), "annotation_info.pth")

    anno_path_list.sort()
    num_box = 0
    name_dict = {}
    labels_list = []
    gt_boxes_list = []
    for i in range(1, len(anno_path_list) + 1):
        filename = os.path.join(path, str(i).zfill(6) + ".txt")
        try:
            os.path.exists(filename)
            objects = get_objects_from_label(filename)
            labels = [i.cls_type for i in objects]
            labels_list.extend(labels)
            num_box += len(objects)
            gt_boxes = get_info(objects)
            gt_boxes_list.extend(gt_boxes)
            for k in range(len(gt_boxes)):
                gt_box = gt_boxes[k]
                if gt_box[0] > 100 or gt_box[1] > 100 or gt_box[2] > 100 or gt_box[0] < -100 or gt_box[1] < -100 or \
                        gt_box[2] < -100:
                    print("error annotation in :", filename)
        except:
            print(filename)
    labels_list = np.array(labels_list)
    gt_boxes_list = np.array(gt_boxes_list)

    car_index = labels_list=="Car"
    pedestrain_index = labels_list == "Pedestrian"
    Bus_index = labels_list == "Bus"
    Rider_index = labels_list == "Rider"
    Car_gt_box = gt_boxes_list[car_index]
    pedestrain_gt_box = gt_boxes_list[pedestrain_index]
    Bus_gt_box = gt_boxes_list[Bus_index]
    Rider_gt_box = gt_boxes_list[Rider_index]

    Car_distances = [np.sqrt(obj[0]**2+obj[1]**2+obj[2]**2) for obj in Car_gt_box]
    Pedestrain_distances = [np.sqrt(obj[0] ** 2 + obj[1] ** 2 + obj[2] ** 2) for obj in pedestrain_gt_box]
    Bus_distances = [np.sqrt(obj[0] ** 2 + obj[1] ** 2 + obj[2] ** 2) for obj in Bus_gt_box]
    Rider_distances = [np.sqrt(obj[0] ** 2 + obj[1] ** 2 + obj[2] ** 2) for obj in Rider_gt_box]

    Car_degree = [obj[6] for obj in Car_gt_box]
    Pedestrain_degree = [obj[6] for obj in pedestrain_gt_box]
    Bus_degree = [obj[6] for obj in Bus_gt_box]
    Rider_degree = [obj[6] for obj in Rider_gt_box]

    sns.distplot(Car_distances, bins=bins_number,color="b")
    plt.title("Radial distance to ego vehicle", fontsize=title_size)
    plt.xticks(range(0,40,5),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("Meters  [m]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Car ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_car_distance.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


    sns.distplot(Pedestrain_distances, bins=bins_number,color="orange")
    # plt.title("Radial distance to ego vehicle", fontsize=title_size)
    plt.xticks(range(0,40,5),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("Meters  [m]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Pedestrain ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_pedestrian_distance.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    sns.distplot(Bus_distances, bins=bins_number,color="g")
    # plt.title("Radial distance to ego vehicle", fontsize=title_size)
    plt.xticks(range(0,40,5),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("Meters  [m]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Bus ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_bus_distance.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


    sns.distplot(Rider_distances, bins=bins_number)
    # plt.title("Radial distance to ego vehicle", fontsize=title_size)
    plt.xticks(range(0,40,5),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Meters  [m]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Rider ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_rider_distance.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




    #degree distribution
    sns.distplot(Car_degree, bins=bins_number, color="b")
    plt.title("Azimuth angle", fontsize=title_size)
    plt.xticks( np.arange(-3.5,4,1),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("degree [Rad]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Car ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_car_degree.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    sns.distplot(Pedestrain_degree, bins=bins_number,color="orange")
    # plt.title("Azimuth angle", fontsize=title_size)
    plt.xticks(np.arange(-3.5,4,1),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("degree [Rad]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Pedestrain ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_pedestrian_degree.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    sns.distplot(Bus_degree, bins=bins_number,color="g")
    # plt.title("Azimuth angle", fontsize=title_size)
    plt.xticks(np.arange(-3.5,4,1), fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.xlabel("degree [Rad]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Bus ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_bus_degree.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    sns.distplot(Rider_degree,bins=bins_number)
    # plt.title("Azimuth angle", fontsize=title_size)
    plt.xticks( fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Degree [Rad]", fontsize=ylabel_size)
    plt.ylabel("Probability density - Rider ", fontsize=ylabel_size)
    save_path = os.path.join(OUTPUT_DIR, "output", "figures", "distribution_rider_degree.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("done")
if __name__ == '__main__':
    # plot_class_num_bar(DATA_DIR +"/annotation_info.pth")
    plot_position_distribution(DATA_DIR + "/annotation")