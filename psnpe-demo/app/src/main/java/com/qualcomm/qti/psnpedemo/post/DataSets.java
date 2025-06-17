package com.qualcomm.qti.psnpedemo.post;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

public class DataSets {

    public static final String ROOT_PATH = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();

    /**
     * 模型结果输出文件路径
     */
    public static final String MODEL_SAVE_DIR = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("save_result").getAbsolutePath();

    /**
     *  coco_images.json：存放所有测试图片的map的集合，格式大体为：
     *  {"image-name":['width','height'],"image-name":['width','height'], ...}
     *  {"000000000139": ["640", "426"], "000000000285": ["586", "640"], ....}
     */
    public static final String DETECTION_IMAGE_LABEL_PATH = ROOT_PATH + "/datasets/" + "coco" + "/coco_images.json";
}
