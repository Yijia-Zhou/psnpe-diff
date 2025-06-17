package com.qualcomm.qti.psnpedemo.post;

import android.graphics.RectF;
import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.utils.Util;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class PostDetectionYolov3 {
    private String TAG = "PostDetectionYolov3";
    private final int[] mOutWidth = new int[]{13, 26, 52};
    private final int NUM_BOXES_PER_BLOCK = 3;
    private float mObjThresh = 0.6f;
    private final int [] mAnchors = new int[]{
            10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    };
    int [][] mMasks = new int[][]{{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
    private float mNosThresh = 0.5f;
    private final int labels_size = 80;
    private int inputSize = 416;
    private final List<Result> resultList = new ArrayList<>();
    //yolo3模型精度用coco api进行评估，coco api 的 category_id 的范围是 0 - 91，yolo3模型是 0 到 80
    //因此需要将yolo3推断的 category_id 结果映射到 coco api对应的 category_id 值上
    //此处为 SSD模型的label中的问号位置与yolo3模型的label位置建立的映射数组
    private final int []size = {
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11};

    public boolean postProcessResult(List<Map<String,float []>> outputs, List<String> Id){
        Log.d(TAG, "start into detection post process!");
//        int imageNum = bulkImage.size();
//        List<Map<String,float []>> outputs = new ArrayList<>(imageNum);
//
//        Map<String, float []> outputMap = null;
//        String[] outputNames =null;
//        for(int i=0; i< imageNum; i++) {
//            /* output:
//             * <outputBuffer1><outputBuffer2>...<imageBulkSize/batchSize>
//             * split output and handle one by one.
//             */
//            int batchCount = i % batchSize;
//            if(batchCount ==0){
//                outputMap = PSNPEManager.getOutputSync(i/batchSize);
//                outputNames = PSNPEManager.getOutputTensorNames();
//            }
//
//            for (String outputName : outputNames) {
//                float[] batchOutput = outputMap.get(outputName);
//                int outputSize = batchOutput.length / batchSize;
//                float[] output = new float[outputSize];
//                System.arraycopy(batchOutput, outputSize * batchCount, output, 0, outputSize);
//                if (outputs.size() < i + 1) {
//                    Map<String, float[]> tmp = new HashMap<>();
//                    outputs.add(i, tmp);
//                }
//
//                outputs.get(i).put(outputName, output);
//            }
//        }
//
//        List<String> Id=new ArrayList<>();
//        for(int j = 0;j<bulkImage.size();j++){
//            String imageName=bulkImage.get(j).getName();
//            String imageId=imageName.split("\\.")[0];
//            Id.add(j,imageId);
//        }
        try {
            getPredictionResult(outputs, Id);
        }catch (Exception e){
            Log.e("测试", "解析失败：" + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean postProcessResult(ArrayList<File> bulkImage, int batchSize){
        Log.d(TAG, "start into detection post process!");
        int imageNum = bulkImage.size();
        List<Map<String,float []>> outputs = new ArrayList<>(imageNum);

        Map<String, float []> outputMap = null;
        String[] outputNames =null;
        for(int i=0; i< imageNum; i++) {
            /* output:
             * <outputBuffer1><outputBuffer2>...<imageBulkSize/batchSize>
             * split output and handle one by one.
             */
            int batchCount = i % batchSize;
            if(batchCount ==0){
                outputMap = PSNPEManager.getOutputSync(i/batchSize);
                outputNames = PSNPEManager.getOutputTensorNames();
            }

            for (String outputName : outputNames) {
                float[] batchOutput = outputMap.get(outputName);
                int outputSize = batchOutput.length / batchSize;
                float[] output = new float[outputSize];
                System.arraycopy(batchOutput, outputSize * batchCount, output, 0, outputSize);
                if (outputs.size() < i + 1) {
                    Map<String, float[]> tmp = new HashMap<>();
                    outputs.add(i, tmp);
                }

                outputs.get(i).put(outputName, output);
            }
        }

        List<String> Id=new ArrayList<>();
        for(int j = 0;j<bulkImage.size();j++){
            String imageName=bulkImage.get(j).getName();
            String imageId=imageName.split("\\.")[0];
            Id.add(j,imageId);
        }
        try {
            getPredictionResult(outputs, Id);
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public void getPredictionResult(List<Map<String, float[]>> outputs, List<String> imageIds) throws Exception{
        String imagesFile= Util.getDatafromFile(DataSets.DETECTION_IMAGE_LABEL_PATH);
        JSONObject json_object= new JSONObject(imagesFile);
        for (int j = 0; j < outputs.size(); j++){
            Map<String, float []> output = outputs.get(j);
            float[]  feature_map_52 = output.get("yolov3/yolov3_head/Conv_22/BiasAdd:0");
            float[]  feature_map_26 = output.get("yolov3/yolov3_head/Conv_14/BiasAdd:0");
            float[]  feature_map_13 = output.get("yolov3/yolov3_head/Conv_6/BiasAdd:0");

            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, feature_map_13);// 1,13,13,(80 + 5) * 3
            outputMap.put(1, feature_map_26);// 1,26,26,(80 + 5) * 3
            outputMap.put(2, feature_map_52);// 1,52,52,(80 + 5) * 3

            ArrayList<Recognition> detections = new ArrayList<>();
            for (int i = 0; i < mOutWidth.length; i++){
                int gridWidth = mOutWidth[i];
                float[] out = (float[]) outputMap.get(i);
                for (int y = 0; y < gridWidth; ++y){
                    for (int x = 0; x < gridWidth; ++x){
                        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b){
                            final int offset =
                                    (gridWidth * (NUM_BOXES_PER_BLOCK * (labels_size + 5))) * y
                                            + (NUM_BOXES_PER_BLOCK * (labels_size + 5)) * x
                                            + (labels_size + 5) * b;
                            final float confidence = expit(out[(y * gridWidth + x) * 255 + (b * (labels_size + 5)) + 4]);
                            int detectedClass = -1;
                            float maxClass = 0;
                            final float[] classess = new float[labels_size];
                            for (int _c = 0; _c < labels_size; ++_c){
                                classess[_c] = out[(y * gridWidth + x) * 255 + (labels_size + 5) * b + 5 + _c];
                            }
                            softMax(classess);
                            for (int _c = 0; _c < labels_size; ++_c){
                                if (classess[_c] > maxClass){
                                    detectedClass = _c;
                                    maxClass = classess[_c];
                                }
                            }
                            final float confidenceInClass = maxClass * confidence;
                            if (confidenceInClass > mObjThresh){
                                final float xPos = (x + expit(out[(y * gridWidth + x) * 255 + (labels_size + 5) * b])) * (inputSize * 1.0f / gridWidth);
                                final float yPos = (y + expit(out[(y * gridWidth + x) * 255 + (labels_size + 5) * b + 1])) * (inputSize * 1.0f / gridWidth);

                                final float w = (float) (Math.exp(out[(y * gridWidth + x) * 255 + (labels_size + 5) * b + 2]) * mAnchors[2 * mMasks[i][b]]);
                                final float h = (float) (Math.exp(out[(y * gridWidth + x) * 255 + (labels_size + 5) * b + 3]) * mAnchors[2 * mMasks[i][b] + 1]);

                                final RectF rect =
                                        new RectF(
                                                Math.max(0, xPos - w / 2),
                                                Math.max(0, yPos - h / 2),
                                                Math.min(inputSize - 1, xPos + w / 2),
                                                Math.min(inputSize - 1, yPos + h / 2));
                                detections.add(new Recognition("" + offset, "" + detectedClass,
                                        confidenceInClass, rect, detectedClass));
                            }
                        }
                    }
                }
            }
            ArrayList<Recognition> recognitions = nms(detections);

            String id = imageIds.get(j);
            JSONArray tmp = json_object.getJSONArray(id);
            int realHeight = Integer.parseInt(tmp.get(1).toString());
            int realWidth = Integer.parseInt(tmp.get(0).toString());
            Log.e("测试", "coco: " + id + " " + realWidth + " " + realHeight);
            int img_id = Integer.parseInt(imageIds.get(j));
            for (Recognition recognition : recognitions) {
                float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
                if (recognition.getConfidence() < MINIMUM_CONFIDENCE_TF_OD_API) {
                    continue;
                }
                resultList.add(new Result(
                        img_id,
                        recognition.getDetectClass() + size[recognition.getDetectClass()],
                        new double[]{
                                recognition.getLocation().left * realWidth / inputSize,
                                recognition.getLocation().top * realHeight / inputSize,
                                (recognition.getLocation().right - recognition.getLocation().left) * realWidth / inputSize,
                                (recognition.getLocation().bottom - recognition.getLocation().top) * realHeight / inputSize
                        },
                        recognition.getConfidence()
                ));
            }
        }
    }

    public void saveData() {
        try {
            File file = new File(DataSets.MODEL_SAVE_DIR, "YoloV3_result.json");
            FileOutputStream fos = new FileOutputStream(file);
            Gson gson = new Gson();
            fos.write(gson.toJson(resultList).getBytes());
            fos.close();
            resultList.clear();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private float expit(final float x){
        return (float) (1. / (1. + Math.exp(-x)));
    }

    protected void softMax(final float[] vals){
        float max = Float.NEGATIVE_INFINITY;
        for(final float val:vals){
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for(int i = 0; i < vals.length; ++i){
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i){
            vals[i] = vals[i] / sum;
        }
    }

    protected ArrayList<Recognition> nms(ArrayList<Recognition> list){
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < labels_size; k++){
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            10,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(Recognition o1, Recognition o2) {
                                    return Float.compare(o2.getConfidence(), o1.getConfidence());
                                }
                            });
            for (int i = 0; i < list.size(); ++i){
                if (list.get(i).getDetectClass() == k){
                    pq.add(list.get(i));
                }
            }
            Log.d(TAG, "class[" + k + "] pq size:" + pq.size());

            while (pq.size() > 0){
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);

                Log.d(TAG, "before nms pq size:" + pq.size());

                pq.clear();

                for (int j = 1; j < detections.length; j++){
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNosThresh){
                        pq.add(detection);
                    }
                }
                Log.d(TAG, "after nms pq size" + pq.size());
            }
        }
        return nmsList;
    }

    protected float box_iou(RectF a, RectF b){
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b){
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0){
            return 0;
        }
        return w * h;
    }

    protected float box_union(RectF a, RectF b){
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    private float overlap(float x1, float w1, float x2, float w2){
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }

    static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        private int detectClass;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location, final int detectClass) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.detectClass = detectClass;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        public int getDetectClass() {
            return detectClass;
        }

        public void setDetectClass(int detectClass) {
            this.detectClass = detectClass;
        }

        @Override
        public String toString() {
            String resultString = "";

            if (id != null){
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            resultString += detectClass + " ";
            return resultString.trim();
        }
    }

    static class Result {
        private int image_id;
        private int category_id;
        private double[] bbox;
        private float score;

        public Result(int image_id, int category_id, double[] bbox, float score) {
            this.image_id = image_id;
            this.category_id = category_id;
            this.bbox = bbox;
            this.score = score;
        }

        public int getImage_id() {
            return image_id;
        }

        public void setImage_id(int image_id) {
            this.image_id = image_id;
        }

        public int getCategory_id() {
            return category_id;
        }

        public void setCategory_id(int category_id) {
            this.category_id = category_id;
        }

        public double[] getBbox() {
            return bbox;
        }

        public void setBbox(double[] bbox) {
            this.bbox = bbox;
        }

        public float getScore() {
            return score;
        }

        public void setScore(float score) {
            this.score = score;
        }
    }
}
