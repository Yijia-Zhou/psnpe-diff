package com.qualcomm.qti.psnpedemo.post;

import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.utils.ComputeUtil;
import com.qualcomm.qti.psnpedemo.utils.Util;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class PostDetectionMobileNetV1SSD {
    private final String TAG = "PostDetectionMobileNetV1SSD";
    public float confidenceThreshold = 0.0f;
    private final List<Result>resultList = new ArrayList<>();

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

    public boolean getPredictionResult(List<Map<String, float[]>> outputs, List<String> imageIds) throws JSONException {
        String imagesFile= Util.getDatafromFile(DataSets.DETECTION_IMAGE_LABEL_PATH);
        JSONObject json_object= new JSONObject(imagesFile);
        if(imageIds == null) {
            Log.d(TAG, "results or images ids null");
            return false;
        }

        for(int i = 0; i < outputs.size(); i++) {
            Map<String, float []> output = outputs.get(i);

            float[] bboxArray = null;
            float[] scoreArray = null;
            float[] classArrayTmp = null;

            for(String key: output.keySet()) {
                if(key.contains("boxes") && bboxArray == null){
                    bboxArray =output.get(key);
                }
                else if(key.contains("scores") && scoreArray == null){
                    scoreArray = output.get(key);
                }
                else if(key.contains("classes") && classArrayTmp == null){
                    classArrayTmp =output.get(key);
                }
            }
            if(bboxArray == null || scoreArray == null || classArrayTmp == null){
                Log.e(TAG,"can't find all outputs layer");
                return false;
            }

            int[] classArray = ComputeUtil.mathFloor(classArrayTmp);
            for(int k = 0; k < Objects.requireNonNull(classArray).length; k++) {
                classArray[k] = classArray[k] + 1;
            }
            for(int j = 0; j < scoreArray.length; j++) {
                if(scoreArray[j] == 0) {
                    break;
                }
                float topConfidence = scoreArray[j];
                if(topConfidence < confidenceThreshold) {
                    continue;
                }

                String img_id = "";
                img_id = imageIds.get(i);
                JSONArray tmp = json_object.getJSONArray(img_id);
                img_id = img_id.replaceAll("^(0+)", "");
                int height = Integer.parseInt(tmp.get(1).toString());
                int width = Integer.parseInt(tmp.get(0).toString());

                float left = (bboxArray[(j * 4) + 1]) * 300;
                float top = (bboxArray[(j * 4)]) * 300;
                float right = (bboxArray[(j * 4) + 3]) * 300;
                float bottom = (bboxArray[(j * 4) + 2]) * 300;

                resultList.add(new Result(
                        Integer.parseInt(img_id),
                        classArray[j],
                        new double[]{
                                left * width / 300.0,
                                top * height / 300.0,
                                (right - left) * width / 300.0,
                                (bottom - top) * height / 300.0
                        },
                        topConfidence));
            }
        }
        return true;

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

    public void saveData() {
        try {
            File file = new File(DataSets.MODEL_SAVE_DIR, "MobileNetV1_SSD.json");
            FileOutputStream fos = new FileOutputStream(file);
            Gson gson = new Gson();
            fos.write(gson.toJson(resultList).getBytes());
            fos.close();
            resultList.clear();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
