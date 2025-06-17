package com.qualcomm.qti.psnpedemo.post;

import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PostSegmentDeeplabv3 {

    private Gson gson = new Gson();
    private String savePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("save_result/deeplabv3").getAbsolutePath();

    public boolean postProcessResult(List<Map<String,float []>> outputs, List<String> Id) {
        try {
            String[] outputNames = PSNPEManager.getOutputTensorNames();
            for(int i=0; i<outputs.size(); i++)
            {
                Map<String, float []> outputMap = outputs.get(i);
                float[] outputData = outputMap.get(outputNames[0]);

                JSONObject img_result = new JSONObject();
                String img_name_ext=Id.get(i);
                img_result.put("id", img_name_ext);
                if (outputData == null) {
                    Log.e("TAG", "output data is null");
                    return false;
                }
                JSONArray img_rows = new JSONArray();
                for (int j = 0; j < 513; j++) {
                    JSONArray img_cols = new JSONArray();
                    for (int k = 0; k < 513; k++) {
                        img_cols.put(outputData[k+513*j]);
                    }
                    img_rows.put(img_cols);
                }
                img_result.put("predict", img_rows);
                Log.d("result",savePath);
                String seg_result_file =savePath + "/" + img_name_ext + ".json";
                Log.d("path", seg_result_file);
                File seg_file = new File(seg_result_file);
                FileOutputStream fos = new FileOutputStream(seg_file);
                fos.write(img_result.toString().getBytes());
                fos.close();
            }
            return true;
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
    }

    public boolean postProcessResult(ArrayList<File> bulkImage) {
        try {
            int imageNum = bulkImage.size();
            int[] inputDims = PSNPEManager.getInputDimensions();
            int imgDimension = inputDims[1];
            int NUM_CLASSES = 21;
            //add each result of image to a List object
            List<float []> results = new ArrayList<>();
            for(int i=0; i<imageNum; i++)
            {
                Map<String, float []> outputMap = PSNPEManager.getOutputSync(i);
                String[] outputNames = PSNPEManager.getOutputTensorNames();
                float[] outputData = outputMap.get(outputNames[0]);

                JSONObject img_result = new JSONObject();
                String img_name_ext=bulkImage.get(i).getName().split(".jpg")[0];
                img_result.put("id", img_name_ext);
                if (outputData == null) {
                    Log.e("TAG", "output data is null");
                    return false;
                }
                JSONArray img_rows = new JSONArray();
                for (int j = 0; j < 513; j++) {
                    JSONArray img_cols = new JSONArray();
                    for (int k = 0; k < 513; k++) {
                        img_cols.put(outputData[k+513*j]);
                    }
                    img_rows.put(img_cols);
                }
                img_result.put("predict", img_rows);
                Log.d("result",savePath);
                String seg_result_file =savePath + "/" + img_name_ext + ".json";
                Log.d("path", seg_result_file);
                File seg_file = new File(seg_result_file);
                FileOutputStream fos = new FileOutputStream(seg_file);
                fos.write(img_result.toString().getBytes());
                fos.close();

//                int [][]segmentationResult = new int[imgDimension][imgDimension];
//                for (int j = 0; j < imgDimension; j++){
//                    for (int k = 0; k < imgDimension; k++){
//                        segmentationResult[j][k] = (int)batchOutput[j * imageNum + k];
//                    }
//                }
//                File file = new File(savePath, bulkImage.get(i).getName().replace(".jpg", ".json"));
//                FileOutputStream fos = new FileOutputStream(file);
//                Result result = new Result(bulkImage.get(i).getName().replace(".jpg", ""),
//                        segmentationResult);
//                Result[] array = {result};
//                fos.write(gson.toJson(array).getBytes());
//                fos.close();
//
//                results.add(batchOutput);
//                Map<String, float []> outputMap = PSNPEManager.getOutputSync(i);
//                for (Map.Entry<String, float[]> output : outputMap.entrySet()) {
//                    String outputLayerName = output.getKey();
//                    float[] outputData = output.getValue();
//                    JSONObject img_result = new JSONObject();
//                    String img_name_ext=bulkImage.get(i).getName().split(".jpg")[0];
//                    img_result.put("id", img_name_ext);
//                    if (outputData == null) {
//                        Log.e("TAG", "output data is null");
//                        return false;
//                    }
//                    JSONArray img_rows = new JSONArray();
//                    for (int j = 0; j < 513; j++) {
//                        JSONArray img_cols = new JSONArray();
//                        for (int k = 0; k < 513; k++) {
//                            img_cols.put(outputData[k+513*j]);
//                        }
//                        img_rows.put(img_cols);
//                    }
//                    img_result.put("predict", img_rows);
//                    Log.d("result",savePath);
//                    String seg_result_file =savePath + "/" + img_name_ext + ".json";
//                    Log.d("path", seg_result_file);
//                    File seg_file = new File(seg_result_file);
//                    FileOutputStream fos = new FileOutputStream(seg_file);
//                    fos.write(img_result.toString().getBytes());
//                    fos.close();
//                }
            }
            return true;
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
    }

    static class Result {
        private String id;
        private int [][]predict;

        public Result(String id, int[][] predict) {
            this.id = id;
            this.predict = predict;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public int[][] getPredict() {
            return predict;
        }

        public void setPredict(int[][] predict) {
            this.predict = predict;
        }
    }

}
