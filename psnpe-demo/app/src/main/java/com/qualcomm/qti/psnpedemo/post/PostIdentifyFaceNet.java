package com.qualcomm.qti.psnpedemo.post;

import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PostIdentifyFaceNet {
    private final String TAG = "PostIdentifyFaceNet";
    private HashMap<String, float[]> datas = new HashMap<>();
    /**
     * 从lfw数据集中，随机挑选 （300组（同人不同图片两张） + 300组（不同人的不同图片各一张））* 2 = 2400 组
     * 文本每行存放一组（2张）图片名称 以及图片是否为同一个人的布尔值，用以‘\t'分隔
     * 例如：
     * 小红_1.jpg 小红_2.jpg True
     * .....
     * 小红_1.jpg 小彪_1.jpg False
     * .....
     */
    private String dataSetsPath = BenchmarkApplication.getExternalDirPath() + "/datasets/lfw/lfw_deal.txt";

    public boolean postProcessResult(ArrayList<File> bulkImage, int batchSize){
        int imgNum = bulkImage.size();
        float[] batchOutput = null;
        for (int i = 0; i < imgNum; ++i) {
            /* output:
             * <image1><image2>...<imageBulkSize>
             * split output and handle one by one.
             */
            int batchCount = i % batchSize;
            if(batchCount == 0){
                Map<String, float []> outputMap = PSNPEManager.getOutputSync(i/batchSize);
                String[] outputNames = PSNPEManager.getOutputTensorNames();
                batchOutput = outputMap.get(outputNames[0]);
                if(batchOutput == null) {
                    Log.e(TAG, "batchOutput data is null");
                    return false;
                }
            }
            int outputSize = batchOutput.length/batchSize;
            Log.e("测试", "outputSize:" + outputSize);
            float[] output1 = new float[outputSize];
            System.arraycopy(batchOutput, outputSize*batchCount, output1, 0, outputSize);
            datas.put(bulkImage.get(i).getName(), output1);
        }
        return saveData();
    }

    private boolean saveData(){
        try {
            OutData outData = new OutData();
            outData.embeddings1 = new ArrayList<>();
            outData.embeddings2 = new ArrayList<>();
            List<Boolean> actualIsSame = new ArrayList<>();
            outData.actual_issame = actualIsSame;

            File file = new File(dataSetsPath);
            FileInputStream inputStream = new FileInputStream(file);
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String[] pair;
            String str;
            while ((str = bufferedReader.readLine()) != null){
                pair = str.split("\t");
                outData.embeddings1.add(datas.get(pair[0]));
                outData.embeddings2.add(datas.get(pair[1]));
                actualIsSame.add("True".equals(pair[2]));
                if (datas.get(pair[0]) == null || datas.get(pair[1])== null){
                    Log.e("测试", "null");
                }
            }
            File saveFile = new File(DataSets.MODEL_SAVE_DIR, "face_net_result.json");
            Gson gson = new Gson();
            OutputStream outputStream = new FileOutputStream(saveFile);
            outputStream.write(gson.toJson(outData).getBytes());
            outputStream.close();
            bufferedReader.close();
            datas.clear();
            return true;
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
    }

    static class OutData{
        int nrof_folds = 2;
        int distance_metric = 1;
        boolean subtract_mean = true;
        List<float[]> embeddings1;
        List<float[]> embeddings2;
        List<Boolean> actual_issame;
    }
}
