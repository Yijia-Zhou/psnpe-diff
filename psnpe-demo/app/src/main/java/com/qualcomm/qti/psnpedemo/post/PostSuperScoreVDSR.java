package com.qualcomm.qti.psnpedemo.post;

import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Map;

public class PostSuperScoreVDSR {
    private final String TAG = "PostSuperScoreVDSR";
    private int radio = 2;
    private String saveDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("save_result/vdsr/" + radio).getAbsolutePath();


    public boolean postProcessResult(ArrayList<File> bulkImage, int batchSize) {
        int imageNum = bulkImage.size();

        float [] groundTruth = null;
        float[] batchOutput = null;
        for(int i=0; i<imageNum; i++) {
            int batchCount = i % batchSize;
            if(batchCount == 0){
                Map<String, float []> outputMap =  PSNPEManager.getOutputSync(i/batchSize);
                if(outputMap.size() == 0){
                    Log.e(TAG, "postProcessResult error: outputMap is null");
                    return false;
                }
                batchOutput = outputMap.get(PSNPEManager.getOutputTensorNames()[0]);
                if(null == batchOutput){
                    Log.e(TAG, "postProcessResult error: output is null");
                    return false;
                }
            }
            int outputSize = batchOutput.length/batchSize;
            float [] output = new float[outputSize];
            System.arraycopy(batchOutput, outputSize*batchCount, output, 0, outputSize);

            try {
                File file = new File(saveDir, bulkImage.get(i).getName().replace(".jpg", ".json"));
                FileOutputStream fos = new FileOutputStream(file);
                Gson gson = new Gson();
                fos.write(gson.toJson(output).getBytes());
                fos.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
        return true;
    }


}
