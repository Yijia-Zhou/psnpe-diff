package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.post.MobileBertUtil;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MobileBertPostProcessor extends PostProcessor {
    public MobileBertPostProcessor(int imageNumber) {
        super(imageNumber);
    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }

    @Override
    public boolean postProcessResult(ArrayList<File> bulkImage) {
        return false;
    }

    @Override
    public void setResult(Result result) {

    }

    public void postProcessResult(MobilebertPreProcessor preProcessor, int MAX_POSITION) throws Exception{
        MobileBertUtil mobileBertUtil = preProcessor.getMobileBertUtil();
        List<MobileBertUtil.Feature> featuresList = preProcessor.getFeaturesList();
        MobileBertUtil.Result mobileBertResult = preProcessor.getMobileBertResult();
        for (int i = 0; i < MAX_POSITION; i++) {
            Map<String, float[]> outputMap = PSNPEManager.getOutputSync(i);
            if (outputMap == null || outputMap.size() == 0) {
                Log.e(TAG, "postProcessResult error: outputMap is null");
                return;
            }
            float[] batchOutput = outputMap.get(PSNPEManager.getOutputTensorNames()[0]);
            if (null == batchOutput) {
                Log.e(TAG, "postProcessResult error: output is null");
            }
            float[] startLogits = new float[384];
            float[] endLogits = new float[384];
            System.arraycopy(batchOutput, 0, startLogits, 0, batchOutput.length/2);
            System.arraycopy(batchOutput, batchOutput.length/2, endLogits, 0,batchOutput.length/2);
            List<MobileBertUtil.QaAnswer> results = mobileBertUtil.getBestAnswers(startLogits, endLogits, featuresList.get(i));
            mobileBertResult.getData().get(i).setResult((results.size() > 0 ?  results.get(0).text : ""));
        }
        String dir = BenchmarkApplication.getExternalDirPath();
        File saveFile = new File(dir, "snpe_MobileBert_result.json");
        OutputStream outputStream = new FileOutputStream(saveFile);
        outputStream.write(new Gson().toJson(mobileBertResult).getBytes());
        outputStream.close();
        Log.e("测试", "保存结果！");
    }

    public void postRawResult(MobilebertPreProcessor preProcessor, int MAX_POSITION){
        MobileBertUtil mobileBertUtil = preProcessor.getMobileBertUtil();
        List<MobileBertUtil.Feature> featuresList = preProcessor.getFeaturesList();
        MobileBertUtil.Result mobileBertResult = preProcessor.getMobileBertResult();

        for (int i = 0; i < MAX_POSITION; i++){
            String filePath = BenchmarkApplication.getExternalDirPath() + "/output_android/Result_" + i + "/transpose_0.raw";
            float[] batchOutput = Util.readFloatArrayFromFile(filePath);
            Log.e("测试",  i + " postRawResult 输出的大小:" + batchOutput.length);
            float[] startLogits = new float[384];
            float[] endLogits = new float[384];
            System.arraycopy(batchOutput, 0, startLogits, 0, batchOutput.length/2);
            System.arraycopy(batchOutput, batchOutput.length/2, endLogits, 0,batchOutput.length/2);
            List<MobileBertUtil.QaAnswer> results = mobileBertUtil.getBestAnswers(startLogits, endLogits, featuresList.get(i));
            Log.e("测试", "结果；" + (results.size() > 0 ?  results.get(0).text : "解析失败"));
            mobileBertResult.getData().get(i).setResult((results.size() > 0 ?  results.get(0).text : ""));
        }
        try {
            String dir = BenchmarkApplication.getExternalDirPath();
            File saveFile = new File(dir, "snpe_MobileBert_result_raw.json");
            OutputStream outputStream = new FileOutputStream(saveFile);
            outputStream.write(new Gson().toJson(mobileBertResult).getBytes());
            outputStream.close();
            Log.e("测试", "保存结果！");
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
