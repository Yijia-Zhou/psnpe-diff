///*
// * Copyright (c) 2022 Qualcomm Technologies, Inc.
// * All Rights Reserved.
// * Confidential and Proprietary - Qualcomm Technologies, Inc.
// */
//package com.qualcomm.qti.psnpedemo.processor;
//import android.util.Log;
//import com.qualcomm.qti.psnpedemo.utils.Util;
//
//import java.io.File;
//import java.util.HashMap;
//
//public class MobilebertPreProcessor extends PreProcessor {
//    private static String TAG = MobilebertPreProcessor.class.getSimpleName();
//    @Override
//    public HashMap<String, float[]> preProcessData(File data) {
//        File[] fileList = data.listFiles();
//        assert fileList != null;
//        File InputIdsRaw = null;
//        File InputMaskRaw= null;
//        File SegmentIdsRaw= null;
//        for (File file : fileList) {
//            String dataName = file.getName();
//            if (dataName.toLowerCase().contains("input_ids.raw")) {
//                InputIdsRaw = file;
//            }
//            if (dataName.toLowerCase().contains("input_mask.raw")) {
//                InputMaskRaw = file;
//            }
//            if (dataName.toLowerCase().contains("segment_ids.raw")) {
//                SegmentIdsRaw = file;
//            }
//        }
//        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
//        //bert model accept 3 inputs
//        //bert/embeddings/ExpandDims:0:=squad11_75_question_float32/1000000000/input_ids.raw
//        //input_mask:0:=squad11_75_question_float32/1000000000/input_mask.raw
//        //segment_ids:0:=/squad11_75_question_float32/1000000000/segment_ids.raw
//        String dataName = InputIdsRaw.getName();
//        if(dataName.toLowerCase().contains(".raw")){
//            outputMap.put("bert/embeddings/ExpandDims:0",preProcessRaw(InputIdsRaw));
//            outputMap.put("input_mask:0",preProcessRaw(InputMaskRaw));
//            outputMap.put("segment_ids:0",preProcessRaw(SegmentIdsRaw));
//            return outputMap;
//        }
//        else {
//            Log.e(TAG, "data format invalid, dataName: " + dataName);
//            return null;
//        }
//    }
//    private float [] preProcessRaw(File data){
//        float[] floatArray = Util.readFloatArrayFromFile(data);
//        return floatArray;
//    }
//}
/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.post.MobileBertUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class MobilebertPreProcessor extends PreProcessor {
    private static String TAG = MobilebertPreProcessor.class.getSimpleName();
    private List<MobileBertUtil.Feature> featuresList = new ArrayList<>();
    private MobileBertUtil mobileBertUtil;
    private MobileBertUtil.Result mobileBertResult;

    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        return null;
    }

    public void preProcessDataAndLoadData(int MAX_POSITION){
        featuresList.clear();
        mobileBertResult = new MobileBertUtil.Result();
        List<MobileBertUtil.Result.Data> resultData = new ArrayList<>();
        mobileBertResult.setData(resultData);
        try {
            mobileBertUtil = new MobileBertUtil(BenchmarkApplication.getExternalDirPath() + "/datasets/nlp/vocab.txt");

            Gson gson = new Gson();
            File file = new File(BenchmarkApplication.getExternalDirPath() + "/datasets/nlp/dev-v1.1.json");
            FileInputStream inputStream = new FileInputStream(file);
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            StringBuilder stringBuilder = new StringBuilder();
            String str;
            while ((str = bufferedReader.readLine()) != null) {
                stringBuilder.append(str);
            }
            MobileBertUtil.Question allData = gson.fromJson(stringBuilder.toString(), MobileBertUtil.Question.class);

            int curPosition = 0;
            for (MobileBertUtil.Question.Item item : allData.getData()) {
                if (curPosition == MAX_POSITION) {
                    break;
                }
                for (MobileBertUtil.Question.Paragraphs paragraphs : item.getParagraphs()) {
                    if (curPosition == MAX_POSITION) {
                        break;
                    }
                    for (MobileBertUtil.Question.Qas qas : paragraphs.getQas()) {
                        String content = paragraphs.getContext();
                        MobileBertUtil.Feature feature = mobileBertUtil.getFeature(qas.getQuestion(), content);
                        HashMap<String, float[]> inputData = new HashMap<>();
                        float[] input_expandDims = new float[384];
                        float[] input_mask = new float[384];
                        float[] segment_ids = new float[384];
                        for (int j = 0; j < MobileBertUtil.MAX_SEQ_LEN; j++) {
                            input_expandDims[j] = feature.inputIds[j];
                            input_mask[j] = feature.inputMask[j];
                            segment_ids[j] = feature.segmentIds[j];
                        }
                        inputData.put("bert/embeddings/ExpandDims:0", input_expandDims);
                        inputData.put("input_mask:0", input_mask);
                        inputData.put("segment_ids:0", segment_ids);
                        if (!PSNPEManager.loadBatchData(inputData, curPosition, 1)) {
                            PSNPEManager.release();
                            Log.e("测试", "Load batch data Failed, batch index: " +  curPosition) ;
                        }
                        featuresList.add(feature);

                        MobileBertUtil.Result.Data data = new MobileBertUtil.Result.Data();
                        String[] answers = new String[qas.getAnswers().size()];
                        for (int i = 0; i < qas.getAnswers().size(); i++) {
                            answers[i] = qas.getAnswers().get(i).getText();
                        }
                        data.setAnswers(answers);
                        resultData.add(data);

                        curPosition++;
                        if (curPosition == MAX_POSITION) {
                            break;
                        }
                    }
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public List<MobileBertUtil.Feature> getFeaturesList() {
        return featuresList;
    }

    public void setFeaturesList(List<MobileBertUtil.Feature> featuresList) {
        this.featuresList = featuresList;
    }

    public MobileBertUtil getMobileBertUtil() {
        return mobileBertUtil;
    }

    public void setMobileBertUtil(MobileBertUtil mobileBertUtil) {
        this.mobileBertUtil = mobileBertUtil;
    }

    public MobileBertUtil.Result getMobileBertResult() {
        return mobileBertResult;
    }

    public void setMobileBertResult(MobileBertUtil.Result mobileBertResult) {
        this.mobileBertResult = mobileBertResult;
    }
}
