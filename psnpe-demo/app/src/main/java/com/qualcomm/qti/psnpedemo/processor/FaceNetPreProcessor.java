/*
 * Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.HashMap;

public class FaceNetPreProcessor extends PreProcessor{
    private static String TAG = FaceNetPreProcessor.class.getSimpleName();
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        String dataName = data.getName();
        if(dataName.toLowerCase().contains(".raw")){
            HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
            String[] key = PSNPEManager.getInputTensorNames();
            outputMap.put(key[0],preProcessRaw(data));
            return outputMap;
        }
        else {
            HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
            String[] key = PSNPEManager.getInputTensorNames();
            outputMap.put(key[0], preData(data));
            return outputMap;
        }
    }

    private float [] preProcessRaw(File data){
        int[] dimensions = PSNPEManager.getInputDimensions();
        int dataSize = 1 * dimensions[1] * dimensions[2] * dimensions[3];
        float[] floatArray = Util.readFloatArrayFromFile(data);
        if(floatArray.length != dataSize){
            Log.e(TAG, String.format("Wrong input data size: %d. Expect %d.", floatArray.length, dataSize));
            return null;
        }
        return floatArray;
    }

    private float[] preData(File data){
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg") || dataName.contains("png"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(tensorShapes.length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR");
            return null;
        }

        double [] meanRGB = {128, 128, 128};
        float [] result = Util.imagePreprocess(data, tensorShapes[1], meanRGB, 128, false, 160);

        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/facenet_input_list.txt", data.getName());
        return result;
    }
}
