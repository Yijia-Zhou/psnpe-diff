/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
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

public class MobileNetYOLO3PreProcessor extends PreProcessor {
    private static String TAG = MobileNetYOLO3PreProcessor.class.getSimpleName();
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
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

        double [] meanRGB = {0.0d, 0.0d, 0.0d};
        float [] result = Util.imagePreprocess(data, tensorShapes[1], meanRGB, 255.0, false, 416);

        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/yolov3_input_list.txt", data.getName());

        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        String[] key = PSNPEManager.getInputTensorNames();
        outputMap.put(key[0],result);

        return outputMap;
    }
}
