/*
 * Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.SegmentationResult;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.post.PostSegmentDeeplabv3;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;
import android.util.Log;
import java.util.List;


public class SegmentationPostProcessor extends PostProcessor {
    private static String TAG = "SegmentationPostProcessor";
    private static final String[] conf = {"TP","FP","FN"};
    private  double GlobalAcc = 0.0;
    private  double MeanIOU = 0.0;
    private  double MeanAccuracy = 0.0;
    private  double MeanPrecision = 0.0;
    private  double MeanF1Score = 0.0;
    private int LABEL_NUM = 21;
    private int BASIC_METRICS_NUM = 3;
    private int GRAY_NUM = 255;
    private int FIXED_HEIGHT = 0;
    private int FIXED_WIDTH = 0;
    private String packagePath;
    private String groundTruthPath;

    private String modelName;

    public SegmentationPostProcessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber) {
        super(imageNumber);
        modelName = modelInfo.getModelName();
        String dataSetName = modelInfo.getDataSetName();
        this.packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        this.groundTruthPath = packagePath + "/datasets/" + dataSetName + "/SegmentationClass/";
    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        if (modelName.contains("deep")) {
            PostSegmentDeeplabv3 deeplabv3 = new PostSegmentDeeplabv3();
            int max = 100;
            int imageNum = inputImages.size();
            int executeTimes = (imageNum + max - 1) / max;
            int nowCount = 0;
            List<Map<String, float[]>> outputs = new ArrayList<>(imageNum);
            List<String> Id = new ArrayList<>();
            for (int i = 0; i < executeTimes; i++) {
                int handleSize = i == executeTimes - 1 ? imageNum - i * max : max;
                for (int j = 0; j < handleSize; j++) {
                    outputs.add(readOutput(nowCount));
                    String imageName = inputImages.get(nowCount).getName().split(".jpg")[0];
                    Id.add(imageName);
                    nowCount++;
                }
                deeplabv3.postProcessResult(outputs, Id);
                outputs.clear();
                Id.clear();
            }
            return true;
        }

        int imageNum = inputImages.size();
        int[] inputDims = PSNPEManager.getInputDimensions();
        int imgDimension=inputDims[1];
        if(imgDimension == 512) {
            FIXED_HEIGHT = imgDimension;
            FIXED_WIDTH = imgDimension;
        } else if (imgDimension == 513) {
            FIXED_HEIGHT = imgDimension;
            FIXED_WIDTH = imgDimension;
        }
        //add each result of image to a List object
        int max = 100;
        int executeTimes = (imageNum + max - 1)/max;
        int nowCount = 0;
        List<float []> results = new ArrayList<>();
        List<String> fileName = new ArrayList<>();
        String[] outputNames = PSNPEManager.getOutputTensorNames();
        for (int i = 0; i < executeTimes; i++){
            int handleSize = i == executeTimes - 1? imageNum - i * max : max;
            for (int j = 0; j < handleSize; j++){
                results.add(readOutput(nowCount).get(outputNames[0]));

                String bulkImagePath = (inputImages.get(nowCount)).getPath();
                File bulkImageFile =new File(bulkImagePath.trim());
                String bulkImageFilename = bulkImageFile.getName();
                String annoName=bulkImageFilename.substring(bulkImageFilename.lastIndexOf(".") );
                String annoImgFilename=bulkImageFilename.substring(0, bulkImageFilename.length()-annoName.length());
                final String annoImgPath =groundTruthPath+annoImgFilename+".png";
                fileName.add(annoImgPath);
                Log.e("测试", "nowCount:" + nowCount);
                nowCount++;
            }
            double[] result = calculateResults(inputImages,groundTruthPath, results, fileName);
            GlobalAcc += result[0];
            MeanIOU += result[1];
            MeanAccuracy += result[2];
            MeanPrecision += result[3];
            MeanF1Score += result[4];
            results.clear();
            fileName.clear();
        }
        GlobalAcc /= executeTimes;
        MeanIOU /= executeTimes;
        MeanAccuracy /= executeTimes;
        MeanPrecision /= executeTimes;
        MeanF1Score /= executeTimes;
//        int imageNum = inputImages.size();
//        int[] inputDims = PSNPEManager.getInputDimensions();
//        int imgDimension=inputDims[1];
//        if(imgDimension == 512) {
//            FIXED_HEIGHT = imgDimension;
//            FIXED_WIDTH = imgDimension;
//        } else if (imgDimension == 513) {
//            FIXED_HEIGHT = imgDimension;
//            FIXED_WIDTH = imgDimension;
//        }
//        //add each result of image to a List object
//        List<float []> results = new ArrayList<>();
//        String[] outputNames = PSNPEManager.getOutputTensorNames();
//        for (int i = 0; i < imageNum; i++) {
//            results.add(readOutput(i).get(outputNames[0]));
//        }
//        calculateResults(inputImages,groundTruthPath, results, imageNum);
        return false;
    }

    public double[] calculateResults(ArrayList<File> inputImages, String annoFolder, List<float []> results, List<String> fileName) {
        int[] labels = new int[LABEL_NUM];
        for(int i = 0; i < LABEL_NUM; i++) {
            labels[i] = i;
        }
        final List<int[][]> confMatrixs = new ArrayList<>();

        for(int imageIndex = 0; imageIndex < fileName.size(); imageIndex++)
        {
            final float[] result = results.get(imageIndex);
            int[][] ConfusionMat = performIOU(fileName.get(imageIndex), result);
            Log.i(TAG, "processing image on performIOU: " + fileName.get(imageIndex));
            if (ConfusionMat != null)
            {
                confMatrixs.add(ConfusionMat);
            }
        }
        int[][] confMatrix = new int[GRAY_NUM][BASIC_METRICS_NUM];
        for(int i = 0; i < confMatrix.length; i++) {
            for(int j = 0; j < confMatrix[0].length; j++) {
                confMatrix[i][j] = 0;
            }
        }
        for(int i = 0; i < confMatrixs.size(); i++){
            int[][] matrix = confMatrixs.get(i);
            for(int k = 0; k < matrix.length; k++) {
                for(int j = 0; j < matrix[0].length; j++) {
                    confMatrix[k][j] += matrix[k][j];
                }
            }
        }
        return MathUtils.calSegIndex(confMatrix, labels);
//        double[] result = MathUtils.calSegIndex(confMatrix, labels);
//        GlobalAcc = result[0];
//        MeanIOU = result[1];
//        MeanAccuracy = result[2];
//        MeanPrecision = result[3];
//        MeanF1Score = result[4];
    }

    public void calculateResults(ArrayList<File> inputImages, String annoFolder, List<float []> results, int imageNum) {
        int[] labels = new int[LABEL_NUM];
        for(int i = 0; i < LABEL_NUM; i++) {
            labels[i] = i;
        }
        final List<int[][]> confMatrixs = new ArrayList<>();

        for(int imageIndex = 0; imageIndex < imageNum; imageIndex++)
        {
            final float[] result = results.get(imageIndex);
            String bulkImagePath = (inputImages.get(imageIndex)).getPath();
            File bulkImageFile =new File(bulkImagePath.trim());
            String bulkImageFilename = bulkImageFile.getName();
            String annoName=bulkImageFilename.substring(bulkImageFilename.lastIndexOf(".") );
            String annoImgFilename=bulkImageFilename.substring(0, bulkImageFilename.length()-annoName.length());
            final String annoImgPath =groundTruthPath+annoImgFilename+".png";
            int[][] ConfusionMat = performIOU(annoImgPath, result);
            Log.i(TAG, "processing image on performIOU: " + annoImgPath);
            if (ConfusionMat != null)
            {
                confMatrixs.add(ConfusionMat);
            }
        }
        int[][] confMatrix = new int[GRAY_NUM][BASIC_METRICS_NUM];
        for(int i = 0; i < confMatrix.length; i++) {
            for(int j = 0; j < confMatrix[0].length; j++) {
                confMatrix[i][j] = 0;
            }
        }
        for(int i = 0; i < confMatrixs.size(); i++){
            int[][] matrix = confMatrixs.get(i);
            for(int k = 0; k < matrix.length; k++) {
                for(int j = 0; j < matrix[0].length; j++) {
                    confMatrix[k][j] += matrix[k][j];
                }
            }
        }
        double[] result = MathUtils.calSegIndex(confMatrix, labels);
        GlobalAcc = result[0];
        MeanIOU = result[1];
        MeanAccuracy = result[2];
        MeanPrecision = result[3];
        MeanF1Score = result[4];
    }

    private int[][] performIOU(String imgPath, float[] resultVec) {
        int height, width;
        int[][] annoImage = Util.readImageToPmode(imgPath);
        height = annoImage.length;
        width = annoImage[0].length;
        int[][] result = Util.getResizedResultImage(resultVec,height, width,FIXED_HEIGHT, FIXED_WIDTH);
        if(annoImage == null || result == null) {
            return null;
        }

        int labelMin = Math.min(MathUtils.minMatrix(annoImage), MathUtils.minMatrix(result));
        labelMin = Math.min(labelMin, 0);

        int labelMax = Math.max(MathUtils.maxMatrix(annoImage),MathUtils.maxMatrix(result));
        labelMax = Math.max(labelMax, LABEL_NUM);

        int[] labels = new int[labelMax - labelMin];
        for(int i = 0; i < labels.length; i++) {
            labels[i] = labelMin++;
        }
        return MathUtils.getConfusionMat(annoImage, result, labels,conf );
    }

    @Override
    public void setResult(Result result) {
        SegmentationResult sgresult = (SegmentationResult) result;
        sgresult.setGlobalAcc(GlobalAcc);
        sgresult.setMeanIOU(MeanIOU);
        sgresult.setMeanAccuracy(MeanAccuracy);
        sgresult.setMeanPrecision(MeanPrecision);
        sgresult.setMeanF1Score(MeanF1Score);

    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }
}
