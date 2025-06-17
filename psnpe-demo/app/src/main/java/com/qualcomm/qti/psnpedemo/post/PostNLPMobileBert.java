package com.qualcomm.qti.psnpedemo.post;

import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.processor.PostProcessor;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

public class PostNLPMobileBert extends PostProcessor {
    public PostNLPMobileBert(int imageNumber) {
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
}
