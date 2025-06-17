package com.qualcomm.qti.psnpedemo.post;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import com.google.gson.Gson;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

public class CreateRaw {
    private final String TAG = "CreateRaw";
    private final String SAVE_DIR = "preData/out/";
    private final String INPUT_DIR = "preData/input/";

    private String getSaveDirPath(String modelName){
        return BenchmarkApplication.getExternalDirPath(SAVE_DIR + modelName);
    }

    private String getInputDirPath(String modelName){
        return BenchmarkApplication.getExternalDirPath(INPUT_DIR + modelName);
    }

    public void createMobileDataRaw(){
        createMobileBertRaw(10);
        createVDSRRaw();
    }

    public void createVDSRRaw(){
        String saveDirPath = getSaveDirPath("VDSR");
        String imageDirPath = getInputDirPath("VDSR");
        Log.i(TAG,"VDSR预处理数据保存地址:" + saveDirPath);
        File imageDir = new File(imageDirPath);
        File[] imageList;
        try {
            imageList = imageDir.listFiles(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String filename) {
                    return filename.toLowerCase().endsWith(".jpg") ||
                            filename.toLowerCase().endsWith(".jpeg") ||
                            filename.toLowerCase().endsWith(".bmp") ||
                            filename.toLowerCase().endsWith(".png") ;
                }
            });
            if (imageList == null){
                return;
            }
            for (File file: imageList){
                for (int i = 2; i < 5; i++){
                    float[] pixelsYFloat = getYLowDataRaw(file, i);
                    String fileName = file.getName().split("\\.")[0] + "_" + i + ".raw";
                    writeArrayToRawFile(saveDirPath + "/" + fileName, pixelsYFloat);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "vdsr预处理数据生成失败： " + e.getMessage());
            e.printStackTrace();
        }
    }

    private float[] getYLowDataRaw(File imageName, int lowRatio){
        float[] pixelsYFloat = null;
        try{
            Bitmap imgRGB = BitmapFactory.decodeFile(imageName.getAbsolutePath());

            int originImgWidth = imgRGB.getWidth();
            int originImgHeight = imgRGB.getHeight();
            int startX = Math.max((int)(originImgWidth - 256) / 2, 0);
            int startY = Math.max((int)(originImgHeight - 256) / 2, 0);
            int width = Math.min(originImgWidth, 256);
            int height = Math.min(originImgHeight, 256);

            Bitmap newImage = Bitmap.createBitmap(imgRGB, startX, startY, width, height, null, false);

            float R, G, B, Y;
            float[][] MatrixY = new float[height][width];
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int pixel = newImage.getPixel(col, row);
                    R = Color.red(pixel);
                    G = Color.green(pixel);
                    B = Color.blue(pixel);
                    /*convert RGB to YCbCr
                     * convert formula:
                     * Y = (0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0
                     * Cb = (-0.148223 * R - 0.290992 * G + 0.439215 * B + 128.0)/255.0
                     * Cr = (0.439215 * R  - 0.367789 * G - 0.071426 * B + 128.0)/255.0
                     * We only use Y channel here.
                     * */
                    Y = (float)((0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0);
                    MatrixY[row][col] = Y;
                }
            }

            if(width < 256 || height < 256){
                /*if input img size is smaller than 256*256, adjust it to 256*256*/
                MatrixY = resizeInsertLinear(MatrixY, 256, 256);
                width = 256;
                height = 256;
            }

            pixelsYFloat = new float[width * height * 1];
            int i = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixelsYFloat[i++] = MatrixY[row][col];
                }
            }
            int resize_width = width / lowRatio;
            int resize_height = height / lowRatio;

            float[][] resizeTmp = resizeInsertLinear(MatrixY, resize_width, resize_height);
            float[][] finalImg = resizeInsertLinear(resizeTmp, width, height);

            i = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixelsYFloat[i++] = finalImg[row][col];
                }
            }
        }catch (Exception e){
            Log.e(TAG, "Exception in image pre-processing: "+ e);
        }
        return pixelsYFloat;
    }

    private float[][] resizeInsertLinear(float [][] src, int dstWidth, int dstHeight){
        if(null == src){
            throw new IllegalArgumentException("src buffer is null");
        }
        if(0 == src.length || 0 == src[0].length){
            throw new IllegalArgumentException(String.format("Wrong resize src buffer size", dstWidth, dstHeight));
        }
        if(0 == dstWidth || 0 == dstHeight){
            throw new IllegalArgumentException(String.format("Wrong resize dstSize(%d, %d)", dstWidth, dstHeight));
        }

        int srcHeight = src.length;
        int srcWidth = src[0].length;
        float[][] dst = new float[dstHeight][dstWidth];

        double scaleX = (double)srcWidth / (double)dstWidth;
        double scaleY = (double)srcHeight / (double)dstHeight;

        for(int dstY = 0; dstY < dstHeight; ++dstY){
            double fy = ((double)dstY + 0.5) * scaleY - 0.5;
            int sy = (int)fy;
            fy -= sy;
            if(sy < 0){
                fy = 0.0; sy = 0;
            }
            if(sy >= srcHeight - 1){
                fy = 0.0; sy = srcHeight - 2;
            }

            for(int dstX = 0; dstX < dstWidth; ++dstX){
                double fx = ((double)dstX + 0.5) * scaleX - 0.5;
                int sx = (int)fx;
                fx -= sx;
                if(sx < 0){
                    fx = 0.0; sx = 0;
                }
                if(sx >= srcWidth - 1){
                    fx = 0.0; sx = srcWidth - 2;
                }

                dst[dstY][dstX] = (float) ((1.0-fx) * (1.0-fy) * src[sy][sx]
                        + fx * (1.0-fy) * src[sy][sx+1]
                        + (1.0-fx) * fy * src[sy+1][sx]
                        + fx * fy * src[sy+1][sx+1]);
            }
        }

        return dst;
    }

    public void createMobileBertRaw(int count){
        String saveDirPath = getSaveDirPath("MobileBert");
        String vocabPath = getInputDirPath("MobileBert") + "/vocab.txt";
        String dataPath = getInputDirPath("MobileBert") + "/dev-v1.1.json";
        Log.i(TAG,"MobileBert预处理数据保存地址:" + saveDirPath);
        try {
            MobileBertUtil mobileBertUtil = new MobileBertUtil(vocabPath);

            Gson gson = new Gson();
            File file = new File(dataPath);
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
                if (curPosition == count) {
                    break;
                }
                for (MobileBertUtil.Question.Paragraphs paragraphs : item.getParagraphs()) {
                    if (curPosition == count) {
                        break;
                    }
                    for (MobileBertUtil.Question.Qas qas : paragraphs.getQas()) {
                        String content = paragraphs.getContext();
                        MobileBertUtil.Feature feature = mobileBertUtil.getFeature(qas.getQuestion(), content);
                        float[] input_expandDims = new float[384];
                        float[] input_mask = new float[384];
                        float[] segment_ids = new float[384];
                        for (int j = 0; j < MobileBertUtil.MAX_SEQ_LEN; j++) {
                            input_expandDims[j] = feature.inputIds[j];
                            input_mask[j] = feature.inputMask[j];
                            segment_ids[j] = feature.segmentIds[j];
                        }
                        String saveRawPath = saveDirPath + "/" + curPosition + "_";
                        writeArrayToRawFile(saveRawPath + "input_ids.raw", input_expandDims);
                        writeArrayToRawFile(saveRawPath + "input_mask.raw", input_mask);
                        writeArrayToRawFile(saveRawPath + "segment_ids.raw", segment_ids);
                        curPosition++;
                        if (curPosition == count) {
                            break;
                        }
                    }
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private static byte[] getByte(float values) {
        byte[] result = new byte[4];
        int data = Float.floatToIntBits(values);
        result[0] = (byte)(data & 0xff);
        result[1] = (byte)(data >> 8 & 0xff);
        result[2] = (byte)(data >> 16 & 0xff);
        result[3] = (byte)(data >> 24 & 0xff);
        return result;
    }

    private static void writeArrayToJsonFile(String filePath, float[] data){
        try {
            FileOutputStream os = new FileOutputStream(filePath);
            os.write(new Gson().toJson(data).getBytes());
            os.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    private static void writeArrayToRawFile(String filePath, float[] data){
        try {
            FileOutputStream os = new FileOutputStream(filePath);
            ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4);
            for (float datum : data) {
                byteBuffer.put(getByte(datum));
            }
            os.write(byteBuffer.array());
            os.close();
            String newFilePath = filePath.replace(".raw", ".json");
            writeArrayToJsonFile(newFilePath, data);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    private static float[] getfloatValues(byte[] bytes) {
        float[] floatArray = new float[bytes.length / 4];
        for (int i = 0; i < bytes.length; i += 4) {
            floatArray[i/4] = Float.intBitsToFloat((0xff & bytes[i]) | (0xff00 & (bytes[i+1] << 8))
                    | (0xff0000 & (bytes[i+2] << 16)) | (0xff000000 & (bytes[i+3] << 24)));
        }
        return floatArray;
    }
}
