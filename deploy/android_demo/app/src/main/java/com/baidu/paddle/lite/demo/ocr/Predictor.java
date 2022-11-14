package com.baidu.paddle.lite.demo.ocr;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Point;
import android.telephony.SmsManager;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import static android.graphics.Color.*;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    public int cpuThreadNum = 4;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String modelName = "";
    protected OCRPredictorNative paddlePredictor = null;
    protected float inferenceTime = 0;
    // Only for object detection
    protected Vector<String> wordLabels = new Vector<String>();
    protected int detLongSize = 960;
    protected float scoreThreshold = 0.1f;
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected volatile String outputResult = "";
    protected float postprocessTime = 0;


    public Predictor() {
    }

    public boolean init(Context appCtx, String modelPath, String labelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode) {
        isLoaded = loadModel(appCtx, modelPath, useOpencl, cpuThreadNum, cpuPowerMode);
        if (!isLoaded) {
            return false;
        }
        isLoaded = loadLabel(appCtx, labelPath);
        return isLoaded;
    }


    public boolean init(Context appCtx, String modelPath, String labelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode, int detLongSize, float scoreThreshold) {
        boolean isLoaded = init(appCtx, modelPath, labelPath, useOpencl, cpuThreadNum, cpuPowerMode);
        if (!isLoaded) {
            return false;
        }
        this.detLongSize = detLongSize;
        this.scoreThreshold = scoreThreshold;
        return true;
    }

    protected boolean loadModel(Context appCtx, String modelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode) {
        // Release model if exists
        releaseModel();

        // Load model
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // Read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }

        OCRPredictorNative.Config config = new OCRPredictorNative.Config();
        config.useOpencl = useOpencl;
        config.cpuThreadNum = cpuThreadNum;
        config.cpuPower = cpuPowerMode;
        config.detModelFilename = realPath + File.separator + "det_db.nb";
        config.recModelFilename = realPath + File.separator + "rec_crnn.nb";
        config.clsModelFilename = realPath + File.separator + "cls.nb";
        Log.i("Predictor", "model path" + config.detModelFilename + " ; " + config.recModelFilename + ";" + config.clsModelFilename);
        paddlePredictor = new OCRPredictorNative(config);

        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public void releaseModel() {
        if (paddlePredictor != null) {
            paddlePredictor.destory();
            paddlePredictor = null;
        }
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }

    protected boolean loadLabel(Context appCtx, String labelPath) {
        wordLabels.clear();
        wordLabels.add("black");
        // Load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
            }
            wordLabels.add(" ");
            Log.i(TAG, "Word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }


    public boolean runModel(int run_det, int run_cls, int run_rec) {
        if (inputImage == null || !isLoaded()) {
            return false;
        }

        // Warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.runImage(inputImage, detLongSize, run_det, run_cls, run_rec);
        }
        warmupIterNum = 0; // do not need warm
        // Run inference
        Date start = new Date();
        ArrayList<OcrResultModel> results = paddlePredictor.runImage(inputImage, detLongSize, run_det, run_cls, run_rec);
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        results = postprocess(results);
        Log.i(TAG, "[stat] Inference Time: " + inferenceTime + " ;Box Size " + results.size());
        drawResults(results);

        return true;
    }

    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    public String modelPath() {
        return modelPath;
    }

    public String modelName() {
        return modelName;
    }

    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    public String cpuPowerMode() {
        return cpuPowerMode;
    }

    public float inferenceTime() {
        return inferenceTime;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public float postprocessTime() {
        return postprocessTime;
    }


    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }

        this.inputImage = image.copy(Bitmap.Config.ARGB_8888, true);
    }

    private ArrayList<OcrResultModel> postprocess(ArrayList<OcrResultModel> results) {
        for (OcrResultModel r : results) {
            StringBuffer word = new StringBuffer();
            for (int index : r.getWordIndex()) {
                if (index >= 0 && index < wordLabels.size()) {
                    word.append(wordLabels.get(index));
                } else {
                    Log.e(TAG, "Word index is not in label list:" + index);
                    word.append("×");
                }
            }
            r.setLabel(word.toString());
            r.setClsLabel(r.getClsIdx() == 1 ? "180" : "0");
        }
        return results;
    }

    public boolean isNowSendMsgHours(Calendar calendar, List<Integer> sendMsgHours) {
        int current_hour = calendar.get(Calendar.HOUR_OF_DAY);

        for (int i = 0; i < sendMsgHours.size(); i++) {
            if (current_hour == sendMsgHours.get(i)) {
                return true;
            }
        }

        return false;
    }

    private String getLogFileNameHelper(Date date, String type, int minute) {
        String d = new SimpleDateFormat("yyyy-MM-dd-HH").format(date);

        return "/sdcard/充电/" + d + "-" + type + "-" + minute + ".log";
    }

    public int getExistLogFileCount(Date date, String s) {
        int res = 0;
        for (int minute = 0; minute < 60; minute++) {
//            /sdcard/充电/2022-11-15-01-success-23.log
            String filename = getLogFileNameHelper(date, s, minute);
            if (isFileExist(filename)) {
                res++;
            }
        }
        return res;
    }


    public String getTypeByChargingState(boolean isCharging) {
        if (isCharging) {
            return "success";
        } else {
            return "failure";
        }
    }

    public void createLogFile(Date date, boolean isCharging) {
        try {
            String s = getTypeByChargingState(isCharging);
            String minute = new SimpleDateFormat("m").format(date);
            String log_filename = getLogFileNameHelper(date, s, Integer.parseInt(minute));
            File f = new File(log_filename);
            if (f.createNewFile()) {
                System.out.println("File created");
            } else {
                System.out.println("File already exists");
            }
        } catch (Exception e) {
        }
    }

    public static boolean isInteger(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            int d = Integer.parseInt(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    public List<Integer> readIntegerLines(String filePath) {
        List<Integer> res = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // process the line.
                line = line.trim();
                if (isInteger(line)) {
                    res.add(Integer.parseInt(line));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }

    public List<String> readStringLines(String filePath) {
        List<String> res = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // process the line.
                line = line.trim();
                if (!line.isEmpty()) {
                    res.add(line);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }

    public List<Integer> getSendMessageHours(String filePath) {
        return readIntegerLines(filePath);
    }

    public List<String> getPhoneListFromFile(String filePath) {
        return readStringLines(filePath);
    }


    public boolean isFileExist(String filepath) {
        File f = new File(filepath);
        if (f.isFile()) {
            return true;
        } else {
            return false;
        }
    }


    public boolean isSendMsg(Calendar calendar, Date date, boolean isCharging, List<Integer> sendMessageHours) {
        if (!isNowSendMsgHours(calendar, sendMessageHours)) {
            //现在不是发短信的时间
            return false;
        }
        //以下是可以发送短信的时间

        if (isCharging) {
            //充电中，仅在每小时的50~59发短信
            int current_minute = calendar.get(Calendar.MINUTE);
            if (current_minute < 50) {
                // 0~49 不发送短信
                return false;
            }
        }
        String type = getTypeByChargingState(isCharging);
        int logFileCount = getExistLogFileCount(date, type);
        if (isCharging) {
            if (logFileCount == 0) {
                //第一次成功，发送短信
                createLogFile(date, true);
                return true;
            } else if (logFileCount >= 1) {
                return false;
            } else {
                return false;
            }
        } else {
            if (logFileCount == 0) {
                //第一次失败，只创建文件，不发送短信
                createLogFile(date, false);
                return false;
            } else if (logFileCount == 1) {
                //进入这里， 说明之前已经失败了一次，这是第二次失败，要发送短信
                createLogFile(date, false);
                return true;
            } else if (logFileCount == 2) {
                return false;
            } else {
                return false;
            }
        }
    }

    public void sendMessageToPhoneList(List<String> phoneList, String message) {
        if (message.isEmpty()) {
            return;
        }
        if (phoneList == null) {
            return;
        }
        if (phoneList.size() == 0) {
            return;
        }
        for (String phone : phoneList) {
            sendSMS(phone, message);
        }
    }

    private void drawResults(ArrayList<OcrResultModel> results) {

        List<String> keywords = readStringLines("/sdcard/充电/keywords.conf");
        if (keywords.size() == 0) {
            keywords.add("车辆");
            keywords.add("向盘");
            keywords.add("功率");
            keywords.add("否则");
            keywords.add("离开");
            keywords.add("满时");
        }

        StringBuffer outputResultSb = new StringBuffer("");
        int match_count = 0;

        for (int i = 0; i < results.size(); i++) {
            OcrResultModel result = results.get(i);
            StringBuilder sb = new StringBuilder("");
            if (result.getPoints().size() > 0) {
                sb.append("Det: ");
                for (Point p : result.getPoints()) {
                    sb.append("(").append(p.x).append(",").append(p.y).append(") ");
                }
            }
            if (result.getLabel().length() > 0) {
                String sentence = result.getLabel();
                for (String keyword : keywords) {
                    if (sentence.contains(keyword)) {
                        match_count++;
                    }
                }
                sb.append("\n Rec: ").append(result.getLabel());
                sb.append(",").append(result.getConfidence());
            }
            if (result.getClsIdx() != -1) {
                sb.append(" Cls: ").append(result.getClsLabel());
                sb.append(",").append(result.getClsConfidence());
            }
            Log.i(TAG, sb.toString()); // show LOG in Logcat panel
            outputResultSb.append(i + 1).append(": ").append(sb.toString()).append("\n");
        }
        Calendar calendar = Calendar.getInstance();
        Date date = calendar.getTime();

        List<Integer> sendMessageHours = getSendMessageHours("/sdcard/充电/可以发送短信的小时数.conf");
        if (sendMessageHours.size() == 0) {
            //没有配置的话， 一般21点，22点，23点，0点，1点还没有睡
            sendMessageHours.add(21);
            sendMessageHours.add(22);
            sendMessageHours.add(23);
            sendMessageHours.add(0);
            sendMessageHours.add(1);
        }

        boolean isCharging = match_count > 0;
        boolean isSend = isSendMsg(calendar, date, isCharging, sendMessageHours);

        if (isSend) {
            String t = new SimpleDateFormat("yyyy-MM-dd HH:mm").format(date);
            String msgContent;
            String phoneListFilePath;

            if (isCharging) {
                msgContent = "车正常在充电，请放心。" + t;
                phoneListFilePath = "/sdcard/充电/成功手机号列表.conf";
                sendSMS("10086", "HFSJLL");//恢复数据流量，免费短信
            } else {
                msgContent = "充电停了，去看看吧。如果确认已完成充电，请忽略！！" + t;
                phoneListFilePath = "/sdcard/充电/失败手机号列表.conf";
                sendSMS("10086", "ZTSJLL"); //暂停数据流量，免费短信
            }
            List<String> phoneList = getPhoneListFromFile(phoneListFilePath);

            sendMessageToPhoneList(phoneList, msgContent);
        }


        outputResult = outputResultSb.toString();
        outputImage = inputImage;
        Canvas canvas = new Canvas(outputImage);
        Paint paintFillAlpha = new Paint();
        paintFillAlpha.setStyle(Paint.Style.FILL);
        paintFillAlpha.setColor(Color.parseColor("#3B85F5"));
        paintFillAlpha.setAlpha(50);

        Paint paint = new Paint();
        paint.setColor(Color.parseColor("#3B85F5"));
        paint.setStrokeWidth(5);
        paint.setStyle(Paint.Style.STROKE);

        for (OcrResultModel result : results) {
            Path path = new Path();
            List<Point> points = result.getPoints();
            if (points.size() == 0) {
                continue;
            }
            path.moveTo(points.get(0).x, points.get(0).y);
            for (int i = points.size() - 1; i >= 0; i--) {
                Point p = points.get(i);
                path.lineTo(p.x, p.y);
            }
            canvas.drawPath(path, paint);
            canvas.drawPath(path, paintFillAlpha);
        }
    }

    public void sendSMS(String phoneNumber, String message) {
        //获取短信管理器
        SmsManager smsManager = SmsManager.getDefault();
        //拆分短信内容（短信长度显示）
        List<String> divideContents = smsManager.divideMessage(message);
        for (String text : divideContents) {
            smsManager.sendTextMessage(phoneNumber, null, text, null, null);
        }
    }

}
