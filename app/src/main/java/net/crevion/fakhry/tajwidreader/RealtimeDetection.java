package net.crevion.fakhry.tajwidreader;

import android.Manifest;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RealtimeDetection extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public static final int CAMERA_PERMISSION_REQUEST_CODE = 3;
    private static String TAG = "MainActivity";
    JavaCameraView javaCameraView;
    private Mat mRgba, imgGray, imgCanny, imgBiner, imgMat, hierarchy, contourMat;
    private List<MatOfPoint> contourList;
    private Random random;
    // to fix camera orientation
//    Mat mRgbaT, mRgbaF;
    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    javaCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    static {

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_realtime_detection);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},CAMERA_PERMISSION_REQUEST_CODE);
        }
        javaCameraView = (JavaCameraView)findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView!=null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView!=null)
            javaCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.i(TAG, "Open cv Loaded success");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else{
            Log.i(TAG, "Open cv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
//        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
//        mRgbaT = new Mat(height, width, CvType.CV_8UC4);
        imgGray = new Mat(height, width, CvType.CV_8UC1);
        imgCanny = new Mat(height, width, CvType.CV_8UC1);
        imgMat= new Mat(height, width, CvType.CV_8UC1);
        hierarchy = new Mat();
        contourMat = new Mat();
        random = new Random();
        contourList = new ArrayList<MatOfPoint>();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        //to rotate mRgba 90 degrees when portait
//        Core.transpose(mRgba, mRgbaT);
//        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(),0,0,0);
//        Core.flip(mRgbaF, mRgba, 1);
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2GRAY);
//        Imgproc.GaussianBlur(imgGray, imgGray, new Size(3,3),0);
//        Imgproc.threshold(imgGray, imgGray, 150, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.Canny(imgGray, imgCanny, 10, 100);
        Imgproc.findContours(imgCanny,contourList, hierarchy, Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
        contourMat.create(imgCanny.rows(), imgCanny.cols(), CvType.CV_8UC3);
        for(int i = 0; i < contourList.size(); i++)
        {
            Imgproc.drawContours(contourMat
                    ,contourList,i,new Scalar(random.nextInt(255)
                            ,random.nextInt(255),random.nextInt(255)), -1);
        }
        return contourMat;
    }
}
