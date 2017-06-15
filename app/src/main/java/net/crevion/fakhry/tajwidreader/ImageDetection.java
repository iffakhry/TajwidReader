package net.crevion.fakhry.tajwidreader;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ImageDetection extends AppCompatActivity {

    private Button captureButton;
    private ImageView imageView;
    private Bitmap currentBitmap, originalBitmap;
    private int imgHeight, imgWidth;
    Mat originalMat, currentMat;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
//                    originalMat=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);

        captureButton = (Button)findViewById(R.id.captureButton);
        imageView = (ImageView) findViewById(R.id.image_view);

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK, Uri.parse("content://media/internal/images/media"));
                startActivityForResult(intent, 0);
            }
        });

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
//        return super.onCreateOptionsMenu(menu);
        getMenuInflater().inflate(R.menu.menu_image, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()){
            case R.id.grayscaleButton:
                grayScale();
                return true;
            case R.id.binerButton:
//                Toast.makeText(this, "biner", Toast.LENGTH_SHORT).show();
                binaryInvers();
                return true;
            case R.id.DifferenceGaussianButton:
                DifferenceOfGaussian();
                return true;
            case R.id.ContourButton:
                contour();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 0 && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = {MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();
// String picturePath contains the path of selected Image
//To speed up loading of image
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 2;
            Bitmap temp = BitmapFactory.decodeFile(picturePath, options);
//Get orientation information
            int orientation = 0;
            try {
                ExifInterface imgParams = new ExifInterface(picturePath);
                orientation =
                        imgParams.getAttributeInt(
                                ExifInterface.TAG_ORIENTATION,
                                ExifInterface.ORIENTATION_UNDEFINED);
            } catch (IOException e) {
                e.printStackTrace();
            }
//Rotating the image to get the correct orientation
            Matrix rotate90 = new Matrix();
            rotate90.postRotate(orientation);
//           originalBitmap = rotateBitmap(temp,orientation);
            originalBitmap = temp;
//Convert Bitmap to Mat
            Bitmap tempBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888,true);
            imgHeight = tempBitmap.getHeight();
            imgWidth = tempBitmap.getWidth();
           originalMat = new Mat(tempBitmap.getHeight(), tempBitmap.getWidth(), CvType.CV_8U);
            Utils.bitmapToMat(tempBitmap, originalMat);
            currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888,false);
            currentMat = originalMat;
            loadImageToImageView();
//            DifferenceOfGaussian();
            }
    }

    public void DifferenceOfGaussian()
    {
        Mat grayMat = new Mat();
        Mat blur1 = new Mat();
        Mat blur2 = new Mat();
            //Converting the image to grayscale
            Imgproc.cvtColor(originalMat
                    ,grayMat,Imgproc.COLOR_BGR2GRAY);
            //Bluring the images using two different blurring radius
            Imgproc.GaussianBlur(grayMat,blur1,new Size(15,15),5);
            Imgproc.GaussianBlur(grayMat,blur2,new Size(21,21),5);
            //Subtracting the two blurred images
            Mat DoG = new Mat();
            Core.absdiff(blur1, blur2,DoG);
            //Inverse Binary Thresholding
            Core.multiply(DoG,new Scalar(100), DoG);
            Imgproc.threshold(DoG,DoG,50,255
                    ,Imgproc.THRESH_BINARY_INV);
            //Converting Mat back to Bitmap
            Utils.matToBitmap(DoG, currentBitmap);
            loadImageToImageView();

    }

    private void contour(){
        Mat grayMat = new Mat();
        Mat cannyEdges = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contourList = new
                ArrayList<MatOfPoint>();
        //A list to store all the contours
        //Converting the image to grayscale
        Imgproc.cvtColor(originalMat,grayMat
                ,Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(grayMat, cannyEdges,10, 100);
        //finding contours
        Imgproc.findContours(cannyEdges,contourList
                ,hierarchy,Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);
        //Drawing contours on a new image
        Mat contours = new Mat();
        contours.create(cannyEdges.rows()
                ,cannyEdges.cols(),CvType.CV_8UC3);
        Random r = new Random();
        for(int i = 0; i < contourList.size(); i++)
        {
            Imgproc.drawContours(contours
                    ,contourList,i,new Scalar(r.nextInt(255)
                            ,r.nextInt(255),r.nextInt(255)), -1);
        }
        //Converting Mat back to Bitmap
        Utils.matToBitmap(contours, currentBitmap);
        loadImageToImageView();
    }

    private void grayScale(){
        Mat grayMat = new Mat();
        Imgproc.cvtColor(originalMat
                ,grayMat,Imgproc.COLOR_BGR2GRAY);
        Utils.matToBitmap(grayMat, currentBitmap);
        currentMat = grayMat;
        loadImageToImageView();
    }

    private void binaryInvers(){
        Mat binInvMat = new Mat();
        Imgproc.threshold(currentMat, binInvMat, 100, 255, Imgproc.THRESH_BINARY_INV);
        Utils.matToBitmap(binInvMat, currentBitmap);
        currentMat = binInvMat;
        loadImageToImageView();

    }

    private void loadImageToImageView() {
        ImageView imgView = (ImageView) findViewById(R.id.image_view);
        imgView.setImageBitmap(currentBitmap);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.i("OpenCV", "Open cv Loaded success");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else{
            Log.i("OpenCV", "Open cv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        }
//        super.onResume();
//        if (!OpenCVLoader.initDebug()) {
//            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
//        } else {
//            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
    }
}
