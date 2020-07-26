package com.example.imask.ui.dashboard;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;

import com.example.imask.MainActivity;
import com.example.imask.R;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static android.app.Activity.RESULT_OK;

public class DashboardFragment extends Fragment {

    private DashboardViewModel dashboardViewModel;
    View root;
    private static int RESULT_LOAD_IMAGE = 1;
    Button buttonLoadImage,detectButton;
    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        dashboardViewModel =
                ViewModelProviders.of(this).get(DashboardViewModel.class);
        root = inflater.inflate(R.layout.fragment_dashboard, container, false);
        buttonLoadImage = (Button) root.findViewById(R.id.button);
        detectButton = (Button) root.findViewById(R.id.detect);

    buttonLoadImage.setOnClickListener(new View.OnClickListener() {

        @Override
        public void onClick(View arg0) {
            TextView textView = root.findViewById(R.id.result_text);
            textView.setText("");
            Intent i = new Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

            startActivityForResult(i, RESULT_LOAD_IMAGE);


        }
    });

        detectButton.setOnClickListener(new View.OnClickListener() {

        @Override
        public void onClick(View arg0) {

            Bitmap bitmap = null;
            Module module = null;

            //Getting the image from the image view
            ImageView imageView = (ImageView) root.findViewById(R.id.image);

            try {
                //Read the image as Bitmap
                bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();

                //Here we reshape the image into 400*400
                bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

                //Loading the model file.
                module = Module.load(fetchModelFile(getActivity(), "resnet18_traced.pt"));
            } catch (IOException e) {
//                finish();
            }

            //Input Tensor
            final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB
            );

            //Calling the forward of the model to run our input
            final Tensor output = module.forward(IValue.from(input)).toTensor();


            final float[] score_arr = output.getDataAsFloatArray();

            // Fetch the index of the value with maximum score
            float max_score = -Float.MAX_VALUE;
            int ms_ix = -1;
            for (int i = 0; i < score_arr.length; i++) {
                if (score_arr[i] > max_score) {
                    max_score = score_arr[i];
                    ms_ix = i;
                }
            }

            //Fetching the name from the list based on the index
            String detected_class = ModelClasses.MODEL_CLASSES[ms_ix];

            //Writing the detected class in to the text view of the layout
            TextView textView = root.findViewById(R.id.result_text);
            textView.setText(detected_class);


        }
    });
        return root;

}
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getActivity().getContentResolver().query(selectedImage,filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) root.findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);


        }


    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }

    }

}

