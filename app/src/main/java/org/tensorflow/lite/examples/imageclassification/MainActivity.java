/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification;


import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageProxy;

import org.tensorflow.lite.examples.imageclassification.databinding.ActivityMainBinding;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

/** Entrypoint for app */
public class MainActivity extends AppCompatActivity {
    private static int PICK_PHOTO_FOR_AVATAR = 9;
    private Button browseFileButton;
    private ImageView imagePreview;
    private LinearLayout classification_results_container;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LayoutInflater li = LayoutInflater.from(this);
        View view = li.inflate(R.layout.activity_main2, null);

        setContentView(view);
        initializeViews();
    }

    private void initializeViews() {
        browseFileButton = findViewById(R.id.browse_button);
        browseFileButton.setOnClickListener(v -> {
            pickImage();
        });
        classification_results_container = findViewById(R.id.classification_results_container);
        imagePreview = findViewById(R.id.image_preview);
    }

    private void setNewResults(List<Classifications> results) {
        classification_results_container.removeAllViews();
        for (Classifications result : results) {
            List<Category> categories = result.getCategories();
            for (Category category : categories) {
                String name = category.getLabel();
                float score = category.getScore()*100;
                DecimalFormat df = new DecimalFormat();
                df.setMaximumFractionDigits(3);
                TextView tv = new TextView(this);
                tv.setText(name+": "+df.format(score)+"%");
                tv.setGravity(Gravity.CENTER);
                tv.setTextSize(TypedValue.COMPLEX_UNIT_SP,28);
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
                tv.setLayoutParams(lp);
                classification_results_container.addView(tv);
                Log.d("supa",category.getLabel());
            }

        }
    }

    public void pickImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_PHOTO_FOR_AVATAR);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_PHOTO_FOR_AVATAR && resultCode == Activity.RESULT_OK) {
            if (data == null) {
                //Display an error
                return;
            }
            try {
                InputStream inputStream = this.getContentResolver().openInputStream(data.getData());
                // String name = data.getData().getPath();
                Bitmap bmp = BitmapFactory.decodeStream(inputStream);

                imagePreview.setImageBitmap(bmp);

                ImageClassifier.ImageClassifierOptions.Builder optionsBuilder =
                        ImageClassifier.ImageClassifierOptions.builder()
                                .setScoreThreshold(0f)
                                .setMaxResults(3);
                ImageClassifier imageClassifier =
                        ImageClassifier.createFromFileAndOptions(
                                this,
                                "alwa_new_quant_metadata.tflite",
                                optionsBuilder.build());
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder().add(new Rot90Op(-0 / 90)).build();
                TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(bmp));

                List<Classifications> results = imageClassifier.classify(tensorImage);

                setNewResults(results);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
