// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model should have float input
 *
 *
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<std::int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

std::vector<float> softmax(const std::vector<float>& logits) {
  std::vector<float> probabilities(logits.size());
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum_exp = 0.0f;

  for (size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(logits[i] - max_logit);
    sum_exp += probabilities[i];
  }

  for (size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] /= sum_exp;
  }

  return probabilities;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

#ifdef _WIN32
int wmain(int argc, ORTCHAR_T* argv[]) {
#else
int main(int argc, ORTCHAR_T* argv[]) {
#endif
  //if (argc != 2) {
  //  std::cout << "Usage: ./onnx-api-example <onnx_model.onnx>" << std::endl;
  //  return -1;
  //}

  //std::basic_string<ORTCHAR_T> model_file = argv[1];

  // onnxruntime setup

 /* 
    Input Node Name /
      Shape(0)
      : data : -1x3x224x224 Output Node Name /
               Shape(0)
      : resnetv15_dense0_fwd : -1x1000
      */

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Session session = Ort::Session(env, LR"(sam_vit_b_merge.onnx)", session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetInputCount(); i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
  }




  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes;

  std::vector<Ort::Value> input_tensors;
    /*
  Input Node Name/Shape (0):
        input_image : 3x-1x-1
        point_coords : -1x2
        point_labels : -1
Output Node Name/Shape (0):
        masks : -1x-1x-1x-1
        iou_predictions : -1x1
  */
  //srcData 
  cv::Mat img = cv::imread(R"(truck.jpg)",
      cv::IMREAD_ANYCOLOR);
  std::vector<cv::Point> point_coords{ {575,750} };
  std::vector<int>point_label{0};
  std::vector<cv::Rect> rects{
      cv::Rect(cv::Point(425,600),cv::Point(700,875))
  };
  //format srcData
  cv::Mat input_image;
  std::vector<std::int64_t> input_image_shape, point_coords_shape, point_labels_shape;
  input_image = img.reshape(1, img.total()).t();
  input_image.convertTo(input_image, CV_32F);
  std::vector<float> point_coords_data;
  std::vector<float>point_label_data;
  for (int i = 0; i < point_coords.size(); i++) {
	  point_coords_data.push_back(point_coords[i].x);
	  point_coords_data.push_back(point_coords[i].y);
      point_label_data.push_back(point_label[i]);
  }
  for (int i = 0; i < rects.size(); i++) {
	  point_coords_data.push_back(rects[i].tl().x);
	  point_coords_data.push_back(rects[i].tl().y);
      point_label_data.push_back(2);
	  point_coords_data.push_back(rects[i].br().x);
	  point_coords_data.push_back(rects[i].br().y);
      point_label_data.push_back(3);
  }

  //dynamic shape
  input_image_shape = { 3, img.rows, img.cols };
  point_coords_shape = { (int64_t)point_coords_data.size()/2, 2};
  point_labels_shape = { (int64_t)point_label_data.size() };

   Ort::MemoryInfo mem_info =
       Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, (float*)input_image.data, img.total() * 3,
                                                           input_image_shape.data(), input_image_shape.size()));
  input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, point_coords_data.data(), point_coords_data.size(),
	  point_coords_shape.data(), point_coords_shape.size()));
  input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, point_label_data.data(), point_label_data.size(), point_labels_shape.data(), point_labels_shape.size()));

  // pass data through model
  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::cout << "Running model..." << std::endl;
  try {
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());
    
    cv::Mat singleMask,singleMaskBinary;
    float defaultThreshold = 0.;
    float* floatarr = output_tensors.front().GetTensorMutableData<float>(); 
    singleMask = cv::Mat(img.size(), CV_32F, floatarr);
    float* singleMask_p = (float*)singleMask.data;
    cv::Vec3b* img_p = (cv::Vec3b*)img.data;
    for (int i = 0; i < singleMask.total(); i++)
    {
        if (singleMask_p[i] > 0)
        {
            img_p[i] = cv::Vec3b(0, 255, 0);
        }
    }
    

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
    assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
  } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
}
